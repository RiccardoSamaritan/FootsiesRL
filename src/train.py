"""PPO self-play training on Footsies (our own PPO, see ppo.py).

Examples:
    uv run python src/train.py --total-steps 20000 --num-envs 4      # smoke test
    uv run python src/train.py --reward r2 --run-name r2_s1       # full run

Step counting: one "step" is one game step of one game instance, so an update
uses num_envs * rollout steps (= 2 * that many agent samples, two per game).
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from env import FootsiesVecEnv
from rewards import REWARDS, make_reward
from ppo import Agent, PPOConfig, compute_gae, ppo_update, save_agent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None)
    p.add_argument("--reward", default="r0", choices=REWARDS, help="training reward variant (see rewards.py)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--total-steps", type=int, default=25_000_000)
    p.add_argument("--num-envs", type=int, default=48)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--port", type=int, default=45001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ent-coef", type=float, default=None, help="default 0.025 (fixed) or 0.1 (start value with --ent-decay)")
    p.add_argument("--ent-decay", action="store_true", help="decay the entropy coefficient linearly to 0")
    p.add_argument("--checkpoint-every", type=int, default=1_000_000, help="steps")
    p.add_argument("--overwrite", action="store_true", help="reuse an existing run directory")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.reward}_lam{args.gae_lambda}_s{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    if os.path.exists(os.path.join(run_dir, "log.csv")) and not args.overwrite:
        sys.exit(f"{run_dir} already has a log.csv: choose another --run-name or pass --overwrite")
    os.makedirs(run_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ent = args.ent_coef if args.ent_coef is not None else (0.1 if args.ent_decay else 0.025)
    cfg = PPOConfig(gamma=args.gamma, gae_lambda=args.gae_lambda, lr=args.lr,
                    ent_coef=ent, ent_decay=args.ent_decay)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "ppo": asdict(cfg)}, f, indent=2)

    reward_fn, env_overrides = make_reward(args.reward, gamma=args.gamma)
    env = FootsiesVecEnv(args.num_envs, port=args.port, env_config=env_overrides, reward_fn=reward_fn)
    S, T = env.num_slots, args.rollout
    agent = Agent(env.obs_dim, env.n_actions, cfg.hidden).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    if args.wandb:
        import wandb
        wandb.init(project="footsies-rl", name=run_name, config=json.load(open(os.path.join(run_dir, "config.json"))))

    obs_buf = torch.zeros((T, S, env.obs_dim), device=device)
    act_buf = torch.zeros((T, S), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, S), device=device)
    rew_buf = torch.zeros((T, S), device=device)
    val_buf = torch.zeros((T + 1, S), device=device)
    term_buf = torch.zeros((T, S), dtype=torch.bool, device=device)
    trunc_buf = torch.zeros((T, S), dtype=torch.bool, device=device)

    steps_per_update = args.num_envs * T
    num_updates = max(args.total_steps // steps_per_update, 1)
    log_path = os.path.join(run_dir, "log.csv")
    log_file, writer = open(log_path, "w", newline=""), None
    next_ckpt = args.checkpoint_every
    steps = 0
    t_start = time.time()

    try:
        obs = torch.as_tensor(env.reset(), device=device)
        for update in range(1, num_updates + 1):
            frac = 1.0 - (update - 1) / num_updates
            if cfg.anneal_lr:
                optimizer.param_groups[0]["lr"] = frac * cfg.lr
            ent_coef = cfg.ent_coef * frac if cfg.ent_decay else cfg.ent_coef

            # Wall-clock time per phase (seconds in this update), to see where
            # the time goes: game server, policy forward, tensor copies, GAE, SGD.
            t_policy = t_env = t_copy = 0.0
            for t in range(T):
                c0 = time.perf_counter()
                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(obs)
                obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, action, logp, value
                actions_np = action.cpu().numpy()
                c1 = time.perf_counter()
                next_obs, rew, term, trunc = env.step(actions_np)
                c2 = time.perf_counter()
                rew_buf[t] = torch.as_tensor(rew, device=device)
                term_buf[t] = torch.as_tensor(term, device=device)
                trunc_buf[t] = torch.as_tensor(trunc, device=device)
                obs = torch.as_tensor(next_obs, device=device)
                steps += args.num_envs
                c3 = time.perf_counter()
                t_policy += c1 - c0
                t_env += c2 - c1
                t_copy += c3 - c2
            c0 = time.perf_counter()
            with torch.no_grad():
                val_buf[T] = agent.get_value(obs)

            adv, ret = compute_gae(rew_buf, val_buf, term_buf, trunc_buf, cfg.gamma, cfg.gae_lambda)
            t_gae = time.perf_counter() - c0
            batch = {
                "obs": obs_buf.reshape(-1, env.obs_dim), "actions": act_buf.reshape(-1),
                "logprobs": logp_buf.reshape(-1), "advantages": adv.reshape(-1),
                "returns": ret.reshape(-1), "values": val_buf[:-1].reshape(-1),
            }
            c0 = time.perf_counter()
            stats = ppo_update(agent, optimizer, batch, cfg, ent_coef)
            t_update = time.perf_counter() - c0

            eps = env.pop_finished()
            row = {"update": update, "steps": steps, "time": round(time.time() - t_start, 1),
                   "lr": optimizer.param_groups[0]["lr"], "ent_coef": ent_coef,
                   "episodes": len(eps),
                   "t_env": t_env, "t_policy": t_policy, "t_copy": t_copy,
                   "t_gae": t_gae, "t_update": t_update,
                   "ep_len": float(np.mean([e["len"] for e in eps])) if eps else float("nan"),
                   "timeout_rate": float(np.mean([e["truncated"] for e in eps])) if eps else float("nan"),
                   **stats}
            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log_file.flush()
            if args.wandb:
                wandb.log(row, step=steps)
            if update % 10 == 0 or update == 1:
                print(f"update {update}/{num_updates} steps {steps} "
                      f"{steps / (time.time() - t_start):.0f} step/s ep_len {row['ep_len']:.0f} "
                      f"timeout {row['timeout_rate']:.2f} ent {stats['entropy']:.3f} "
                      f"kl {stats['approx_kl']:.4f} ev {stats['explained_var']:.2f} | "
                      f"time/update: env {t_env:.2f}s policy {t_policy:.2f}s copy {t_copy:.2f}s "
                      f"gae {t_gae:.2f}s sgd {t_update:.2f}s", flush=True)

            if steps >= next_ckpt:
                save_agent(os.path.join(run_dir, f"ckpt_{steps}.pt"), agent, cfg, env.obs_dim, env.n_actions, steps)
                next_ckpt += args.checkpoint_every
        save_agent(os.path.join(run_dir, "final.pt"), agent, cfg, env.obs_dim, env.n_actions, steps)
    finally:
        log_file.close()
        env.close()


if __name__ == "__main__":
    main()
