# train.py
# Training of PPO in self-play on Footsies (the PPO is mine, see ppo.py).
#
# Run with commands like:
#     uv run python src/train.py --total-steps 20000 --num-envs 4      # quick test
#     uv run python src/train.py --reward r2 --run-name r2_s1          # full run
#
# How I count the steps: one "step" is one step of one game, so one update uses
# num_envs * rollout steps (and twice as many samples for the agent, because
# each game has two players and both are controlled by the same network).

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

from env import ENV_CONFIG, FootsiesVecEnv
from rewards import REWARDS, make_reward
from ppo import Agent, PPOConfig, compute_gae, ppo_update, resume_state, save_agent


def parse_args():
    """
    Command line options
    """
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
    p.add_argument("--adv-std-floor", type=float, default=0.0,
                   help="lower bound for the std used to normalize advantages (0 = off)")
    p.add_argument("--special-charge", action="store_true",
                   help="game option use_special_charge_action: 9 actions, 88-dim observation")
    p.add_argument("--action-delay", type=int, default=None,
                   help=f"game option action_delay (default {ENV_CONFIG['action_delay']})")
    p.add_argument("--ent-decay", action="store_true", help="decay the entropy coefficient linearly to 0")
    p.add_argument("--checkpoint-every", type=int, default=1_000_000, help="steps")
    p.add_argument("--overwrite", action="store_true", help="reuse an existing run directory")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run from its latest checkpoint (same arguments as the original run)")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    """
    Runs one full training: rollouts in self-play, GAE, PPO update, log and checkpoints.
    """
    args = parse_args()
    run_name = args.run_name or f"{args.reward}_lam{args.gae_lambda}_s{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    # check that we are not overwriting an old run by mistake
    if args.resume:
        if not os.path.exists(os.path.join(run_dir, "log.csv")):
            sys.exit(f"--resume: {run_dir} has no log.csv")
        if os.path.exists(os.path.join(run_dir, "final.pt")):
            sys.exit(f"--resume: {run_dir} is already complete (final.pt exists)")
    elif os.path.exists(os.path.join(run_dir, "log.csv")) and not args.overwrite:
        sys.exit(f"{run_dir} already has a log.csv: choose another --run-name or pass --overwrite")
    os.makedirs(run_dir, exist_ok=True)

    # seeds, so that a run can be repeated
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # game settings: we save them in the checkpoints, so the evaluation plays with the same rules
    game_cfg = {"action_delay": ENV_CONFIG["action_delay"] if args.action_delay is None else args.action_delay,
                "use_special_charge_action": args.special_charge}

    # entropy coefficient: 0.025 fixed, or 0.1 -> 0 for "PPO Sched."
    ent = args.ent_coef if args.ent_coef is not None else (0.1 if args.ent_decay else 0.025)
    cfg = PPOConfig(gamma=args.gamma, gae_lambda=args.gae_lambda, lr=args.lr,
                    ent_coef=ent, ent_decay=args.ent_decay,
                    adv_std_floor=args.adv_std_floor)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "ppo": asdict(cfg), "game": game_cfg}, f, indent=2)

    reward_fn, env_overrides = make_reward(args.reward, gamma=args.gamma)
    env = FootsiesVecEnv(args.num_envs, port=args.port, env_config={**env_overrides, **game_cfg}, reward_fn=reward_fn)
    S, T = env.num_slots, args.rollout  # S = 2N slots (both players), T = rollout length
    agent = Agent(env.obs_dim, env.n_actions, cfg.hidden).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    if args.wandb:
        import wandb
        wandb.init(project="footsies-rl", name=run_name, config=json.load(open(os.path.join(run_dir, "config.json"))))

    # rollout buffers, shape (T, S)
    obs_buf = torch.zeros((T, S, env.obs_dim), device=device)
    act_buf = torch.zeros((T, S), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, S), device=device)
    rew_buf = torch.zeros((T, S), device=device)
    val_buf = torch.zeros((T + 1, S), device=device)  # +1 for the bootstrap value at the end
    term_buf = torch.zeros((T, S), dtype=torch.bool, device=device)
    trunc_buf = torch.zeros((T, S), dtype=torch.bool, device=device)

    steps_per_update = args.num_envs * T
    num_updates = max(args.total_steps // steps_per_update, 1)
    log_path = os.path.join(run_dir, "log.csv")
    start_update, steps = 1, 0
    if args.resume:
        # continue from the last checkpoint (see resume_state in ppo.py)
        steps, start_update, fields = resume_state(run_dir, agent, optimizer, steps_per_update, log_path)
        log_file = open(log_path, "a", newline="")
        writer = csv.DictWriter(log_file, fieldnames=fields)
        torch.manual_seed(args.seed + start_update)  # so we don't repeat the same random numbers as the first part
    else:
        log_file, writer = open(log_path, "w", newline=""), None
    next_ckpt = (steps // args.checkpoint_every + 1) * args.checkpoint_every
    t_start = time.time()

    try:
        obs = torch.as_tensor(env.reset(), device=device)
        for update in range(start_update, num_updates + 1):
            # learning rate (and entropy, for Sched.) go down linearly to 0
            frac = 1.0 - (update - 1) / num_updates
            if cfg.anneal_lr:
                optimizer.param_groups[0]["lr"] = frac * cfg.lr
            ent_coef = cfg.ent_coef * frac if cfg.ent_decay else cfg.ent_coef

            # Also measure how much time each part takes (game server, network,
            # tensor copies, GAE, SGD) to understand where the time goes
            t_policy = t_env = t_copy = 0.0
            # 1. collect a rollout of T steps
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
            # 2. advantages with GAE
            c0 = time.perf_counter()
            with torch.no_grad():
                val_buf[T] = agent.get_value(obs)  # value of the last obs, for bootstrapping

            adv, ret = compute_gae(rew_buf, val_buf, term_buf, trunc_buf, cfg.gamma, cfg.gae_lambda)
            t_gae = time.perf_counter() - c0
            # flatten (T, S) -> (T*S) for the minibatches
            batch = {
                "obs": obs_buf.reshape(-1, env.obs_dim), "actions": act_buf.reshape(-1),
                "logprobs": logp_buf.reshape(-1), "advantages": adv.reshape(-1),
                "returns": ret.reshape(-1), "values": val_buf[:-1].reshape(-1),
            }
            # 3. PPO update
            c0 = time.perf_counter()
            stats = ppo_update(agent, optimizer, batch, cfg, ent_coef)
            t_update = time.perf_counter() - c0

            # 4. log
            eps = env.pop_finished()  # games finished during this update
            row = {"update": update, "steps": steps, "time": round(time.time() - t_start, 1),
                   "lr": optimizer.param_groups[0]["lr"], "ent_coef": ent_coef,
                   "episodes": len(eps),
                   "t_env": t_env, "t_policy": t_policy, "t_copy": t_copy,
                   "t_gae": t_gae, "t_update": t_update,
                   "ep_len": float(np.mean([e["len"] for e in eps])) if eps else float("nan"),
                   "timeout_rate": float(np.mean([e["truncated"] for e in eps])) if eps else float("nan"),
                   # results of the games finished in this update (p1 = first player of the game).
                   # In self-play the mean return is ~0 by symmetry, so I log the wins of each side
                   # and the mean |raw return| (about WIN if the game is decided, 0 for a draw)
                   "p1_win_rate": float(np.mean([e["p1_outcome"] == 1 for e in eps])) if eps else float("nan"),
                   "p2_win_rate": float(np.mean([e["p1_outcome"] == -1 for e in eps])) if eps else float("nan"),
                   "decided_rate": float(np.mean([e["p1_outcome"] != 0 for e in eps])) if eps else float("nan"),
                   "abs_return": float(np.mean([abs(e["p1_return"]) for e in eps])) if eps else float("nan"),
                   # mean |training reward| per step (the mean itself is 0, the game is zero-sum)
                   "train_abs_reward": float(rew_buf.abs().mean()),
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

            # 5. checkpoint every checkpoint_every steps
            if steps >= next_ckpt:
                save_agent(os.path.join(run_dir, f"ckpt_{steps}.pt"), agent, cfg, env.obs_dim, env.n_actions, steps,
                           game_cfg, optimizer)
                next_ckpt += args.checkpoint_every
        save_agent(os.path.join(run_dir, "final.pt"), agent, cfg, env.obs_dim, env.n_actions, steps, game_cfg)
    finally:
        # always close the log and the game server, even if something crashes
        log_file.close()
        env.close()


if __name__ == "__main__":
    main()
