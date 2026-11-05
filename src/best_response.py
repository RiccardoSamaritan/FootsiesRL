"""Approximate best response: train PPO against a frozen checkpoint.

    uv run python src/best_response.py --target runs/r0_s1/final.pt               # full run
    uv run python src/best_response.py --target runs/r0_s1/final.pt \
        --total-steps 200000 --num-envs 8 --eval-games 40 --out-dir /tmp/br     # smoke test

How strong can a dedicated opponent get against a fixed agent? The target plays
with its frozen policy (actions sampled, as in evaluation); a new network (the
learner) is trained from scratch with our PPO against it. The learner's final win
rate against the target is an approximation of the target's exploitability: a
LOWER bound, since PPO only finds the weaknesses it manages to discover in its
budget (as in the FootsiesGym paper, Section 4, and Timbers et al. 2020).

Differences from train.py:
  * no self-play: the learner plays one side of every game, the target the other
    (learner = P1 in even games, P2 in odd games, so sides are balanced), and only
    the learner's samples go into the PPO batch (half of train.py's batch);
  * the reward is always the terminal one (+-10, 0 on timeout), whatever reward the
    target was trained with, so that exploitabilities are comparable;
  * the env uses the target's game settings (action delay, charge actions).
The PPO hyperparameters are those of the fixed-entropy runs (entropy 0.025).

Output: runs_br/<name>/ (log.csv with the learner's win/draw/loss rate per update,
config.json, checkpoints, final.pt) and one line per evaluation appended to
results/br_<target run>.csv: the final learner against the target, sides alternated.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from env import ENV_CONFIG, FootsiesVecEnv
from ppo import Agent, PPOConfig, compute_gae, ppo_update, save_agent
from policy import load_policy_opponent

RESULTS = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="checkpoint to exploit (e.g. runs/r0_s1/final.pt)")
    p.add_argument("--run-name", default=None, help="default: br_<target run>_b<seed>")
    p.add_argument("--out-dir", default="runs_br")
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--num-envs", type=int, default=48)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--seed", type=int, default=1, help="seed of the learner")
    p.add_argument("--port", type=int, default=46001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ent-coef", type=float, default=0.025)
    p.add_argument("--checkpoint-every", type=int, default=5_000_000, help="steps")
    p.add_argument("--eval-games", type=int, default=500, help="final games against the target (0 = skip)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training and only run the final evaluation of an existing final.pt")
    return p.parse_args()


def game_settings(target):
    """Game settings the target was trained with (defaults for older checkpoints)."""
    cfg = target.env_config
    return {"action_delay": cfg.get("action_delay", ENV_CONFIG["action_delay"]),
            "use_special_charge_action": cfg.get("use_special_charge_action", False)}


def final_evaluation(final_path, target_path, games, seed, run_name, steps):
    """Final learner against the target with alternating sides; append to results/br_<run>.csv."""
    from evaluate import evaluate, wilson_interval  # imported here: needs only the game server

    learner = load_policy_opponent(final_path)
    row = evaluate(learner, [target_path], games, seed=seed)[0]
    target_run = Path(target_path).parent.name
    out = RESULTS / f"br_{target_run}.csv"
    new_file = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = wilson_interval(row["wins"], games)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), "target": target_path,
              "target_run": target_run, "br_run": run_name, "br_steps": steps, "games": games,
              "br_wins": row["wins"], "draws": row["draws"], "br_losses": row["losses"],
              "br_win_rate": round(row["wins"] / games, 4), "ci_low": round(lo, 4),
              "ci_high": round(hi, 4)}
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        if new_file:
            writer.writeheader()
        writer.writerow(record)
    print(f"\nbest response vs {target_path}: {row['wins']} wins, {row['draws']} draws, "
          f"{row['losses']} losses over {games} games (BR win rate {row['wins'] / games:.3f}, "
          f"95% CI [{lo:.3f}, {hi:.3f}]) -> {out}")


def main():
    args = parse_args()
    target = load_policy_opponent(args.target, device=args.device)
    target_run = Path(args.target).parent.name
    run_name = args.run_name or f"br_{target_run}_b{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    if args.eval_only:
        final_path = os.path.join(run_dir, "final.pt")
        if not os.path.exists(final_path):
            sys.exit(f"--eval-only: {final_path} does not exist")
        steps = int(torch.load(final_path, map_location="cpu")["steps"])
        final_evaluation(final_path, args.target, args.eval_games, args.seed, run_name, steps)
        return
    if os.path.exists(os.path.join(run_dir, "log.csv")) and not args.overwrite:
        sys.exit(f"{run_dir} already has a log.csv: choose another --run-name or pass --overwrite")
    os.makedirs(run_dir, exist_ok=True)
    assert args.num_envs % 2 == 0, "--num-envs must be even (half the games per side)"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    game_cfg = game_settings(target)
    cfg = PPOConfig(gamma=args.gamma, gae_lambda=args.gae_lambda, lr=args.lr, ent_coef=args.ent_coef)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "ppo": asdict(cfg), "game": game_cfg,
                   "target_steps": target.steps}, f, indent=2)

    # terminal reward only (no reward_fn, no guard reward), with the target's game settings
    env = FootsiesVecEnv(args.num_envs, port=args.port, env_config=game_cfg)
    if env.obs_dim != target.obs_dim or env.n_actions != target.n_actions:
        sys.exit("the env does not match the target checkpoint (observation or action size)")
    N, T = args.num_envs, args.rollout
    games = np.arange(N)
    learner_is_p1 = games % 2 == 0
    # slot of the learner and of the target in each game (2N layout: p1 slots [0, N), p2 slots [N, 2N))
    learn_idx = np.where(learner_is_p1, games, N + games)
    target_idx = np.where(learner_is_p1, N + games, games)

    agent = Agent(env.obs_dim, env.n_actions, cfg.hidden).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    obs_buf = torch.zeros((T, N, env.obs_dim), device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T + 1, N), device=device)
    term_buf = torch.zeros((T, N), dtype=torch.bool, device=device)
    trunc_buf = torch.zeros((T, N), dtype=torch.bool, device=device)

    num_updates = max(args.total_steps // (N * T), 1)
    log_file, writer = open(os.path.join(run_dir, "log.csv"), "w", newline=""), None
    next_ckpt = args.checkpoint_every
    steps = 0
    t_start = time.time()
    final_path = os.path.join(run_dir, "final.pt")
    try:
        obs = env.reset()
        for update in range(1, num_updates + 1):
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.lr
            outcomes, lengths = [], []
            for t in range(T):
                x = torch.as_tensor(obs[learn_idx], device=device)
                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(x)
                obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = x, action, logp, value
                actions = np.zeros(2 * N, dtype=np.int64)
                actions[learn_idx] = action.cpu().numpy()
                actions[target_idx] = target.act_batch(obs[target_idx])
                t_before = env.t.copy()  # per-game step counter before this step
                obs, rew, term, trunc = env.step(actions)
                r = rew[learn_idx]  # terminal reward of the learner (+-10, 0 on timeout)
                done = term[:N] | trunc[:N]
                rew_buf[t] = torch.as_tensor(r, device=device)
                term_buf[t] = torch.as_tensor(term[:N], device=device)
                trunc_buf[t] = torch.as_tensor(trunc[:N] & ~term[:N], device=device)
                for i in np.flatnonzero(done):
                    outcomes.append(int(r[i] > 0) - int(r[i] < 0))
                    lengths.append(int(t_before[i]) + 1)
                steps += N
            env.pop_finished()  # per-episode stats of the wrapper are not used here
            with torch.no_grad():
                val_buf[T] = agent.get_value(torch.as_tensor(obs[learn_idx], device=device))
            adv, ret = compute_gae(rew_buf, val_buf, term_buf, trunc_buf, cfg.gamma, cfg.gae_lambda)
            batch = {"obs": obs_buf.reshape(-1, env.obs_dim), "actions": act_buf.reshape(-1),
                     "logprobs": logp_buf.reshape(-1), "advantages": adv.reshape(-1),
                     "returns": ret.reshape(-1), "values": val_buf[:-1].reshape(-1)}
            stats = ppo_update(agent, optimizer, batch, cfg, cfg.ent_coef)

            o = np.array(outcomes)
            nan = float("nan")
            row = {"update": update, "steps": steps, "time": round(time.time() - t_start, 1),
                   "lr": optimizer.param_groups[0]["lr"], "episodes": len(o),
                   "br_win": float(np.mean(o == 1)) if len(o) else nan,
                   "br_draw": float(np.mean(o == 0)) if len(o) else nan,
                   "br_loss": float(np.mean(o == -1)) if len(o) else nan,
                   "ep_len": float(np.mean(lengths)) if lengths else nan, **stats}
            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log_file.flush()
            if update % 10 == 0 or update == 1:
                print(f"update {update}/{num_updates} steps {steps} "
                      f"{steps / (time.time() - t_start):.0f} step/s  BR win {row['br_win']:.2f} "
                      f"draw {row['br_draw']:.2f} loss {row['br_loss']:.2f}  ent {stats['entropy']:.3f} "
                      f"ev {stats['explained_var']:.2f}", flush=True)
            if steps >= next_ckpt:
                save_agent(os.path.join(run_dir, f"ckpt_{steps}.pt"), agent, cfg, env.obs_dim,
                           env.n_actions, steps, game_cfg)
                next_ckpt += args.checkpoint_every
        save_agent(final_path, agent, cfg, env.obs_dim, env.n_actions, steps, game_cfg)
    finally:
        log_file.close()
        env.close()

    if args.eval_games > 0:
        final_evaluation(final_path, args.target, args.eval_games, args.seed, run_name, steps)


if __name__ == "__main__":
    main()
