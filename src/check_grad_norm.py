"""How much does the joint gradient-norm clipping shrink the actor's update?

PPO clips the global norm of the gradient of (policy loss + vf_coef * value loss - entropy bonus)
over ALL parameters (max_grad_norm = 0.5, as in the FootsiesGym paper). Actor and critic are
separate networks, so the global norm is sqrt(|g_actor|^2 + |g_critic|^2): a large critic
gradient (returns on a +-10 scale) can shrink the actor's step as well.

For each checkpoint this script plays a self-play rollout with the training settings, computes
GAE and the PPO losses exactly as in training (no optimizer step), and reports per minibatch:
  * actor and critic gradient norms;
  * joint clip scale    = min(1, max_grad_norm / global norm)   (what training does);
  * separate clip scale = min(1, max_grad_norm / actor norm)    (if the actor were clipped alone);
  * shrink factor       = separate / joint: how many times smaller the actor step is because
    of the critic. 1 = the critic does not matter.

Usage:
    uv run python src/check_grad_norm.py                       # default runs, checkpoints at 1M, 5M, 10M, 15M, final
    uv run python src/check_grad_norm.py --runs r0_s2 r1_s3 --warmup 300
Results: printed and written to results/grad_norms.csv.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import portpicker
import torch

from env import FootsiesVecEnv
from ppo import PPOConfig, compute_gae, load_agent
from rewards import make_reward

DEFAULT_RUNS = ["r0_s2", "r0_s5", "r1_s3", "r0_floor_s5"]
MARKS = [1, 5, 10, 15]  # million steps; the final checkpoint is always added


def pick_checkpoints(run):
    cks = {int(os.path.basename(p)[5:-3]): p for p in glob.glob(f"runs/{run}/ckpt_*.pt")}
    out = []
    for m in MARKS:
        above = [s for s in cks if s >= m * 1_000_000]
        if above:
            out.append(cks[min(above)])
    return out + [f"runs/{run}/final.pt"]


@torch.no_grad()
def rollout(env, agent, obs, T):
    S = env.num_slots
    buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "term", "trunc")}
    for _ in range(T):
        a, lp, _, v = agent.get_action_and_value(obs)
        nobs, r, te, tr = env.step(a.numpy())
        for k, x in zip(buf, (obs, a, lp, v, torch.as_tensor(r), torch.as_tensor(te), torch.as_tensor(tr))):
            buf[k].append(x)
        obs = torch.as_tensor(nobs)
    vals = torch.stack(buf["val"] + [agent.get_value(obs)])
    adv, ret = compute_gae(torch.stack(buf["rew"]).float(), vals, torch.stack(buf["term"]),
                           torch.stack(buf["trunc"]), PPOConfig.gamma, PPOConfig.gae_lambda)
    batch = {"obs": torch.stack(buf["obs"]).reshape(-1, obs.shape[-1]), "actions": torch.stack(buf["act"]).reshape(-1),
             "advantages": adv.reshape(-1), "returns": ret.reshape(-1)}
    return obs, batch


def grad_norms(agent, batch, cfg, ent_coef):
    n = batch["obs"].shape[0]
    mb = n // cfg.num_minibatches
    perm = torch.randperm(n)
    rows = []
    for start in range(0, n - mb + 1, mb):
        idx = perm[start:start + mb]
        _, logp, ent, v = agent.get_action_and_value(batch["obs"][idx], batch["actions"][idx])
        adv = batch["advantages"][idx]
        adv = (adv - adv.mean()) / (adv.std().clamp(min=cfg.adv_std_floor) + 1e-8)
        pg_loss = -(adv * (logp - logp.detach()).exp()).mean()  # first epoch: ratio = 1, same gradient as PPO
        v_loss = 0.5 * ((v - batch["returns"][idx]) ** 2).mean()
        agent.zero_grad()
        (pg_loss + cfg.vf_coef * v_loss - ent_coef * ent.mean()).backward()
        ga = torch.sqrt(sum((p.grad ** 2).sum() for p in agent.actor.parameters())).item()
        gc = torch.sqrt(sum((p.grad ** 2).sum() for p in agent.critic.parameters())).item()
        joint = min(1.0, cfg.max_grad_norm / max(np.hypot(ga, gc), 1e-12))
        separate = min(1.0, cfg.max_grad_norm / max(ga, 1e-12))
        rows.append((ga, gc, joint, separate, v_loss.item()))
    return np.array(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--warmup", type=int, default=200, help="steps played before measuring (games in progress)")
    p.add_argument("--rollouts", type=int, default=3, help="measured rollouts per checkpoint")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    out = []
    print(f"{'checkpoint':34s} {'|g_actor|':>9s} {'|g_critic|':>10s} {'joint':>7s} {'separate':>8s} {'shrink':>7s} {'v_loss':>7s}")
    for run in args.runs:
        conf = json.load(open(f"runs/{run}/config.json"))
        a = conf["args"]
        reward_fn, overrides = make_reward(a["reward"], gamma=a["gamma"])
        env = FootsiesVecEnv(a["num_envs"], port=portpicker.pick_unused_port(),
                             env_config={**overrides, **conf.get("game", {})}, reward_fn=reward_fn)
        try:
            for ck in pick_checkpoints(run):
                agent, meta = load_agent(ck)
                cfg = PPOConfig(**{k: v for k, v in meta["cfg"].items() if k in PPOConfig.__dataclass_fields__})
                frac = 1.0 - meta["steps"] / a["total_steps"]
                ent_coef = cfg.ent_coef * max(frac, 0.0) if cfg.ent_decay else cfg.ent_coef
                obs = torch.as_tensor(env.reset())
                with torch.no_grad():
                    for _ in range(args.warmup):
                        act, _, _, _ = agent.get_action_and_value(obs)
                        obs = torch.as_tensor(env.step(act.numpy())[0])
                stats = []
                for _ in range(args.rollouts):
                    obs, batch = rollout(env, agent, obs, a["rollout"])
                    stats.append(grad_norms(agent, batch, cfg, ent_coef))
                s = np.median(np.concatenate(stats), axis=0)
                shrink = s[3] / s[2]
                label = f"{run}/{os.path.basename(ck)[:-3]}"
                print(f"{label:34s} {s[0]:9.3f} {s[1]:10.3f} {s[2]:7.3f} {s[3]:8.3f} {shrink:7.1f} {s[4]:7.3f}", flush=True)
                out.append({"run": run, "checkpoint": os.path.basename(ck), "steps": meta["steps"],
                            "actor_grad_norm": s[0], "critic_grad_norm": s[1], "joint_clip_scale": s[2],
                            "separate_clip_scale": s[3], "shrink_factor": shrink, "v_loss": s[4]})
        finally:
            env.close()
    with open("results/grad_norms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("saved results/grad_norms.csv")


if __name__ == "__main__":
    main()
