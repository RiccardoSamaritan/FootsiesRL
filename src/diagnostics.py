"""Behaviour diagnostics of trained agents: separate "stall" from "reactive" play.

For each checkpoint it plays a fixed number of steps on a vectorized env, in self-play
(the checkpoint on both sides) and against noop / back / random, and records:

  * action distribution of the agent (share of each action; attack = ATTACK,
    BACK_ATTACK, FORWARD_ATTACK; special = the three charge actions, if enabled);
  * distance |x_own - x_opp| (observation features 2 and 43): median and 10th percentile;
  * per finished game: outcome, length, whether anybody attacked, time of the first
    attack and who made it (agent / opponent / same step).

Reading guide:
  * vs noop a good agent walks up and attacks (short games, wins). An agent that
    never attacks vs noop is passive, not just "stalled by its opponent".
  * In self-play, share of games with no attack at all = total stall.
  * Special-charge actions share > 0 shows that the option is really used.

Examples:
    uv run python src/diagnostics.py runs/r0_s1/final.pt runs/r0_charge_s1/final.pt
    uv run python src/diagnostics.py runs/r0_s1/final.pt --steps 2000 --matchups self noop

Results are appended to results/diag_<run>.csv (one row per matchup).
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import portpicker

from env import FootsiesVecEnv, default_platform
from evaluate import _act_batch, _game_config, _opponent_label, make_player
from env import ENV_CONFIG

POS_OWN, POS_OPP = 2, 43  # footsiesgym feature_indices.json
ACTION_NAMES = ["none", "back", "forward", "attack", "back_attack", "forward_attack",
                "charge", "fwd_charge", "back_charge"]
ATTACKS = (3, 4, 5)
SPECIALS = (6, 7, 8)
MATCHUPS = ("self", "noop", "back", "random")


def diagnose(venv, agent, opponent, steps: int) -> dict:
    """Play `steps` vector steps. opponent=None means self-play (agent on both sides)."""
    n = venv.num_envs
    is_p1 = np.arange(n) % 2 == 0
    a_idx = np.where(is_p1, np.arange(n), n + np.arange(n))
    o_idx = np.where(is_p1, n + np.arange(n), np.arange(n))
    n_actions = agent.n_actions
    counts = np.zeros(n_actions)
    dists = []
    t = np.zeros(n, dtype=np.int64)
    first_t = np.full(n, -1)
    first_who = np.zeros(n, dtype=np.int64)  # 0 none, 1 agent, 2 opponent, 3 same step
    games = []

    agent.reset()
    if opponent is not None:
        opponent.reset()
    obs = venv.reset()
    for _ in range(steps):
        act = np.zeros(2 * n, dtype=np.int64)
        if opponent is None:
            act[:] = agent.act_batch(obs)
        else:
            act[a_idx] = agent.act_batch(obs[a_idx])
            act[o_idx] = _act_batch(opponent, obs[o_idx])
        counts += np.bincount(act[a_idx], minlength=n_actions)[:n_actions]
        dists.append(np.abs(obs[:n, POS_OWN] - obs[:n, POS_OPP]))

        a_att = np.isin(act[a_idx], ATTACKS + SPECIALS)
        o_att = np.isin(act[o_idx], ATTACKS + SPECIALS)
        new = (first_t < 0) & (a_att | o_att)
        first_t[new] = t[new]
        first_who[new] = a_att[new] * 1 + o_att[new] * 2

        obs, rew, term, trunc = venv.step(act)
        done = term[:n] | trunc[:n]
        t += 1
        for i in np.flatnonzero(done):
            r = rew[:n][i] if is_p1[i] else -rew[:n][i]  # raw reward only if venv has no reward_fn
            games.append((int(r > 0) - int(r < 0), int(t[i]), int(first_t[i]), int(first_who[i])))
            t[i] = 0
            first_t[i] = -1
            first_who[i] = 0

    d = np.concatenate(dists)
    share = counts / counts.sum()
    g = np.array(games) if games else np.zeros((0, 4), dtype=np.int64)
    out = {"slot_steps": int(counts.sum()), "games": len(g),
           "dist_median": float(np.median(d)), "dist_p10": float(np.percentile(d, 10))}
    for k in range(n_actions):
        out[f"a_{ACTION_NAMES[k]}"] = float(share[k])
    out["attack_share"] = float(share[list(ATTACKS)].sum())
    out["special_share"] = float(share[[k for k in SPECIALS if k < n_actions]].sum())
    if len(g):
        decided = g[:, 0] != 0
        attacked = g[:, 2] >= 0
        out.update({
            "win": float((g[:, 0] == 1).mean()), "draw": float((g[:, 0] == 0).mean()),
            "loss": float((g[:, 0] == -1).mean()), "ep_len_mean": float(g[:, 1].mean()),
            "no_attack_games": float((~attacked).mean()),
            "first_attack_t": float(g[attacked, 2].mean()) if attacked.any() else float("nan"),
            "first_by_agent": float((g[attacked, 3] == 1).mean()) if attacked.any() else float("nan"),
            "first_by_opp": float((g[attacked, 3] == 2).mean()) if attacked.any() else float("nan"),
            "decided_len": float(g[decided, 1].mean()) if decided.any() else float("nan"),
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agents", nargs="+", help="checkpoint paths")
    p.add_argument("--matchups", nargs="+", default=list(MATCHUPS), choices=MATCHUPS)
    p.add_argument("--steps", type=int, default=4500,
                   help="vector steps per matchup (>= max_t so that timeouts finish)")
    p.add_argument("--parallel", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    for path in args.agents:
        agent = make_player(path, seed=args.seed, frame_skip=ENV_CONFIG["frame_skip"])
        cfg = _game_config(agent)
        label = _opponent_label(path)
        run = Path(path).parent.name
        rows = []
        for m in args.matchups:
            opp = None if m == "self" else make_player(
                m, seed=args.seed, frame_skip=ENV_CONFIG["frame_skip"])
            venv = FootsiesVecEnv(args.parallel, port=portpicker.pick_unused_port(),
                                  platform=default_platform(), env_config=cfg)
            try:
                res = diagnose(venv, agent, opp, args.steps)
            finally:
                venv.close()
            row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "agent": label, "matchup": m, **res}
            rows.append(row)
            acts = " ".join(f"{k[2:]}={v:.2f}" for k, v in res.items() if k.startswith("a_"))
            print(f"{label} vs {m}: games={res['games']} win={res.get('win', float('nan')):.2f} "
                  f"draw={res.get('draw', float('nan')):.2f} len={res.get('ep_len_mean', float('nan')):.0f} "
                  f"no_attack={res.get('no_attack_games', float('nan')):.2f} "
                  f"first_t={res.get('first_attack_t', float('nan')):.0f} "
                  f"dist_med={res['dist_median']:.2f}\n    actions: {acts}", flush=True)
        out = Path(args.out_dir) / f"diag_{run}.csv"
        out.parent.mkdir(exist_ok=True)
        new = not out.exists()
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("timestamp", "agent", "matchup"), k))
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
