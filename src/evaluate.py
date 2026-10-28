"""Evaluate an agent against the fixed opponents.

The agent plays `games` games against each opponent, half as P1 and half
as P2. Actions are sampled from the agent's policy, never argmax. The env
uses only the terminal reward (+-10 for a win/loss, 0 on timeout), so the
sign of the agent's final reward gives the outcome.

Examples:
    uv run python src/evaluate.py --agent random --games 1000
    uv run python src/evaluate.py --agent runs/smoke/final.pt --games 200
    uv run python src/evaluate.py --agent footsies_bot --opponents noop back
    uv run python src/evaluate.py --agent runs/a/final.pt --opponents runs/b/final.pt

Results are appended to results/eval_<run>.csv (one line per opponent); use --no-save
to only print the table.
"""
import argparse
import csv
import math
import os
import re
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path

import footsiesgym
import numpy as np
import portpicker
import torch

from env import ENV_CONFIG as TRAIN_ENV_CONFIG, FootsiesVecEnv, default_platform
from opponents import (OPPONENTS, ConstantActionOpponent, Opponent,
                       RandomOpponent, make_opponent)
from policy import PolicyOpponent, load_policy_opponent

PLATFORM = default_platform()

# Same game settings as training (single source: env.py).
ENV_CONFIG = {
    **TRAIN_ENV_CONFIG,
    "return_fight_state_in_infos": True,  # needed by FootsiesBot
}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (well behaved at 0% and 100%)."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def wait_for_server(port: int, timeout: float = 120.0) -> None:
    """Wait until the game server accepts connections (it can start slowly, e.g. on macOS)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Footsies server did not start on port {port}")


def make_player(spec: str, *, seed: int, frame_skip: int,
                device: str = "cpu") -> Opponent:
    """Build a player from a fixed-opponent name or from a checkpoint path."""
    if spec in OPPONENTS:
        return make_opponent(spec, seed=seed, frame_skip=frame_skip)
    if not os.path.isfile(spec):
        raise ValueError(
            f"{spec!r} is neither a fixed opponent ({sorted(OPPONENTS)}) "
            "nor the path of a checkpoint file")
    return load_policy_opponent(spec, device=device)


def play_game(env, agent: Opponent, opponent: Opponent, agent_side: str) -> int:
    """Play one game. Return +1 if the agent wins, -1 if it loses, 0 for a draw."""
    opponent_side = "p2" if agent_side == "p1" else "p1"
    players = {agent_side: agent, opponent_side: opponent}
    for player in players.values():
        player.reset()
    obs, infos = env.reset()
    while True:
        actions = {side: player.act(side, obs[side], infos.get(side, {}))
                   for side, player in players.items()}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        if terminateds["p1"] or truncateds["p1"]:
            reward = rewards[agent_side]
            return (reward > 0) - (reward < 0)


def _batchable(player) -> bool:
    """Players that can decide for many games at once (no per-game state, no infos)."""
    return isinstance(player, (PolicyOpponent, RandomOpponent, ConstantActionOpponent))


def _act_batch(player, obs: np.ndarray) -> np.ndarray:
    if isinstance(player, PolicyOpponent):
        return player.act_batch(obs)
    if isinstance(player, RandomOpponent):
        return player.rng.integers(player.num_actions, size=len(obs))
    return np.full(len(obs), player.action, dtype=np.int64)


def play_games_vec(venv: FootsiesVecEnv, agent, opponent, games: int) -> list[int]:
    """Play `games` games at once on a vectorized env.

    Game slot i is fixed to one side (agent = p1 for even i, p2 for odd i), so
    sides are balanced. Every slot plays the same number of games (+1 for the
    first few): the first games to finish are the short ones, so stopping at
    the first `games` finished episodes would over-count quick wins.
    Return the outcomes (+1 win, 0 draw, -1 loss) from the agent's side.
    """
    n = venv.num_envs
    agent_is_p1 = np.arange(n) % 2 == 0
    # slot index (in the 2N stacked layout) of the agent and of the opponent in each game
    agent_idx = np.where(agent_is_p1, np.arange(n), n + np.arange(n))
    opp_idx = np.where(agent_is_p1, n + np.arange(n), np.arange(n))
    quota = np.full(n, games // n) + (np.arange(n) < games % n)
    played = np.zeros(n, dtype=np.int64)
    outcomes = []

    agent.reset()
    opponent.reset()
    obs = venv.reset()
    while (played < quota).any():
        actions = np.zeros(2 * n, dtype=np.int64)
        actions[agent_idx] = _act_batch(agent, obs[agent_idx])
        actions[opp_idx] = _act_batch(opponent, obs[opp_idx])
        obs, rew, term, trunc = venv.step(actions)
        done = term[:n] | trunc[:n]
        for i in np.flatnonzero(done):
            if played[i] >= quota[i]:
                continue  # extra episode of a slot that already finished its share
            played[i] += 1
            r = rew[:n][i] if agent_is_p1[i] else -rew[:n][i]  # p1 reward; zero-sum
            outcomes.append(int(r > 0) - int(r < 0))
    return outcomes


def evaluate(agent: Opponent, opponent_names: list[str], games: int,
             seed: int = 0, parallel: int = 48) -> list[dict]:
    """Play `games` games against each opponent and return one row per opponent.

    Players that decide independently per game (checkpoints, random, noop, back) are
    evaluated on a vectorized env with up to `parallel` simultaneous games. The
    others (footsies_bot needs the fight state in infos, only given by a single env)
    play one game at a time.
    """
    n_par = max(2, min(parallel, games) // 2 * 2)  # even, >= 2
    venv = single = None
    rows = []
    try:
        for name in opponent_names:
            opponent = make_player(name, seed=seed, frame_skip=ENV_CONFIG["frame_skip"])
            if _batchable(agent) and _batchable(opponent):
                if venv is None:
                    port = portpicker.pick_unused_port()
                    venv = FootsiesVecEnv(n_par, port=port, platform=PLATFORM)
                outcomes = play_games_vec(venv, agent, opponent, games)
            else:
                if single is None:
                    port = portpicker.pick_unused_port()
                    single = footsiesgym.make(config={**ENV_CONFIG, "port": port},
                                              platform=PLATFORM)
                    wait_for_server(port)
                outcomes = [play_game(single, agent, opponent, "p1" if i % 2 == 0 else "p2")
                            for i in range(games)]
            wins, draws, losses = (outcomes.count(1), outcomes.count(0),
                                   outcomes.count(-1))
            low, high = wilson_interval(wins, games)
            rows.append({"opponent": os.path.basename(name), "opponent_spec": name, "games": games, "wins": wins,
                         "draws": draws, "losses": losses,
                         "win_rate": wins / games, "ci_low": low, "ci_high": high})
            print(f"done: {name}")
    finally:
        for e in (venv, single):
            if e is not None:
                e.close()
    return rows


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def default_out(agent: str) -> Path:
    """One CSV per run: results/eval_<run>.csv (results/eval_other.csv for a non-run agent)."""
    return RESULTS_DIR / f"eval_{_run_name(agent) or 'other'}.csv"
CSV_FIELDS = ["eval_id", "timestamp", "label", "agent", "run", "steps", "opponent",
              "games", "wins", "draws", "losses", "win_rate", "ci_low", "ci_high", "seed"]


def _run_name(agent: str) -> str:
    """Run folder of a checkpoint path (runs/<run>/<file>.pt), '' for a fixed opponent."""
    p = Path(agent)
    return p.parent.name if p.suffix == ".pt" else ""


def _ckpt_steps(agent: str):
    m = re.search(r"ckpt_(\d+)", Path(agent).stem)
    return int(m.group(1)) if m else ""


def _upgrade_csv(path: Path) -> None:
    """Rewrite an older CSV (without eval_id/label/run/steps) in the current format.

    Rows written by the same evaluation share a timestamp and get the same eval_id.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == CSV_FIELDS:
            return
        old = list(reader)
    ids = {}
    out = []
    for r in old:
        stamp = r["timestamp"]
        key = (stamp, r["agent"])
        if key not in ids:
            ids[key] = stamp.replace("-", "").replace(":", "").replace("T", "-") + f"-{len(ids):02d}"
        out.append({"eval_id": ids[key], "timestamp": stamp, "label": "",
                    "agent": r["agent"], "run": _run_name(r["agent"]),
                    "steps": _ckpt_steps(r["agent"]), **{k: r[k] for k in CSV_FIELDS[6:]}})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(out)


def save_rows(path, agent: str, seed: int, rows: list[dict], label: str = "",
              steps=None) -> str:
    """Append one line per opponent to a CSV (created with a header if missing).

    All lines of one call share an eval_id, so each evaluation can be told apart.
    Returns the eval_id.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _upgrade_csv(path)
    new_file = not path.exists()
    now = datetime.now()
    eval_id = now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    if steps is None:
        steps = _ckpt_steps(agent)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow({"eval_id": eval_id, "timestamp": now.isoformat(timespec="seconds"),
                             "label": label, "agent": agent, "run": _run_name(agent),
                             "steps": steps, "opponent": r["opponent_spec"],
                             "games": r["games"], "wins": r["wins"], "draws": r["draws"],
                             "losses": r["losses"], "win_rate": round(r["win_rate"], 4),
                             "ci_low": round(r["ci_low"], 4), "ci_high": round(r["ci_high"], 4),
                             "seed": seed})
    return eval_id


def print_table(rows: list[dict]) -> None:
    print(f"\n{'opponent':<14}{'games':>6}{'win':>6}{'draw':>6}{'loss':>6}"
          f"{'win rate':>10}  95% CI")
    for r in rows:
        print(f"{r['opponent']:<14}{r['games']:>6}{r['wins']:>6}{r['draws']:>6}"
              f"{r['losses']:>6}{r['win_rate']:>10.3f}  "
              f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="random",
                   help="fixed opponent name or path to a .pt checkpoint")
    p.add_argument("--opponents", nargs="+", default=list(OPPONENTS),
                   help=f"fixed opponent names ({sorted(OPPONENTS)}) and/or checkpoint paths")
    p.add_argument("--games", type=int, default=1000,
                   help="games per opponent, sides alternate P1/P2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", type=int, default=48,
                   help="games played at the same time (vectorized env)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None,
                   help="CSV file where results are appended (default: results/eval_<run>.csv)")
    p.add_argument("--label", default="", help="free text stored with the results (e.g. \"r1 vs fixed opponents\")")
    p.add_argument("--no-save", action="store_true", help="only print the table")
    args = p.parse_args()

    # Agent and opponents get different seeds so that random vs random
    # does not play mirrored actions.
    torch.manual_seed(args.seed + 1)  # sampling of a PolicyOpponent
    agent = make_player(args.agent, seed=args.seed + 1,
                        frame_skip=ENV_CONFIG["frame_skip"],
                        device=args.device)
    rows = evaluate(agent, args.opponents, args.games, args.seed, args.parallel)
    print_table(rows)
    if not args.no_save:
        args.out = args.out or default_out(args.agent)
        eval_id = save_rows(args.out, args.agent, args.seed, rows, label=args.label,
                            steps=getattr(agent, "steps", None))
        print(f"\nresults appended to {args.out} (eval_id {eval_id})")


if __name__ == "__main__":
    main()
