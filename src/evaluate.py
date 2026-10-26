"""Evaluate an agent against the fixed opponents.

The agent plays `games` games against each opponent, half as P1 and half
as P2. Actions are sampled from the agent's policy, never argmax. The env
uses only the terminal reward (+-10 for a win/loss, 0 on timeout), so the
sign of the agent's final reward gives the outcome.

Examples:
    uv run python src/evaluate.py --agent random --games 1000
    uv run python src/evaluate.py --agent runs/smoke/final.pt --games 200
    uv run python src/evaluate.py --agent footsies_bot --opponents noop back
"""
import argparse
import math
import os
import sys

import footsiesgym
import torch

from opponents import OPPONENTS, Opponent, make_opponent
from policy import load_policy_opponent

PLATFORM = "mac" if sys.platform == "darwin" else "linux"

ENV_CONFIG = {
    "frame_skip": 4,
    "action_delay": 8,
    "max_t": 4000,
    "win_reward_scaling_coeff": 10.0,
    "return_fight_state_in_infos": True,  # needed by FootsiesBot
}


def normal_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% normal-approximation interval p +- z * sqrt(p(1-p)/n), clipped to [0, 1]."""
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)


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


def evaluate(agent: Opponent, opponent_names: list[str], games: int,
             seed: int = 0) -> list[dict]:
    """Play `games` games against each opponent and return one row per opponent."""
    env = footsiesgym.make(config=ENV_CONFIG, platform=PLATFORM)
    rows = []
    try:
        for name in opponent_names:
            opponent = make_opponent(name, seed=seed,
                                     frame_skip=ENV_CONFIG["frame_skip"])
            outcomes = [play_game(env, agent, opponent, "p1" if i % 2 == 0 else "p2")
                        for i in range(games)]
            wins, draws, losses = (outcomes.count(1), outcomes.count(0),
                                   outcomes.count(-1))
            low, high = normal_interval(wins, games)
            rows.append({"opponent": name, "games": games, "wins": wins,
                         "draws": draws, "losses": losses,
                         "win_rate": wins / games, "ci_low": low, "ci_high": high})
            print(f"done: {name}")
    finally:
        env.close()
    return rows


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
                   choices=sorted(OPPONENTS))
    p.add_argument("--games", type=int, default=1000,
                   help="games per opponent, sides alternate P1/P2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # Agent and opponents get different seeds so that random vs random
    # does not play mirrored actions.
    torch.manual_seed(args.seed + 1)  # sampling of a PolicyOpponent
    agent = make_player(args.agent, seed=args.seed + 1,
                        frame_skip=ENV_CONFIG["frame_skip"],
                        device=args.device)
    print_table(evaluate(agent, args.opponents, args.games, args.seed))


if __name__ == "__main__":
    main()
