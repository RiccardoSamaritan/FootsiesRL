"""Watch a trained agent play against a fixed opponent in the windowed game.

We open the graphical build of Footsies (headless=False) and play a few games in real
time, so they can be recorded and shown (for example in the slides). The agent samples
its actions from the policy, as in the evaluation.

Examples:
    uv run python src/watch.py --agent runs/r1_floor_s1/final.pt
    uv run python src/watch.py --agent runs/r0_s2/final.pt --opponent footsies_bot --games 5
    uv run python src/watch.py --agent runs/r1_floor_s1/final.pt --until-win --speed 0.5

"""
import argparse
import time

import footsiesgym
import portpicker

from evaluate import ENV_CONFIG, PLATFORM, _game_config, make_player, wait_for_server


def parse_args():
    """Command line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, help="checkpoint of the trained agent (e.g. runs/r1_floor_s1/final.pt)")
    p.add_argument("--opponent", default="footsies_bot",
                   help="fixed opponent (footsies_bot, random, noop, back) or another checkpoint")
    p.add_argument("--side", choices=["p1", "p2"], default="p1", help="side of the agent (p1 = left)")
    p.add_argument("--games", type=int, default=3, help="number of games to play")
    p.add_argument("--until-win", action="store_true",
                   help="keep playing until the agent wins a game (at most --max-games)")
    p.add_argument("--max-games", type=int, default=20)
    p.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0.5 = half speed, 2 = double")
    p.add_argument("--start-delay", type=float, default=3.0,
                   help="seconds to wait before the first game (time to start the recording)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    """Opens the windowed game and plays the games, printing the result of each one."""
    args = parse_args()
    frame_skip = ENV_CONFIG["frame_skip"]
    agent = make_player(args.agent, seed=args.seed, frame_skip=frame_skip)
    opponent = make_player(args.opponent, seed=args.seed, frame_skip=frame_skip)
    # same game rules the agent was trained with (action delay, charge actions)
    game_cfg = _game_config(agent) or {}

    port = portpicker.pick_unused_port()
    env = footsiesgym.make(config={**ENV_CONFIG, **game_cfg, "headless": False, "port": port},
                           platform=PLATFORM)
    wait_for_server(port)  # the windowed build can be slow to start

    opp_side = "p2" if args.side == "p1" else "p1"
    players = {args.side: agent, opp_side: opponent}
    dt = frame_skip / 60 / args.speed  # one env step = frame_skip frames of a 60 fps game
    print(f"agent ({args.agent}) plays as {args.side} against {args.opponent}")

    played = wins = 0
    try:
        time.sleep(args.start_delay)
        while True:
            for player in players.values():
                player.reset()
            obs, infos = env.reset()
            steps = 0
            while True:
                actions = {side: player.act(side, obs[side], infos.get(side, {}))
                           for side, player in players.items()}
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                steps += 1
                time.sleep(dt)
                if terminateds["p1"] or truncateds["p1"]:
                    break
            reward = rewards[args.side]
            outcome = "win" if reward > 0 else ("loss" if reward < 0 else "draw (timeout)")
            played += 1
            wins += reward > 0
            print(f"game {played}: agent {outcome} after {steps} steps ({steps * frame_skip / 60:.1f} s of game)",
                  flush=True)
            if args.until_win:
                if reward > 0 or played >= args.max_games:
                    break
            elif played >= args.games:
                break
            time.sleep(1.0)  # short pause between games, nicer in the video
    finally:
        env.close()
    print(f"agent won {wins} of {played} games")


if __name__ == "__main__":
    main()
