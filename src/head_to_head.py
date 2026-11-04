"""Head-to-head matches between checkpoints, and the resulting matrix.

    # final checkpoints of every run (one match per pair, sides alternate)
    uv run python src/head_to_head.py --agents 'runs/*/final.pt' --games 100 --label final

    # checkpoints of one run through training
    uv run python src/head_to_head.py --agents runs/r1_s1/ckpt_5001216.pt \
        runs/r1_s1/ckpt_10002432.pt runs/r1_s1/final.pt --label r1_s1_time

    # only redraw the figure from the saved CSV
    uv run python src/head_to_head.py --label final --plot-only

Each pair plays once (agent i against agent j, i < j) with `games` games, half as P1
and half as P2, both agents sampling from their policy. Results go to
results/h2h_<label>.csv, one line per pair; pairs already in the file are skipped, so
an interrupted run can be restarted with the same command (and a pool of agents can be
extended without replaying old pairs).

Score of A against B = (wins + 0.5 * draws) / games: 0.5 means even. Draws are timeouts
(both players passive) and count half: the draw share is shown in the figure.

Figures (results/figures/h2h_<label>.png):
  * left: score of the row agent against the column agent, one cell per pair;
  * right (only if several agents share a variant and a checkpoint name, e.g. the 5
    seeds of r1): the same matrix averaged over the pairs of agents of each variant.
    The diagonal compares different seeds of the same variant: it shows the spread
    between training runs. Colors: orange = row loses, blue = row wins.
A ranking (mean score against all the other agents) is printed and saved to
results/h2h_<label>_ranking.csv. Self-play is not transitive: A can beat B, B beat C
and C beat A, so read the matrix and not only the ranking.
"""
import argparse
import csv
import glob
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("h2h", ["#eb6834", "#f3f2ee", "#2a78d6"])
FIELDS = ["timestamp", "a", "b", "games", "wins_a", "draws", "wins_b", "score_a",
          "ci_low", "ci_high", "seed"]


# ---------------------------------------------------------------- names
def agent_label(path):
    """<run>/<file> as stored in the CSV, e.g. r1_s1/final."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def variant_of(label):
    """r1_s3/final -> r1; r0_sched_s2/ckpt_5001216 -> r0_sched."""
    return re.sub(r"_s\d+$", "", label.split("/")[0])


def group_of(label):
    """Variant plus checkpoint name, so final checkpoints and 5M checkpoints stay apart."""
    return f"{variant_of(label)}/{label.split('/')[1]}"


def wilson_interval(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------- playing
def csv_path(label, shard=(0, 1)):
    """results/h2h_<label>.csv; a shard k of n (n > 1) writes its own file h2h_<label>.s<k>of<n>.csv."""
    k, n = shard
    return RESULTS / (f"h2h_{label}.csv" if n == 1 else f"h2h_{label}.s{k}of{n}.csv")


def load_pairs(label):
    """Pairs saved so far in every file of the pool (plain and shards): {(a, b): row}."""
    files = [RESULTS / f"h2h_{label}.csv", *sorted(RESULTS.glob(f"h2h_{label}.s*of*.csv"))]
    pairs = {}
    for path in files:
        if path.exists():
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    pairs[(r["a"], r["b"])] = r
    return pairs


def sort_key(label):
    """Order agents by variant, then seed, then checkpoint (r0_s2/final after r0_s2/ckpt_5001216)."""
    run, stem = label.split("/")
    m = re.search(r"_s(\d+)$", run)
    steps = int(stem.split("_")[1]) if stem.startswith("ckpt_") else 10**12
    return (variant_of(label), int(m.group(1)) if m else 0, steps)


def run_pairs(agent_paths, games, parallel, seed, label, shard=(0, 1)):
    """Play every missing pair of the shard and append it to the shard's CSV.

    All pairs i < j are numbered in a fixed order; shard k of n plays those with
    number % n == k, so n jobs can share a pool without playing the same pair.
    """
    from evaluate import evaluate, make_player  # needs the game server: imported only here

    labels = [agent_label(p) for p in agent_paths]
    if len(set(labels)) != len(labels):
        raise SystemExit("two agents have the same <run>/<file> name")
    k, n_shards = shard
    out = csv_path(label, shard)
    done = load_pairs(label)
    new_file = not Path(out).exists()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # pair number of (i, j): position in the fixed order i < j
    number = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            number[(i, j)] = len(number)
    mine = lambda i, j: number[(i, j)] % n_shards == k and \
        (labels[i], labels[j]) not in done and (labels[j], labels[i]) not in done
    todo = sum(1 for (i, j) in number if mine(i, j))
    print(f"{len(labels)} agents, shard {k}/{n_shards}: {todo} pairs to play ({games} games each)", flush=True)
    played = 0
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, path_i in enumerate(agent_paths):
            rivals = [j for j in range(i + 1, len(labels)) if mine(i, j)]
            if not rivals:
                continue
            agent = make_player(path_i, seed=seed + 1, frame_skip=4)
            rows = evaluate(agent, [agent_paths[j] for j in rivals], games, seed, parallel)
            for j, r in zip(rivals, rows):
                lo, hi = wilson_interval(r["wins"] + 0.5 * r["draws"], games)
                writer.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"),
                                 "a": labels[i], "b": labels[j], "games": games,
                                 "wins_a": r["wins"], "draws": r["draws"], "wins_b": r["losses"],
                                 "score_a": round((r["wins"] + 0.5 * r["draws"]) / games, 4),
                                 "ci_low": round(lo, 4), "ci_high": round(hi, 4), "seed": seed})
                played += 1
            f.flush()
            print(f"  {labels[i]}: {len(rivals)} pairs ({played}/{todo})", flush=True)


# ---------------------------------------------------------------- matrices
def build_matrices(pairs, labels):
    """Score and draw-share matrices over `labels` (row agent against column agent)."""
    n = len(labels)
    idx = {l: k for k, l in enumerate(labels)}
    score = np.full((n, n), np.nan)
    draw = np.full((n, n), np.nan)
    for (a, b), r in pairs.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            score[i, j], score[j, i] = float(r["score_a"]), 1 - float(r["score_a"])
            d = float(r["draws"]) / float(r["games"])
            draw[i, j] = draw[j, i] = d
    return score, draw


def group_matrices(score, draw, labels):
    """Average the agent matrices over the pairs of agents of each variant/checkpoint group."""
    groups = list(dict.fromkeys(group_of(l) for l in labels))
    member = {g: [k for k, l in enumerate(labels) if group_of(l) == g] for g in groups}
    gs = np.full((len(groups), len(groups)), np.nan)
    gd = np.full_like(gs, np.nan)
    for a, g in enumerate(groups):
        for b, h in enumerate(groups):
            cells = [(i, j) for i in member[g] for j in member[h] if i != j]
            s = [score[i, j] for i, j in cells if not np.isnan(score[i, j])]
            d = [draw[i, j] for i, j in cells if not np.isnan(draw[i, j])]
            if s:
                gs[a, b], gd[a, b] = np.mean(s), np.mean(d)
    return groups, gs, gd


def draw_matrix(ax, score, draw, labels, title, annotate):
    n = len(labels)
    im = ax.imshow(score, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    small = n > 10
    ax.set_xticklabels(labels, rotation=90, fontsize=6 if small else 8.5, color=MUTED)
    ax.set_yticklabels(labels, fontsize=6 if small else 8.5, color=MUTED)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    if annotate:
        for i in range(n):
            for j in range(n):
                if not np.isnan(score[i, j]):
                    ax.text(j, i, f"{score[i, j]:.2f}\n{draw[i, j]:.0%} dr", ha="center", va="center",
                            fontsize=7.5, color=INK)
    return im


def ranking(score, labels):
    """Mean score against the agents each one has played, best first."""
    with np.errstate(all="ignore"):
        mean = np.nanmean(score, axis=1)
    order = np.argsort(-np.nan_to_num(mean, nan=-1))
    return [(labels[k], float(mean[k]), int(np.sum(~np.isnan(score[k])))) for k in order]


def plot_matrix(pairs, labels, out_png, label):
    score, draw = build_matrices(pairs, labels)
    groups, gs, gd = group_matrices(score, draw, labels)
    two = 1 < len(groups) < len(labels)
    n = len(labels)
    size = max(5.5, 0.55 * n + 3) if n <= 10 else max(7.5, 0.28 * n + 3)
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(size * (1.9 if two else 1.1), size),
                             facecolor=SURFACE, squeeze=False)
    im = draw_matrix(axes[0, 0], score, draw, labels,
                     "Score of row against column (0.5 = even)", annotate=n <= 8)
    if two:
        draw_matrix(axes[0, 1], gs, gd, groups,
                    "Mean over agent pairs of each group (diagonal: different seeds)",
                    annotate=True)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("score = (wins + 0.5 draws) / games", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(f"Head-to-head: {label} (orange: row loses, blue: row wins; dr = share of draws)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(out_png, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def save_ranking(rank, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "mean_score", "opponents"])
        w.writerows([(a, round(m, 4), k) for a, m, k in rank])


# ---------------------------------------------------------------- main
def expand(patterns):
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        out += hits if hits else [pat]
    return list(dict.fromkeys(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="*", default=[], help="checkpoint paths or globs (quote the globs)")
    p.add_argument("--label", required=True, help="name of the pool: results/h2h_<label>.csv")
    p.add_argument("--games", type=int, default=100, help="games per pair, sides alternate")
    p.add_argument("--parallel", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-only", action="store_true", help="do not play, redraw from the CSV")
    p.add_argument("--shard", default="0/1",
                   help="K/N: play only the pairs with number %% N == K (N jobs share one pool); "
                        "draw the figure afterwards with --plot-only")
    args = p.parse_args()

    k, n_shards = (int(x) for x in args.shard.split("/"))
    if not 0 <= k < n_shards:
        raise SystemExit("--shard K/N needs 0 <= K < N")
    if not args.plot_only:
        paths = expand(args.agents)
        if len(paths) < 2:
            raise SystemExit("need at least two agents")
        run_pairs(paths, args.games, args.parallel, args.seed, args.label, (k, n_shards))
        if n_shards > 1:
            print("shard done: when every shard has finished, draw the figure with --plot-only")
            return
    pairs = load_pairs(args.label)
    if not pairs:
        raise SystemExit(f"no pairs for pool {args.label!r} in {RESULTS}")
    labels = sorted({x for ab in pairs for x in ab}, key=sort_key)
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_matrix(pairs, labels, fig_dir / f"h2h_{args.label}.png", args.label)
    score, _ = build_matrices(pairs, labels)
    rank = ranking(score, labels)
    save_ranking(rank, RESULTS / f"h2h_{args.label}_ranking.csv")
    print(f"\nwrote {fig_dir / f'h2h_{args.label}.png'}")
    print(f"\n{'agent':<28}{'mean score':>11}{'opponents':>11}")
    for a, m, k in rank:
        print(f"{a:<28}{m:>11.3f}{k:>11}")


if __name__ == "__main__":
    main()
