"""Figures in the style of the FootsiesGym paper (Sect. 4).

  results/figures/paper_br_outcomes.png   best-response training: win / draw / loss rates of the
                                          TARGET (the trained policy), as in the paper's Figure 4
  results/figures/paper_heuristics.png    win rate vs Random and vs No-op over training, mean over
                                          5 seeds with 95% confidence band, as in the paper's Figure 3

Usage:  python3 -I src/plot_paper_figures.py [--br-suffix _floor]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/figures")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
WIN, DRAW, LOSS = "#1b9e77", "#bdbdbd", "#d95f02"
T_975_DF4 = 2.776  # Student t quantile for a 95% interval with 5 seeds

VARIANTS = [("r0", "R0 (PPO)", "#4c9fd6"), ("r0_sched", "R0 Sched (PPO Sched.)", "#1f4e8c"),
            ("r1", "R1", "#2ca25f"), ("r2", "R2", "#9e6bc4"), ("r3", "R3", "#d95f02"),
            ("r0_floor", "R0 + floor", "#e7a33e"), ("r1_floor", "R1 + floor", "#c2185b")]
TARGETS = [("r0_s1", "R0"), ("r0_sched_s1", "R0 Sched"), ("r1_s1", "R1"), ("r2_s1", "R2"), ("r3_s1", "R3")]


def read_log(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda v: float(v) if v not in ("", "nan") else np.nan
    return {k: np.array([f(r[k]) for r in rows]) for k in ("steps", "episodes", "br_win", "br_draw", "br_loss")}


def binned_outcomes(logs, bin_steps=1_000_000):
    """Win/draw/loss rates of the best response per bin: episode-weighted within a BR seed, then
    averaged over BR seeds with equal weight (pooling episodes would let the seed that wins, and
    therefore plays more short games, dominate the average).

    Within a seed the rates count the games that FINISHED in the bin, so short games (decisive
    ones) are over-represented with respect to a fixed-size evaluation: the curves are a training
    signal, the final-evaluation bar is the measurement."""
    top = max(l["steps"].max() for l in logs)
    edges = np.arange(0, top + bin_steps, bin_steps)
    per_seed = []
    for l in logs:
        totals = np.zeros((len(edges) - 1, 3))
        ep = np.nan_to_num(l["episodes"])
        idx = np.clip(np.digitize(l["steps"], edges) - 1, 0, len(edges) - 2)
        for j, key in enumerate(("br_win", "br_draw", "br_loss")):
            np.add.at(totals[:, j], idx, np.nan_to_num(l[key]) * ep)
        per_seed.append(totals / np.maximum(totals.sum(axis=1, keepdims=True), 1))
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, np.mean(per_seed, axis=0)


def final_eval(target, suffix):
    """Mean over BR seeds of the final 500-game evaluation: (br_win, draw, br_loss) rates."""
    path = Path(f"results/br_{target}.csv")
    if not path.exists():
        return None, 0
    latest = {}
    for x in csv.DictReader(open(path)):
        run = x["br_run"]
        if run.endswith("_floor") == bool(suffix):
            g = int(x["games"])
            latest[run] = (int(x["br_wins"]) / g, int(x["draws"]) / g, int(x["br_losses"]) / g)
    if not latest:
        return None, 0
    return np.mean(list(latest.values()), axis=0), len(latest)


def fig_br(suffix):
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 3.6), sharey=True)
    for ax, (target, name) in zip(axes, TARGETS):
        logs = [read_log(p) for b in (1, 2, 3)
                if (p := Path(f"runs_br/br_{target}_b{b}{suffix}/log.csv")).exists()]
        if not logs:
            ax.set_title(f"{name}: no runs", loc="left")
            continue
        x, r = binned_outcomes(logs)
        # from the TARGET's point of view: target win = BR loss, target loss = BR win
        ax.stackplot(x / 1e6, r[:, 2], r[:, 1], r[:, 0], colors=[WIN, DRAW, LOSS], alpha=0.9)
        ev, n_ev = final_eval(target, suffix)
        xmax = x.max() / 1e6
        if ev is not None:
            bx, w = xmax + 2.2, 2.2
            ax.bar(bx, ev[2], width=w, color=WIN, edgecolor="black", linewidth=0.6)
            ax.bar(bx, ev[1], width=w, bottom=ev[2], color=DRAW, edgecolor="black", linewidth=0.6)
            ax.bar(bx, ev[0], width=w, bottom=ev[2] + ev[1], color=LOSS, edgecolor="black", linewidth=0.6)
            ax.text(bx, 1.02, f"final\nloss {ev[0]:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_title(f"{name} ({len(logs)} BR seeds)", loc="left", fontsize=11.5, pad=22)
        ax.set_xlim(0, xmax + 3.6)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlabel("BR environment steps (M)")
    axes[0].set_ylabel("Outcome rate of the target")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (LOSS, DRAW, WIN)]
    fig.legend(handles, ["Loss", "Draw", "Win"], loc="lower center", ncol=3, frameon=False)
    title = "Approximate best responses against the seed-1 final models"
    title += " (BR with advantage-std floor)" if suffix else ""
    fig.suptitle(title + ": outcomes from the target's point of view.\n"
                 "Areas: games finished during BR training (mean over BR seeds; decisive games over-represented). "
                 "Bar: final evaluation, 500 games per BR seed.", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    name = f"paper_br_outcomes{suffix}.png"
    fig.savefig(OUT / name, dpi=200)
    return name


def checkpoint_curve(variant, opponent):
    """Per-seed win rate at each evaluated checkpoint (latest evaluation per step)."""
    per_seed = []
    for s in range(1, 6):
        run = f"{variant}_s{s}"
        latest = {}
        for x in csv.DictReader(open(f"results/eval_{run}.csv")):
            if x["run"] == run and x["opponent"] == opponent:
                latest[int(x["steps"])] = float(x["win_rate"])
        per_seed.append(dict(sorted(latest.items())))
    steps = sorted(set.intersection(*[set(d) for d in per_seed]))
    vals = np.array([[d[k] for k in steps] for d in per_seed])
    return np.array(steps), vals


def fig_heuristics():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (opp, title) in zip(axes, (("random", "(a) vs. random opponent"), ("noop", "(b) vs. no-op opponent"))):
        for v, name, col in VARIANTS:
            steps, vals = checkpoint_curve(v, opp)
            mean = vals.mean(axis=0)
            half = T_975_DF4 * vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
            ax.plot(steps / 1e6, mean, marker="o", color=col, linewidth=2, label=name)
            ax.fill_between(steps / 1e6, np.clip(mean - half, 0, 1), np.clip(mean + half, 0, 1),
                            color=col, alpha=0.10)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("Environment step (M)")
        ax.set_xlim(4, 21)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Win rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Win rate against heuristic opponents during training: mean over 5 seeds, 95% CI "
                 "(checkpoints at 5M, 10M, 15M, 20M)", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(OUT / "paper_heuristics.png", dpi=200)
    return "paper_heuristics.png"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--br-suffix", default="", help="'' for the plain BRs, '_floor' for the floor BRs")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("saved", fig_br(args.br_suffix), fig_heuristics())
