"""Which runs end in a stall: one cell per variant and seed.

A run ends in a stall if, in the last 10% of its updates, on average more than 99% of the
self-play games end in a timeout (column timeout_rate of runs/<run>/log.csv, present in every
run). The number in each cell is that share of timeouts, in percent.

  results/figures/stall_grid.png

Usage:  uv run python plot_scripts/plot_stall_grid.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_paper_figures import LOSS  # paper palette (also sets the rcParams)

OUT = Path("results/figures")
VARIANTS = [("r0", "R0"), ("r0_sched", "R0 Sched"), ("r1", "R1"), ("r2", "R2"), ("r3", "R3"),
            ("r0_floor", "R0 + floor"), ("r1_floor", "R1 + floor")]
SEEDS = range(1, 6)
LAST = 0.10       # final part of training that we look at
THRESHOLD = 0.99  # share of timeouts above which we call it a stall
STALL, DECIDED = LOSS, "#1f4e8c"


def final_timeout_share(run):
    """Mean timeout share of the self-play games in the last 10% of the updates."""
    rows = list(csv.DictReader(open(f"runs/{run}/log.csv")))
    t = np.array([float(r["timeout_rate"]) if r.get("timeout_rate") not in (None, "", "nan") else np.nan
                  for r in rows])
    return float(np.nanmean(t[int(len(t) * (1 - LAST)):]))


def main():
    share = np.array([[final_timeout_share(f"{v}_s{s}") for s in SEEDS] for v, _ in VARIANTS])
    stall = share > THRESHOLD

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i in range(len(VARIANTS)):
        for j in range(len(SEEDS)):
            ax.add_patch(plt.Rectangle((j, i), 0.92, 0.86, color=STALL if stall[i, j] else DECIDED))
            ax.text(j + 0.46, i + 0.43, f"{100 * share[i, j]:.0f}%", ha="center", va="center",
                    color="white", fontsize=10)
        ax.text(len(SEEDS) + 0.15, i + 0.43, f"{stall[i].sum()}/{len(SEEDS)}", va="center", fontsize=11)
    ax.set_xlim(0, len(SEEDS) + 0.8)
    ax.set_ylim(len(VARIANTS), 0)
    ax.set_xticks([j + 0.46 for j in range(len(SEEDS))])
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_yticks([i + 0.43 for i in range(len(VARIANTS))])
    ax.set_yticklabels([n for _, n in VARIANTS])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (STALL, DECIDED)]
    fig.legend(handles, ["Stall: > 99% of games time out", "Games still decided"],
               loc="lower center", ncol=2, frameon=False)
    n_plain = stall[:5].sum()
    n_floor = stall[5:].sum()
    fig.suptitle(f"Runs that end in a stall: {n_plain} of 25 without the floor, {n_floor} of 10 with it.\n"
                 "Cell value: share of self-play games ending in timeout over the last 10% of training",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "stall_grid.png", dpi=200)
    print("saved", OUT / "stall_grid.png")


if __name__ == "__main__":
    main()
