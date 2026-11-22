"""Figures from results/diag_<run>.csv (src/diagnostics.py): seed 1, final models.

  results/figures/diag_stall.png    self-play: share of decided games and share of attack actions
  results/figures/diag_actions.png  self-play: action distribution per model

Usage:  python3 -I plots/plot_diagnostics.py
"""
import csv
from pathlib import Path

import matplotlib
from selection import keep_run
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("r0_s1", "R0"), ("r0_floor_s1", "R0 floor"), ("r1_floor_s1", "R1 floor"),
        ("r0_charge_s1", "R0 charge"), ("r1_s1", "R1"), ("r2_s1", "R2"),
        ("r3_s1", "R3"), ("r0_sched_s1", "R0 Sched"), ("r0_lam0_s1", "R0 λ=0")]
RUNS = [r for r in RUNS if keep_run(r[0])]  # only the variants listed in selection.py
OK, STALL = "#2a7ab0", "#d9534f"
ACTIONS = [("none", "none", "#c9ccd1"), ("back", "back", "#8fb8de"),
           ("forward", "forward", "#2a7ab0"), ("attack", "attack", "#f0a04b"),
           ("back_attack", "back + attack", "#d9534f"), ("forward_attack", "forward + attack", "#8e44ad")]


def load(run):
    rows = [r for r in csv.DictReader(open(f"results/diag_{run}.csv")) if r["matchup"] == "self"]
    return rows[-1]


def f(r, k):
    return float(r[k]) if r.get(k) not in (None, "") else 0.0


def main():
    data = [(label, load(run)) for run, label in RUNS]
    decided = [100 * (1 - f(r, "draw")) for _, r in data]
    attack = [100 * f(r, "attack_share") for _, r in data]
    colors = [STALL if d < 5 else OK for d in decided]
    names = [n for n, _ in data]
    ypos = list(range(len(names)))[::-1]

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, vals, title in ((axes[0], decided, "Decided self-play games (%)"),
                            (axes[1], attack, "Steps with an attack action (%)")):
        ax.barh(ypos, vals, color=colors, height=0.62)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.25)
        for y, v in zip(ypos, vals):
            if v > 85:
                ax.text(v - 1.5, y, f"{v:.0f}", va="center", ha="right", fontsize=10, color="white")
            else:
                ax.text(v + 1.5, y, f"{v:.0f}", va="center", fontsize=10)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(names)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=OK), plt.Rectangle((0, 0), 1, 1, color=STALL)],
               labels=["games decided", "stall (all timeouts)"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Stalled agents are not passive: they attack more than the others, but never finish the game (seed 1, final models)",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "diag_stall.png", dpi=200)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    left = [0.0] * len(data)
    special = [100 * f(r, "special_share") for _, r in data]
    for key, lab, col in ACTIONS + [("special", "special moves (charge)", "#3b3f45")]:
        vals = special if key == "special" else [100 * f(r, f"a_{key}") for _, r in data]
        ax.barh(ypos, vals, left=left, color=col, height=0.62, label=lab, edgecolor="white", linewidth=0.8)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of steps (%)")
    ax.set_title("Action distribution in self-play (seed 1, final models)", loc="left", fontsize=12.5)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out / "diag_actions.png", dpi=200)
    print("saved", out / "diag_stall.png", out / "diag_actions.png")


if __name__ == "__main__":
    # we draw the diagnostics in the same style as the paper figures (plot_local.fig_diag)
    import plot_local
    print("saved", *plot_local.fig_diag(("diag_stall.png", "diag_actions.png")))
