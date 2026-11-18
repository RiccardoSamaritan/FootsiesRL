"""Figures from results/diag_<run>.csv (src/diagnostics.py): seed 1, final models.

  results/figures/diag_stall.png    self-play: share of decided games and share of attack actions
  results/figures/diag_actions.png  self-play: action distribution per model

Usage:  python3 -I plots/plot_diagnostics.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("r0_s1", "R0"), ("r0_floor_s1", "R0 floor"), ("r1_floor_s1", "R1 floor"),
        ("r0_charge_s1", "R0 charge"), ("r1_s1", "R1"), ("r2_s1", "R2"),
        ("r3_s1", "R3"), ("r0_sched_s1", "R0 Sched"), ("r0_lam0_s1", "R0 λ=0")]
OK, STALL = "#2a7ab0", "#d9534f"
ACTIONS = [("none", "nessuna", "#c9ccd1"), ("back", "indietro", "#8fb8de"),
           ("forward", "avanti", "#2a7ab0"), ("attack", "attacco", "#f0a04b"),
           ("back_attack", "attacco indietro", "#d9534f"), ("forward_attack", "attacco avanti", "#8e44ad")]


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
    for ax, vals, title in ((axes[0], decided, "Partite decise in self-play (%)"),
                            (axes[1], attack, "Passi con un'azione di attacco (%)")):
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
               labels=["partite decise", "stallo (tutti timeout)"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Gli agenti in stallo non sono passivi: attaccano di più degli altri, ma non concludono (seed 1, modelli finali)",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "diag_stall.png", dpi=200)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    left = [0.0] * len(data)
    special = [100 * f(r, "special_share") for _, r in data]
    for key, lab, col in ACTIONS + [("special", "mosse speciali (charge)", "#3b3f45")]:
        vals = special if key == "special" else [100 * f(r, f"a_{key}") for _, r in data]
        ax.barh(ypos, vals, left=left, color=col, height=0.62, label=lab, edgecolor="white", linewidth=0.8)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("quota dei passi (%)")
    ax.set_title("Distribuzione delle azioni in self-play (seed 1, modelli finali)", loc="left", fontsize=12.5)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out / "diag_actions.png", dpi=200)
    print("saved", out / "diag_stall.png", out / "diag_actions.png")


if __name__ == "__main__":
    main()
