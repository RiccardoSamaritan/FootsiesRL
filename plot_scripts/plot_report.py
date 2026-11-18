"""Figures for the report (results/figures/report_*.png).

  report_seeds.png      one dot per seed: final win rate vs Random and vs FootsiesBot, per variant
  report_stall.png      timeouts and advantage std during training, R0 seed 5 with and without the floor
  report_br_curves.png  best-response win rate during training, two BR seeds per target
  report_noop.png       win rate vs No-op at 5M, 10M, 15M and 20M steps, mean over seeds

Usage:  python3 -I plots/plot_report.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/figures")
VARIANTS = [("r0", "R0"), ("r0_sched", "R0 Sched"), ("r1", "R1"), ("r2", "R2"), ("r3", "R3"),
            ("r0_floor", "R0 con correzione"), ("r1_floor", "R1 con correzione")]
BASE, FLOOR, INK = "#2a7ab0", "#e07b39", "#333333"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def eval_rows(run):
    return list(csv.DictReader(open(f"results/eval_{run}.csv")))


def final_rate(run, opp):
    best = None
    for x in eval_rows(run):
        if x["opponent"] == opp and x["label"] in (f"{run}-final", f"{run}-final-vec", f"{run}-final-bot"):
            best = float(x["win_rate"])
    return best


def column(path, key):
    rows = list(csv.DictReader(open(path)))
    f = lambda v: float(v) if v not in ("", "nan") else np.nan
    return np.array([f(r["steps"]) for r in rows]), np.array([f(r[key]) for r in rows])


def smooth(y, w):
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w + 1): i + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg)
    return out


def fig_seeds():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ypos = np.arange(len(VARIANTS))[::-1]
    for ax, opp, title in ((axes[0], "random", "Contro Random"), (axes[1], "footsies_bot", "Contro FootsiesBot")):
        for y, (v, _) in zip(ypos, VARIANTS):
            vals = [final_rate(f"{v}_s{i}", opp) for i in range(1, 6)]
            col = FLOOR if "floor" in v else BASE
            ax.scatter(vals, [y] * 5, s=60, color=col, alpha=0.75, edgecolor="white", linewidth=1, zorder=3)
            ax.plot([np.mean(vals)] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=2.2, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("win rate del modello finale")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([n for _, n in VARIANTS])
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BASE, markersize=8),
                        plt.Line2D([], [], marker="o", ls="", color=FLOOR, markersize=8),
                        plt.Line2D([], [], color=INK, linewidth=2.2)],
               labels=["un seed", "un seed, con correzione sui vantaggi", "media dei 5 seed"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("I seed di una stessa variante sono più sparsi delle medie tra varianti", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUT / "report_seeds.png", dpi=200)


def fig_stall():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    for run, lab, col in (("r0_s5", "R0 seed 5", BASE), ("r0_floor_s5", "R0 seed 5, con correzione", FLOOR)):
        s, t = column(f"runs/{run}/log.csv", "timeout_rate")
        _, a = column(f"runs/{run}/log.csv", "adv_std")
        axes[0].plot(s / 1e6, 100 * smooth(t, 60), color=col, linewidth=2, label=lab)
        axes[1].plot(s / 1e6, smooth(a, 60), color=col, linewidth=2)
    axes[0].set_ylabel("partite in timeout (%)")
    axes[0].set_ylim(0, 102)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("dev. std. dei vantaggi")
    axes[1].set_xlabel("passi di addestramento (milioni)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Quando tutte le partite vanno in timeout, la deviazione standard dei vantaggi crolla",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "report_stall.png", dpi=200)


def fig_br():
    targets = [("r0_s1", "R0"), ("r0_sched_s1", "R0 Sched"), ("r1_s1", "R1"), ("r2_s1", "R2"), ("r3_s1", "R3")]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.4), sharey=True)
    for ax, (t, name) in zip(axes, targets):
        for b, col in ((1, BASE), (2, FLOOR)):
            p = Path(f"runs_br/br_{t}_b{b}/log.csv")
            if not p.exists():
                continue
            s, w = column(p, "br_win")
            ax.plot(s / 1e6, 100 * smooth(w, 150), color=col, linewidth=2, label=f"best response, seed {b}")
        ax.set_title(f"contro {name}", loc="left", fontsize=11.5)
        ax.set_ylim(-2, 102)
        ax.set_xlim(0, 20)
        ax.grid(alpha=0.25)
        ax.set_xlabel("passi (milioni)")
    axes[0].set_ylabel("vittorie della best response (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("La best response dipende dal proprio seed: contro R0, Sched e R3 un seed esce dai pareggi, l'altro no",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(OUT / "report_br_curves.png", dpi=200)


def fig_noop():
    marks = [5, 10, 15, 20]
    grid = np.full((len(VARIANTS), len(marks)), np.nan)
    for i, (v, _) in enumerate(VARIANTS):
        per = {m: [] for m in marks}
        for sd in range(1, 6):
            run = f"{v}_s{sd}"
            latest = {}
            for x in eval_rows(run):
                if x["opponent"] == "noop" and x["run"] == run:
                    m = min(marks, key=lambda k: abs(k * 1e6 - float(x["steps"])))
                    latest[m] = float(x["win_rate"])
            for m, w in latest.items():
                per[m].append(w)
        for j, m in enumerate(marks):
            if per[m]:
                grid[i, j] = np.mean(per[m])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(100 * grid, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                v = 100 * grid[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > 55 else INK)
    ax.set_xticks(range(len(marks)))
    ax.set_xticklabels([f"{m}M" for m in marks])
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels([n for _, n in VARIANTS])
    ax.set_xlabel("passi di addestramento")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Vittorie contro No-op (%), media sui seed:\nla capacità di battere un avversario fermo si perde",
                 loc="left", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(OUT / "report_noop.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_seeds, fig_stall, fig_br, fig_noop):
        f()
        print("ok", f.__name__)
