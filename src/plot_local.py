"""
Usage:  uv run python src/plot_local.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import head_to_head
import plot_diagnostics
import plot_report
from plot_paper_figures import LOSS, OUT, VARIANTS, WIN, checkpoint_curve  # also sets the rcParams

NAMES = {v: name for v, name, _ in VARIANTS} | {"r0_charge": "R0 + charge", "r0_lam0": "R0, λ = 0"}
COLORS = {v: col for v, _, col in VARIANTS}
DECIDED, STALL = "#1f4e8c", LOSS
ACTIONS = [("none", "none", "#c9ccd1"), ("back", "back", "#8fb8de"), ("forward", "forward", "#4c9fd6"),
           ("attack", "attack", "#e7a33e"), ("back_attack", "back + attack", "#d95f02"),
           ("forward_attack", "forward + attack", "#9e6bc4"), ("special", "special move (charge)", "#3b3f45")]
H2H_CMAP = LinearSegmentedColormap.from_list("h2h", [LOSS, "#f7f7f7", WIN])


def variant(run):
    """r0_sched_s1 -> r0_sched."""
    return run.rsplit("_s", 1)[0]


def fig_diag():
    data = [(NAMES[variant(run)], plot_diagnostics.load(run)) for run, _ in plot_diagnostics.RUNS]
    f = plot_diagnostics.f
    names = [n for n, _ in data]
    ypos = np.arange(len(names))[::-1]
    decided = [100 * (1 - f(r, "draw")) for _, r in data]
    attack = [100 * f(r, "attack_share") for _, r in data]
    colors = [STALL if d < 5 else DECIDED for d in decided]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, vals, title in ((axes[0], decided, "(a) decided games (%)"),
                            (axes[1], attack, "(b) steps with an attack action (%)")):
        ax.barh(ypos, vals, color=colors, height=0.62)
        for y, v in zip(ypos, vals):
            ax.text(v + 1.5, y, f"{v:.0f}", va="center", fontsize=10)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlim(0, 105)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(names)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (DECIDED, STALL)]
    fig.legend(handles, ["Some games decided", "Stall (all games time out)"],
               loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Self-play of the seed-1 final models: decided games and attack actions",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUT / "paper_diag_stall.png", dpi=200)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    left = np.zeros(len(data))
    for key, label, col in ACTIONS:
        share = "special_share" if key == "special" else f"a_{key}"
        vals = np.array([100 * f(r, share) for _, r in data])
        ax.barh(ypos, vals, left=left, color=col, height=0.62, label=label, edgecolor="white", linewidth=0.8)
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of steps (%)")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Action distribution in self-play of the seed-1 final models",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "paper_diag_actions.png", dpi=200)
    return "paper_diag_stall.png", "paper_diag_actions.png"


def fig_seeds():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    ypos = np.arange(len(VARIANTS))[::-1]
    for ax, (opp, title) in zip(axes, (("random", "(a) vs. random opponent"),
                                       ("footsies_bot", "(b) vs. FootsiesBot"))):
        for y, (v, _, col) in zip(ypos, VARIANTS):
            vals = [plot_report.final_rate(f"{v}_s{s}", opp) for s in range(1, 6)]
            ax.scatter(vals, [y] * 5, s=60, color=col, alpha=0.8, edgecolor="white", linewidth=1, zorder=3)
            ax.plot([np.mean(vals)] * 2, [y - 0.3, y + 0.3], color="black", linewidth=2, zorder=4)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("Win rate of the final model")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([name for _, name, _ in VARIANTS])
    handles = [plt.Line2D([], [], marker="o", ls="", color="grey", markersize=8),
               plt.Line2D([], [], color="black", linewidth=2)]
    fig.legend(handles, ["One seed", "Mean over 5 seeds"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Final win rate of each seed against heuristic opponents",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUT / "paper_report_seeds.png", dpi=200)
    return "paper_report_seeds.png"


def fig_noop():
    rows = [checkpoint_curve(v, "noop") for v, _, _ in VARIANTS]
    steps = rows[0][0]
    grid = np.array([vals.mean(axis=0) for _, vals in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{s / 1e6:.0f}" for s in steps])
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels([name for _, name, _ in VARIANTS])
    ax.set_xlabel("Environment step (M)")
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Win rate")
    fig.suptitle("Win rate against the no-op opponent during training:\nmean over 5 seeds",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "paper_report_noop.png", dpi=200)
    return "paper_report_noop.png"


def h2h_panel(ax, score, draw, labels, title, annotate):
    n = len(labels)
    im = ax.imshow(score, cmap=H2H_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7 if n > 10 else 9.5)
    ax.set_yticklabels(labels, fontsize=7 if n > 10 else 9.5)
    ax.set_title(title, loc="left", fontsize=12)
    for side in ax.spines.values():
        side.set_visible(False)
    if annotate:
        for i in range(n):
            for j in range(n):
                if not np.isnan(score[i, j]):
                    ax.text(j, i, f"{score[i, j]:.2f}\n{draw[i, j]:.0%} D", ha="center", va="center", fontsize=8)
    return im


def fig_h2h():
    pairs = head_to_head.load_pairs("final")
    labels = sorted({x for ab in pairs for x in ab}, key=head_to_head.sort_key)
    score, draw = head_to_head.build_matrices(pairs, labels)
    groups, gs, gd = head_to_head.group_matrices(score, draw, labels)
    short = lambda v: NAMES[v].split(" (")[0]  # "R0 Sched (PPO Sched.)" -> "R0 Sched"
    agents = [f"{short(head_to_head.variant_of(l))}, s{l.split('/')[0].rsplit('_s', 1)[1]}" for l in labels]
    variants = [short(g.split("/")[0]) for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), gridspec_kw={"width_ratios": [1.35, 1]})
    im = h2h_panel(axes[0], score, draw, agents, "(a) every pair of final models", annotate=False)
    h2h_panel(axes[1], gs, gd, variants, "(b) mean over the models of each variant", annotate=True)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75, pad=0.02, label="Score of row vs. column")
    fig.suptitle("Head-to-head between the final models: score = (wins + 0.5 draws) / games, 100 games per pair.\n"
                 "Green: row wins, orange: row loses. D = share of draws. "
                 "Diagonal of (b): different seeds of the same variant.", x=0.01, ha="left", fontsize=12.5)
    fig.savefig(OUT / "paper_h2h_final.png", dpi=200, bbox_inches="tight")
    return "paper_h2h_final.png"


def main():
    os.chdir(head_to_head.ROOT)  # the CSV paths are relative to the repository root
    OUT.mkdir(parents=True, exist_ok=True)
    print("saved", *fig_diag(), fig_seeds(), fig_noop(), fig_h2h())


if __name__ == "__main__":
    main()
