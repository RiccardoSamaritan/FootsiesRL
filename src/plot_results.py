"""Figures from training logs (runs/<run>/log.csv) and evaluations (results/eval_<run>.csv).

    uv run python src/plot_results.py                          # every run in runs/
    uv run python src/plot_results.py --runs r0_s1 r1_s1 r2_s1

Writes PNG files to results/figures/:
  training_curves.png  per-run training diagnostics (episode length, timeouts,
                       entropy, critic explained variance, advantage std, KL)
  eval_curves.png      win rate and draw rate of the checkpoints against the
                       fixed opponents, as a function of training steps

Colors follow the reward variant (r0 blue, r1 orange, r2 aqua, r3 yellow), so a
run keeps its color in every figure.
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REWARD_SLOT = {"r0": 0, "r1": 1, "r2": 2, "r3": 3}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FIXED_OPPONENTS = ["random", "footsies_bot", "noop", "back"]
OPPONENT_TITLES = {"random": "vs random", "footsies_bot": "vs FootsiesBot",
                   "noop": "vs noop (stands still)", "back": "vs back (holds back)"}


def run_colors(runs):
    """One color per run: by reward variant (r0..r3), other runs take the next free slots."""
    colors, spare = {}, iter(SLOTS[4:])
    for run in runs:
        m = re.match(r"(r\d)", run)
        if m and m.group(1) in REWARD_SLOT:
            colors[run] = SLOTS[REWARD_SLOT[m.group(1)]]
        else:
            colors[run] = next(spare, "#52514e")
    return colors


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def read_log(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
    return {k: col(k) for k in ["steps", "episodes", "ep_len", "timeout_rate", "entropy",
                                "explained_var", "return_std", "adv_std", "approx_kl"]}


def rolling(values, weights, w):
    """Moving average over w updates of `values`, weighted by `weights` (nan values are skipped)."""
    ok = ~np.isnan(values) & (weights > 0)
    num = np.convolve(np.where(ok, values * weights, 0.0), np.ones(w), mode="valid")
    den = np.convolve(np.where(ok, weights, 0.0), np.ones(w), mode="valid")
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def plot_training(runs, runs_dir, out, smooth):
    logs = {r: read_log(runs_dir / r / "log.csv") for r in runs}
    colors = run_colors(runs)
    panels = [
        ("Episode length (steps, log scale)", "ep_len", "episodes", "log", None),
        ("Share of episodes ending by timeout", "timeout_rate", "episodes", "linear", (0, 1.02)),
        ("Policy entropy (max 1.79)", "entropy", None, "linear", None),
        ("Critic explained variance (where returns vary)", "explained_var", "signal", "linear", (-1, 1)),
        ("Advantage std (log scale)", "adv_std", None, "log", None),
        ("Approx. KL between policies, per update", "approx_kl", None, "linear", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE, sharex=True)
    for ax, (title, key, weight, scale, ylim) in zip(axes.flat, panels):
        style(ax)
        for run, log in logs.items():
            w = min(smooth, len(log["steps"]))
            x = log["steps"][w - 1:] / 1e6
            if weight == "episodes":
                y = rolling(log[key], log["episodes"], w)
            elif weight == "signal":  # explained variance is meaningless when returns are ~constant
                mask = np.where(log["return_std"] > 0.5, 1.0, 0.0)
                y = rolling(log[key], mask, w)
            else:
                y = rolling(log[key], np.ones_like(log[key]), w)
            ax.plot(x, y, color=colors[run], linewidth=2, label=run)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_yscale(scale)
        if ylim:
            ax.set_ylim(*ylim)
    for ax in axes[-1]:
        ax.set_xlabel("training steps (millions)", color=MUTED, fontsize=9)
    if len(runs) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(runs), frameon=False,
                   fontsize=10, labelcolor=INK)
    fig.suptitle(f"Training diagnostics (moving average over {smooth} updates)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def load_eval(paths, runs_dir):
    """Latest evaluation of each (run, steps, opponent) against a fixed opponent."""
    final_steps = {}
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    latest = {}
    for r in rows:
        if not r.get("run"):
            # old-format rows: derive the run name from the agent path
            r["run"] = Path(r["agent"]).parent.name
        if not r["run"] or r["opponent"] not in FIXED_OPPONENTS:
            continue
        steps = r.get("steps", "")
        if not steps:
            m = re.search(r"ckpt_(\d+)", r["agent"])
            if m:
                steps = m.group(1)
            elif Path(r["agent"]).stem == "final":  # last logged step of the run
                log = runs_dir / r["run"] / "log.csv"
                if r["run"] not in final_steps and log.exists():
                    final_steps[r["run"]] = int(read_log(log)["steps"][-1])
                steps = final_steps.get(r["run"], "")
        if not steps:
            continue
        key = (r["run"], int(float(steps)), r["opponent"])
        if key not in latest or r["timestamp"] >= latest[key]["timestamp"]:
            latest[key] = {**r, "steps": int(float(steps))}
    return list(latest.values())


def plot_eval(rows, runs, out):
    opponents = [o for o in FIXED_OPPONENTS if any(r["opponent"] == o for r in rows)]
    colors = run_colors(runs)
    fig, axes = plt.subplots(2, len(opponents), figsize=(3.6 * len(opponents) + 1, 6.6),
                             facecolor=SURFACE, sharex=True, squeeze=False)
    for j, opp in enumerate(opponents):
        for run in runs:
            pts = sorted((r for r in rows if r["run"] == run and r["opponent"] == opp),
                         key=lambda r: r["steps"])
            if not pts:
                continue
            x = np.array([p["steps"] for p in pts]) / 1e6
            games = np.array([float(p["games"]) for p in pts])
            win = np.array([float(p["win_rate"]) for p in pts])
            lo = np.array([float(p["ci_low"]) for p in pts])
            hi = np.array([float(p["ci_high"]) for p in pts])
            draw = np.array([float(p["draws"]) for p in pts]) / games
            axes[0, j].errorbar(x, win, yerr=[win - lo, hi - win], color=colors[run], linewidth=2,
                                marker="o", markersize=6, capsize=3, label=run)
            axes[1, j].plot(x, draw, color=colors[run], linewidth=2, marker="o", markersize=6, label=run)
        for i, ylabel in enumerate(["win rate (95% CI)", "share of draws (timeouts)"]):
            ax = axes[i, j]
            style(ax)
            ax.set_ylim(-0.02, 1.02)
            if j == 0:
                ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        axes[0, j].set_title(OPPONENT_TITLES[opp], loc="left", fontsize=10.5, color=INK)
        axes[1, j].set_xlabel("checkpoint (millions of steps)", color=MUTED, fontsize=9)
    if len(runs) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(runs), frameon=False,
                   fontsize=10, labelcolor=INK)
    fig.suptitle("Checkpoints against fixed opponents", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", help="run names (default: every run with a log.csv)")
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--eval-csv", nargs="*",
                   help="evaluation CSVs (default: every results/eval*.csv)")
    p.add_argument("--out", default=str(ROOT / "results" / "figures"))
    p.add_argument("--smooth", type=int, default=100, help="moving-average window, in updates")
    args = p.parse_args()

    runs_dir, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = args.runs or sorted(d.name for d in runs_dir.iterdir() if (d / "log.csv").exists())
    if not runs:
        raise SystemExit(f"no runs with a log.csv in {runs_dir}")
    plot_training(runs, runs_dir, out / "training_curves.png", args.smooth)
    print(f"wrote {out / 'training_curves.png'} ({len(runs)} runs)")

    eval_files = [Path(f) for f in args.eval_csv] if args.eval_csv else \
        sorted((ROOT / "results").glob("eval*.csv"))
    rows = [r for r in load_eval(eval_files, runs_dir) if r["run"] in runs]
    if rows:
        plot_eval(rows, runs, out / "eval_curves.png")
        print(f"wrote {out / 'eval_curves.png'} ({len(rows)} evaluations)")
    else:
        print("no checkpoint evaluations in results/eval_*.csv yet: eval_curves.png not written")


if __name__ == "__main__":
    main()
