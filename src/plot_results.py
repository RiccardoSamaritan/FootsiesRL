"""Figures from training logs (runs/<run>/log.csv) and evaluations (results/eval_<run>.csv).

    uv run python src/plot_results.py                          # every run in runs/, grouped by variant
    uv run python src/plot_results.py --runs r0_s1 r1_s1 r2_s1
    uv run python src/plot_results.py --per-run                # one line per run, no grouping

Runs are grouped by variant: the run name without its seed suffix (r1_s3 -> r1,
r0_sched_s2 -> r0_sched). A variant with several seeds is drawn as the mean over
seeds with a band from the minimum to the maximum seed; a variant with a single
run is drawn as that run (evaluation points keep their 95% Wilson interval).

Writes to results/figures/:
  training_curves.png   training diagnostics (episode length, timeouts, entropy,
                        critic explained variance, advantage std, KL)
  selfplay_curves.png   self-play outcomes (share of decided games, P1 minus P2 win
                        rate, mean |return|, mean |training reward|); only when the
                        logs have these columns (runs launched after they were added)
  eval_curves.png       win rate and draw rate of the checkpoints against the fixed
                        opponents, as a function of training steps
and results/summary_final.csv: final checkpoint against each fixed opponent, mean and
min-max over the seeds of each variant.

Colors follow the reward variant (r0 blue, r1 orange, r2 aqua, r3 yellow); other
variants (r0_sched, r1_floor, ...) take the next free colors.
"""
import argparse
import csv
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REWARD_SLOT = {"r0": 0, "r1": 1, "r2": 2, "r3": 3}
# fixed colors of the variants that are not a plain reward, so they never look like another variant
VARIANT_COLOR = {"r0_sched": "#e87ba4", "r1_floor": "#6250d6"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FIXED_OPPONENTS = ["random", "footsies_bot", "noop", "back"]
OPPONENT_TITLES = {"random": "vs random", "footsies_bot": "vs FootsiesBot",
                   "noop": "vs noop (stands still)", "back": "vs back (holds back)"}
LOG_COLUMNS = ["steps", "episodes", "ep_len", "timeout_rate", "entropy",
               "explained_var", "return_std", "adv_std", "approx_kl"]
# columns added later to train.py: missing (all nan) in older runs
OPTIONAL_COLUMNS = ["p1_win_rate", "p2_win_rate", "decided_rate", "abs_return", "train_abs_reward"]


# ---------------------------------------------------------------- grouping and colors
def variant_of(run):
    """Run name without the seed suffix: r1_s3 -> r1, r0_sched_s2 -> r0_sched."""
    return re.sub(r"_s\d+$", "", run)


def make_series(runs, per_run):
    """List of series {label, key, members}: one per variant, or one per run with --per-run."""
    if per_run:
        return [{"label": r, "key": r, "members": [r]} for r in runs]
    groups = {}
    for r in runs:
        groups.setdefault(variant_of(r), []).append(r)
    return [{"label": f"{v} (n={len(m)})" if len(m) > 1 else m[0], "key": v, "members": m}
            for v, m in groups.items()]


def series_colors(series):
    """The plain reward variants (r0..r3, any seed) keep one color each; every other
    series (r0_sched, r1_floor, ...) takes the next free slot."""
    colors, spare = {}, iter(SLOTS[4:])
    for s in series:
        key = variant_of(s["key"])
        if key in REWARD_SLOT:
            colors[s["key"]] = SLOTS[REWARD_SLOT[key]]
        else:
            colors[s["key"]] = VARIANT_COLOR.get(variant_of(s["key"]), next(spare, MUTED))
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


# ---------------------------------------------------------------- training logs
def read_log(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def col(k):
        return np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])

    log = {k: col(k) for k in LOG_COLUMNS + OPTIONAL_COLUMNS}
    log["side_gap"] = log["p1_win_rate"] - log["p2_win_rate"]
    return log


def rolling(values, weights, w):
    """Moving average over w updates of `values`, weighted by `weights` (nan values are skipped)."""
    ok = ~np.isnan(values) & (weights > 0)
    num = np.convolve(np.where(ok, values * weights, 0.0), np.ones(w), mode="valid")
    den = np.convolve(np.where(ok, weights, 0.0), np.ones(w), mode="valid")
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def smoothed(log, key, weight, w):
    """Smoothed curve of one run (the x axis is the steps of that run)."""
    w = min(w, len(log["steps"]))
    if weight == "episodes":
        y = rolling(log[key], log["episodes"], w)
    elif weight == "signal":  # explained variance is meaningless when returns are ~constant
        y = rolling(log[key], np.where(log["return_std"] > 0.5, 1.0, 0.0), w)
    else:
        y = rolling(log[key], np.ones_like(log[key]), w)
    return log["steps"][w - 1:] / 1e6, y


def draw_series(ax, curves, color, label):
    """Mean over the members with a min-max band (a single member: just the line).

    Members of one variant share the same update schedule, so the curves are cut to the
    shortest one and compared update by update.
    """
    curves = [(x, y) for x, y in curves if not np.all(np.isnan(y))]
    if not curves:
        return
    n = min(len(y) for _, y in curves)
    x = curves[0][0][:n]
    ys = np.vstack([y[:n] for _, y in curves])
    if len(ys) > 1:
        with warnings.catch_warnings():  # all-nan columns (no finished game) give a warning
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, lo, hi = np.nanmean(ys, axis=0), np.nanmin(ys, axis=0), np.nanmax(ys, axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
    else:
        mean = ys[0]
    ax.plot(x, mean, color=color, linewidth=2, label=label)


def plot_panels(series, logs, colors, panels, out, smooth, shape, title):
    fig, axes = plt.subplots(*shape, figsize=(4.3 * shape[1], 3.5 * shape[0]), facecolor=SURFACE,
                             sharex=True, squeeze=False)
    for ax, (ptitle, key, weight, scale, ylim) in zip(axes.flat, panels):
        style(ax)
        for s in series:
            curves = [smoothed(logs[r], key, weight, smooth) for r in s["members"]]
            draw_series(ax, curves, colors[s["key"]], s["label"])
        ax.set_title(ptitle, loc="left", fontsize=10.5, color=INK)
        ax.set_yscale(scale)
        if ylim:
            ax.set_ylim(*ylim)
    for ax in axes[-1]:
        ax.set_xlabel("training steps (millions)", color=MUTED, fontsize=9)
    if len(series) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False,
                   fontsize=10, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_training(series, runs, runs_dir, colors, out_dir, smooth):
    logs = {r: read_log(runs_dir / r / "log.csv") for r in runs}
    band = "; line: mean over seeds, band: min-max" if any(len(s["members"]) > 1 for s in series) else ""
    plot_panels(series, logs, colors, [
        ("Episode length (steps, log scale)", "ep_len", "episodes", "log", None),
        ("Share of episodes ending by timeout", "timeout_rate", "episodes", "linear", (0, 1.02)),
        ("Policy entropy (max 1.79)", "entropy", None, "linear", None),
        ("Critic explained variance (where returns vary)", "explained_var", "signal", "linear", (-1, 1)),
        ("Advantage std (log scale)", "adv_std", None, "log", None),
        ("Approx. KL between policies, per update", "approx_kl", None, "linear", None),
    ], out_dir / "training_curves.png", smooth, (2, 3),
        f"Training diagnostics (moving average over {smooth} updates{band})")
    print(f"wrote {out_dir / 'training_curves.png'} ({len(runs)} runs)")

    if any(not np.all(np.isnan(logs[r]["decided_rate"])) for r in runs):
        plot_panels(series, logs, colors, [
            ("Share of decided games (not timeouts)", "decided_rate", "episodes", "linear", (0, 1.02)),
            ("P1 win rate minus P2 win rate (side bias)", "side_gap", "episodes", "linear", (-1, 1)),
            ("Mean |raw return| of finished games", "abs_return", "episodes", "linear", None),
            ("Mean |training reward| per step (log scale)", "train_abs_reward", None, "log", None),
        ], out_dir / "selfplay_curves.png", smooth, (2, 2),
            f"Self-play outcomes (moving average over {smooth} updates{band})")
        print(f"wrote {out_dir / 'selfplay_curves.png'}")


# ---------------------------------------------------------------- evaluations
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


def eval_points(rows, series, opp):
    """Per training step: win rate, interval and draw rate of each member of the series."""
    by_step = {}
    for r in rows:
        if r["run"] in series["members"] and r["opponent"] == opp:
            by_step.setdefault(r["steps"], []).append(
                {"win": float(r["win_rate"]), "lo": float(r["ci_low"]), "hi": float(r["ci_high"]),
                 "draw": float(r["draws"]) / float(r["games"])})
    return dict(sorted(by_step.items()))


def plot_eval(rows, series, colors, out):
    opponents = [o for o in FIXED_OPPONENTS if any(r["opponent"] == o for r in rows)]
    fig, axes = plt.subplots(2, len(opponents), figsize=(3.6 * len(opponents) + 1, 6.6),
                             facecolor=SURFACE, sharex=True, squeeze=False)
    for j, opp in enumerate(opponents):
        for s in series:
            pts = eval_points(rows, s, opp)
            if not pts:
                continue
            color = colors[s["key"]]
            x = np.array(list(pts)) / 1e6
            multi = max(len(v) for v in pts.values()) > 1
            for i, field in enumerate(["win", "draw"]):
                ax = axes[i, j]
                vals = [np.array([p[field] for p in v]) for v in pts.values()]
                mean = np.array([v.mean() for v in vals])
                if multi:  # several seeds: mean line, min-max band and one dot per seed
                    ax.fill_between(x, [v.min() for v in vals], [v.max() for v in vals],
                                    color=color, alpha=0.10, linewidth=0)
                    ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=5, label=s["label"])
                    for xi, v in zip(x, vals):
                        ax.scatter([xi] * len(v), v, color=color, s=8, alpha=0.5, zorder=3)
                elif field == "win":  # one run: 95% Wilson interval of the evaluation games
                    lo = np.array([v[0]["lo"] for v in pts.values()])
                    hi = np.array([v[0]["hi"] for v in pts.values()])
                    ax.errorbar(x, mean, yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)], color=color, linewidth=2,
                                marker="o", markersize=6, capsize=3, label=s["label"])
                else:
                    ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=6, label=s["label"])
        for i, ylabel in enumerate(["win rate", "share of draws (timeouts)"]):
            ax = axes[i, j]
            style(ax)
            ax.set_ylim(-0.02, 1.02)
            if j == 0:
                ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        axes[0, j].set_title(OPPONENT_TITLES[opp], loc="left", fontsize=10.5, color=INK)
        axes[1, j].set_xlabel("checkpoint (millions of steps)", color=MUTED, fontsize=9)
    if len(series) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False,
                   fontsize=10, labelcolor=INK)
    fig.suptitle("Checkpoints against fixed opponents", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.945, "several seeds: mean line, min-max band, dots = seeds; one run: 95% CI",
             ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def summarize_final(rows, series, out):
    """Final checkpoint of each run against each fixed opponent, summarized per variant."""
    table = []
    for s in series:
        for opp in FIXED_OPPONENTS:
            finals = []
            for run in s["members"]:
                mine = [r for r in rows if r["run"] == run and r["opponent"] == opp]
                if mine:
                    last = max(mine, key=lambda r: r["steps"])
                    games = float(last["games"])
                    finals.append((float(last["win_rate"]), float(last["draws"]) / games,
                                   float(last["losses"]) / games, last["steps"]))
            if not finals:
                continue
            win, draw, loss, steps = (np.array(c) for c in zip(*finals))
            table.append({"variant": s["key"], "opponent": opp, "seeds": len(finals),
                          "steps_min": int(steps.min()),
                          "win_mean": round(win.mean(), 4), "win_min": round(win.min(), 4),
                          "win_max": round(win.max(), 4), "draw_mean": round(draw.mean(), 4),
                          "loss_mean": round(loss.mean(), 4)})
    if not table:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\nfinal checkpoint, mean [min-max] over seeds -> {out}")
    print(f"{'variant':<14}{'opponent':<14}{'seeds':>6}{'win':>8}{'min':>8}{'max':>8}{'draw':>8}")
    for t in table:
        print(f"{t['variant']:<14}{t['opponent']:<14}{t['seeds']:>6}{t['win_mean']:>8.3f}"
              f"{t['win_min']:>8.3f}{t['win_max']:>8.3f}{t['draw_mean']:>8.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", help="run names (default: every run with a log.csv)")
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--eval-csv", nargs="*",
                   help="evaluation CSVs (default: every results/eval*.csv)")
    p.add_argument("--out", default=str(ROOT / "results" / "figures"))
    p.add_argument("--summary", default=str(ROOT / "results" / "summary_final.csv"))
    p.add_argument("--smooth", type=int, default=100, help="moving-average window, in updates")
    p.add_argument("--per-run", action="store_true",
                   help="one line per run instead of one series per variant (mean over seeds)")
    args = p.parse_args()

    runs_dir, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = args.runs or sorted(d.name for d in runs_dir.iterdir() if (d / "log.csv").exists())
    if not runs:
        raise SystemExit(f"no runs with a log.csv in {runs_dir}")
    series = make_series(runs, args.per_run)
    colors = series_colors(series)
    plot_training(series, runs, runs_dir, colors, out, args.smooth)

    eval_files = [Path(f) for f in args.eval_csv] if args.eval_csv else \
        sorted((ROOT / "results").glob("eval*.csv"))
    rows = [r for r in load_eval(eval_files, runs_dir) if r["run"] in runs]
    if rows:
        plot_eval(rows, series, colors, out / "eval_curves.png")
        print(f"wrote {out / 'eval_curves.png'} ({len(rows)} evaluations)")
        summarize_final(rows, series, Path(args.summary))
    else:
        print("no checkpoint evaluations in results/eval_*.csv yet: eval_curves.png not written")


if __name__ == "__main__":
    main()
