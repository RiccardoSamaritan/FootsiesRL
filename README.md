# FootsiesRL

PPO agents for the fighting game **Footsies** ([FootsiesGym](https://pypi.org/project/footsies-gym/)),
trained in self-play. The project compares training rewards
and asks whether an agent that beats simple opponents is also hard to exploit.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and macOS or Linux (on Windows see below). Python 3.12
and all packages are pinned by `.python-version` and `uv.lock`.

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: brew install uv
uv sync                                           # creates .venv
uv run python test.py                             # smoke test: prints the observation shapes
```

`visual_test.py` shows random agents in the windowed game (macOS).

### Windows (WSL2)

FootsiesGym only ships macOS and Linux game binaries, so the project does not run natively on
Windows. Use WSL2 (Windows Subsystem for Linux), which gives you a real Linux environment:

```powershell
# in PowerShell (as administrator), once; then reboot
wsl --install -d Ubuntu
```

Then open the Ubuntu terminal and work **inside Linux**, not in `/mnt/c/...` (the Windows
filesystem is slow and breaks file permissions):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # then reopen the terminal
git clone <repo-url> ~/FootsiesRL
cd ~/FootsiesRL
uv sync
uv run python test.py
```

## Usage

Run everything from the repository root.

**Train**

```bash
uv run python src/train.py --run-name smoke --total-steps 200000 --num-envs 8    # a few minutes
uv run python src/train.py --run-name r0_s1 --reward r0 --seed 1 --total-steps 20000000
```

Output in `runs/<run-name>/`: `config.json`, `log.csv`, `ckpt_<steps>.pt`, `final.pt`.

| Option | Meaning |
|---|---|
| `--reward r0\|r1\|r2\|r3` | terminal ±10 / dense guard-break reward / potential shaping on the guard / win scaled by game length |
| `--ent-decay` | entropy 0.1 → 0 ("PPO Sched."), default fixed 0.025 |
| `--adv-std-floor X` | normalize advantages by max(std, X) (stall fix) |
| `--gae-lambda`, `--gamma`, `--lr`, `--ent-coef` | PPO hyperparameters (0.95, 0.995, 1e-4, 0.025) |
| `--special-charge`, `--action-delay N` | game options (charge actions, delay in frames, default 8) |
| `--num-envs`, `--port`, `--resume`, `--wandb` | parallel games (≥ 2), server port (one per run), continue a stopped run, W&B logging |

Full list: `uv run python src/train.py --help`.

**Evaluate and analyse**

```bash
uv run python src/evaluate.py --agent runs/r0_s1/final.pt --games 100            # vs random, noop, back
uv run python src/evaluate.py --agent runs/r0_s1/final.pt --opponents footsies_bot
uv run python src/head_to_head.py --agents runs/r0_s1/final.pt runs/r1_s1/final.pt --label test
uv run python src/best_response.py --target runs/r0_s1/final.pt --total-steps 20000000
uv run python src/diagnostics.py runs/r0_s1/final.pt
```

Results are appended to `results/` (`eval_<run>.csv`, `h2h_*.csv`, `br_<target>.csv`,
`diag_<run>.csv`). Actions are always sampled from the policy, never argmax.

`runs/` and `runs_br/` (training logs and checkpoints) are not in git, see `RSYNC.md`.
`plot_diagnostics.py`, `make_tables.py` and the head-to-head figure read only `results/`;
the training and best-response curves also need the `log.csv` files of the runs.

**Cluster (SLURM)**: every experiment has a script in `slurm/` with its usage in the header,
e.g. `sbatch slurm/seeds_array.sbatch`. Written for Orfeo (partition `THIN`, CPU only): adapt
partition and account.

## Repository

```text
src/           PPO, training, rewards, evaluation, tables
slurm/         sbatch scripts of all experiments
results/       evaluation CSVs, tables, figures
docs/          main paper
test.py, visual_test.py   environment checks
```

## Authors

[Alessio Martucci](https://www.github.com/tushytypes)

[Giovanni Oro](https://www.github.com/GioOro)

[Riccardo Samaritan](https://www.github.com/RiccardoSamaritan)