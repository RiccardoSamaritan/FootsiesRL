#!/bin/bash
# One full PPO run on Orfeo. Extra arguments are passed to src/train.py:
#   sbatch slurm/train.sbatch --run-name r0_s1 --seed 1
#   sbatch slurm/train.sbatch --run-name lam0.5_s1 --seed 1 --gae-lambda 0.5
# Default: 25M steps (~1.7 h on THIN, ~2.2 h on GENOA).
#SBATCH --job-name=footsies
#SBATCH --partition=THIN            # GENOA is the second choice (about 25% slower)
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# torch sees every core of the node: limit its threads (measured: 4 is best)
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# unique port per job: several jobs can share a node
PORT=$((45001 + SLURM_JOB_ID % 1000))

uv run python src/train.py --port "$PORT" "$@"
