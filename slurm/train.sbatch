#!/bin/bash
# One full PPO run on Orfeo. Extra arguments are passed to src/train.py:
#   sbatch slurm/train.sbatch --run-name r0_s1 --seed 1
#   sbatch slurm/train.sbatch --run-name lam0.5_s1 --seed 1 --gae-lambda 0.5
# Orfeo allows 2 h per job: 20M steps (~1.4 h on THIN) fit, 25M (~1.7 h) is tight.
# Pass the step budget with --total-steps (it goes to train.py).
#SBATCH --job-name=footsies
#SBATCH --partition=THIN            # GENOA is the second choice (about 25% slower)
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00            # Orfeo limit per job (MaxWall) is 2 h
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
