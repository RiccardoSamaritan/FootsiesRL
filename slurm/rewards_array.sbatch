#!/bin/bash
# Reward comparison: one PPO run per array task, same budget and seed.
# R0 is already done (runs/r0_s1, same 20M steps, seed 1), so the array covers R1-R3.
#   sbatch slurm/rewards_array.sbatch
# To rerun a single task: sbatch --array=1 slurm/rewards_array.sbatch
#SBATCH --job-name=rewards
#SBATCH --partition=THIN
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00             # Orfeo limit per job is 2 h; 20M steps take ~1.4 h on THIN
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# name | extra train.py arguments (one line per run)
RUNS=(
  "r1_s1|--reward r1 --seed 1"
  "r2_s1|--reward r2 --seed 1"
  "r3_s1|--reward r3 --seed 1"
)
IFS='|' read -r NAME ARGS <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

# torch sees every core of the node: limit its threads (measured: 4 is best)
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# unique port per task: tasks of the array can share a node
PORT=$((45001 + 10 * SLURM_ARRAY_TASK_ID + SLURM_JOB_ID % 1000))

uv run python src/train.py --run-name "$NAME" --port "$PORT" --total-steps 20000000 $ARGS
