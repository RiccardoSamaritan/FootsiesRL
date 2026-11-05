#!/bin/bash
# Approximate best responses (src/best_response.py): one PPO learner trained from scratch
# against the frozen final checkpoint of seed 1 of each comparison variant.
#   task 0  r0_s1        task 3  r2_s1
#   task 1  r0_sched_s1  task 4  r3_s1
#   task 2  r1_s1
# Each task trains 20M steps (~1.5 h) and then plays 500 games against its target;
# the result is appended to results/br_<target>.csv.
#   sbatch slurm/br_array.sbatch
# A second learner seed for every target:  SEED=2 sbatch slurm/br_array.sbatch
# If a task hits the time limit after training, evaluate its final.pt only:
#   uv run python src/best_response.py --target runs/r0_s1/final.pt --eval-only
#SBATCH --job-name=br
#SBATCH --partition=THIN
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

TARGETS=(r0_s1 r0_sched_s1 r1_s1 r2_s1 r3_s1)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEED:-1}

export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
PORT=$((46001 + 10 * SLURM_ARRAY_TASK_ID + SLURM_JOB_ID % 1000))

uv run python src/best_response.py --target "runs/$TARGET/final.pt" --seed "$SEED" --port "$PORT" \
  --total-steps 20000000 --eval-games 500
