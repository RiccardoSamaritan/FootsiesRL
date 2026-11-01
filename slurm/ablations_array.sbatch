#!/bin/bash
# Bias/variance study and game ablations, all on R0 (terminal reward), seed 1, 20M steps.
# The reference is r0_s1 (GAE lambda 0.95, gamma 0.995, action delay 8, no charge actions).
#   task 0  r0_lam1_s1      GAE lambda 1 (Monte Carlo returns)
#   task 1  r0_lam0_s1      GAE lambda 0 (one-step TD)
#   task 2  r0_gamma099_s1  gamma 0.99
#   task 3  r0_charge_s1    use_special_charge_action (9 actions, 88-dim observation)
#   task 4  r0_delay0_s1    action delay 0
#   task 5  r0_delay12_s1   action delay 12
# Orfeo allows 20 submitted jobs (queued + running) and 5 running: submit this array
# when the seeds array has freed enough slots (check: squeue -u $USER | wc -l, minus 1).
#   sbatch slurm/ablations_array.sbatch
# Rerun some tasks only:  sbatch --array=3 slurm/ablations_array.sbatch
#SBATCH --job-name=ablations
#SBATCH --partition=THIN
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00             # Orfeo limit per job is 2 h; 20M steps take ~1.4 h on THIN
#SBATCH --array=0-5%5
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# name | extra train.py arguments (one line per run, index = array task id)
RUNS=(
  "r0_lam1_s1|--reward r0 --seed 1 --gae-lambda 1.0"
  "r0_lam0_s1|--reward r0 --seed 1 --gae-lambda 0.0"
  "r0_gamma099_s1|--reward r0 --seed 1 --gamma 0.99"
  "r0_charge_s1|--reward r0 --seed 1 --special-charge"
  "r0_delay0_s1|--reward r0 --seed 1 --action-delay 0"
  "r0_delay12_s1|--reward r0 --seed 1 --action-delay 12"
)
IFS='|' read -r NAME ARGS <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

# torch sees every core of the node: limit its threads (measured: 4 is best)
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# unique port per task: tasks of the array can share a node
PORT=$((45001 + 10 * SLURM_ARRAY_TASK_ID + SLURM_JOB_ID % 1000))

uv run python src/train.py --run-name "$NAME" --port "$PORT" --total-steps 20000000 $ARGS
