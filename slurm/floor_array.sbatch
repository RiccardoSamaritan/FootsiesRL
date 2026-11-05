#!/bin/bash
# Stall fix on 5 seeds: R0 and R1 trained with --adv-std-floor 0.1 (the advantage std used
# for normalization is at least 0.1, so near-zero advantages are not blown up into noise).
# r1_floor_s1 already exists (same settings), so this array covers the 9 missing runs:
#   tasks 0-4  r0_floor seeds 1-5
#   tasks 5-8  r1_floor seeds 2-5
# Same 20M steps and hyperparameters as the comparison runs: compare with r0_s* and r1_s*.
#   sbatch slurm/floor_array.sbatch
# Rerun some tasks only:  sbatch --array=3,7 slurm/floor_array.sbatch
#SBATCH --job-name=floor
#SBATCH --partition=THIN
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00             # Orfeo limit per job is 2 h; 20M steps take ~1.4 h on THIN
#SBATCH --array=0-8%5
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# name | extra train.py arguments (one line per run, index = array task id)
RUNS=(
  "r0_floor_s1|--reward r0 --seed 1 --adv-std-floor 0.1"
  "r0_floor_s2|--reward r0 --seed 2 --adv-std-floor 0.1"
  "r0_floor_s3|--reward r0 --seed 3 --adv-std-floor 0.1"
  "r0_floor_s4|--reward r0 --seed 4 --adv-std-floor 0.1"
  "r0_floor_s5|--reward r0 --seed 5 --adv-std-floor 0.1"
  "r1_floor_s2|--reward r1 --seed 2 --adv-std-floor 0.1"
  "r1_floor_s3|--reward r1 --seed 3 --adv-std-floor 0.1"
  "r1_floor_s4|--reward r1 --seed 4 --adv-std-floor 0.1"
  "r1_floor_s5|--reward r1 --seed 5 --adv-std-floor 0.1"
)
IFS='|' read -r NAME ARGS <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

# torch sees every core of the node: limit its threads (measured: 4 is best)
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# unique port per task: tasks of the array can share a node
PORT=$((45001 + 10 * SLURM_ARRAY_TASK_ID + SLURM_JOB_ID % 1000))

uv run python src/train.py --run-name "$NAME" --port "$PORT" --total-steps 20000000 $ARGS
