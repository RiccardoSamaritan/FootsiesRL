#!/bin/bash
# Reward comparison with 5 seeds (1-5) per variant: R0, R1, R2, R3 and PPO Sched.
# Seed 1 of R0, R1, R2 is already done (runs/r0_s1, r1_s1, r2_s1), so this array
# covers the 22 missing runs (same 20M steps), in this order:
#   tasks  0-3   r0 seeds 2-5      tasks 12-16  r3 seeds 1-5
#   tasks  4-7   r1 seeds 2-5      tasks 17-21  r0_sched (entropy 0.1 -> 0) seeds 1-5
#   tasks  8-11  r2 seeds 2-5
# Orfeo limits on THIN: at most 20 submitted jobs (queued + running) and 5 running at
# once, so the 22 tasks go in two submissions (5 waves of ~1.5 h in total):
#   sbatch slurm/seeds_array.sbatch                       # tasks 0-19, 5 at a time
#   sbatch --array=20-21 slurm/seeds_array.sbatch         # when some of them are done
# Rerun some tasks only:  sbatch --array=3,7 slurm/seeds_array.sbatch
#SBATCH --job-name=seeds
#SBATCH --partition=THIN
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00             # Orfeo limit per job is 2 h; 20M steps take ~1.4 h on THIN
#SBATCH --array=0-19%5
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# name | extra train.py arguments (one line per run, index = array task id)
RUNS=(
  "r0_s2|--reward r0 --seed 2"
  "r0_s3|--reward r0 --seed 3"
  "r0_s4|--reward r0 --seed 4"
  "r0_s5|--reward r0 --seed 5"
  "r1_s2|--reward r1 --seed 2"
  "r1_s3|--reward r1 --seed 3"
  "r1_s4|--reward r1 --seed 4"
  "r1_s5|--reward r1 --seed 5"
  "r2_s2|--reward r2 --seed 2"
  "r2_s3|--reward r2 --seed 3"
  "r2_s4|--reward r2 --seed 4"
  "r2_s5|--reward r2 --seed 5"
  "r3_s1|--reward r3 --seed 1"
  "r3_s2|--reward r3 --seed 2"
  "r3_s3|--reward r3 --seed 3"
  "r3_s4|--reward r3 --seed 4"
  "r3_s5|--reward r3 --seed 5"
  "r0_sched_s1|--reward r0 --seed 1 --ent-decay"
  "r0_sched_s2|--reward r0 --seed 2 --ent-decay"
  "r0_sched_s3|--reward r0 --seed 3 --ent-decay"
  "r0_sched_s4|--reward r0 --seed 4 --ent-decay"
  "r0_sched_s5|--reward r0 --seed 5 --ent-decay"
)
IFS='|' read -r NAME ARGS <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

# torch sees every core of the node: limit its threads (measured: 4 is best)
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# unique port per task: tasks of the array can share a node
PORT=$((45001 + 10 * SLURM_ARRAY_TASK_ID + SLURM_JOB_ID % 1000))

uv run python src/train.py --run-name "$NAME" --port "$PORT" --total-steps 20000000 $ARGS
