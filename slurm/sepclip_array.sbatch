#!/bin/bash
# Separate gradient clipping on 5 seeds: R0 trained with --clip-separately (actor and critic
# gradients clipped to 0.5 each, instead of their joint norm). check_grad_norm.py showed that the
# joint clip shrinks the actor's step 20-80x whenever games are decided, but not during stalls.
#   tasks 0-4  r0_sepclip seeds 1-5
# Same 20M steps and hyperparameters as the comparison runs: compare with r0_s* and r0_floor_s*.
#   sbatch slurm/sepclip_array.sbatch
# Rerun some tasks only:  sbatch --array=3 slurm/sepclip_array.sbatch
# Continue tasks stopped by the time limit from their latest checkpoint:
#   RESUME=1 sbatch --array=1,4 slurm/sepclip_array.sbatch
#SBATCH --job-name=sepclip
#SBATCH --partition=THIN
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00             # Orfeo limit per job is 2 h; 20M steps take ~1.4 h on THIN
#SBATCH --array=0-4%5
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# name | extra train.py arguments (one line per run, index = array task id)
RUNS=(
  "r0_sepclip_s1|--reward r0 --seed 1 --clip-separately"
  "r0_sepclip_s2|--reward r0 --seed 2 --clip-separately"
  "r0_sepclip_s3|--reward r0 --seed 3 --clip-separately"
  "r0_sepclip_s4|--reward r0 --seed 4 --clip-separately"
  "r0_sepclip_s5|--reward r0 --seed 5 --clip-separately"
)
IFS='|' read -r NAME ARGS <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

# torch sees every core of the node: limit its threads (measured: 4 is best)
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# unique port per task: tasks of the array can share a node
PORT=$((45001 + 10 * SLURM_ARRAY_TASK_ID + SLURM_JOB_ID % 1000))

uv run python src/train.py --run-name "$NAME" --port "$PORT" --total-steps 20000000 $ARGS ${RESUME:+--resume}
