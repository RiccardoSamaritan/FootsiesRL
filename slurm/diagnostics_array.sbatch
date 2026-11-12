#!/bin/bash
# Behaviour diagnostics (src/diagnostics.py): self-play, noop, back, random for 9 checkpoints,
# 3 per task (about 15 minutes per checkpoint). Results: results/diag_<run>.csv
#   sbatch slurm/diagnostics_array.sbatch
#SBATCH --job-name=diag
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

GROUPS_=("r0_s1 r1_s1 r2_s1" "r3_s1 r0_sched_s1 r1_floor_s1" "r0_charge_s1 r0_lam0_s1 r0_floor_s1")
for RUN in ${GROUPS_[$SLURM_ARRAY_TASK_ID]}; do
  if [ ! -f "runs/$RUN/final.pt" ]; then echo "skip $RUN: no final.pt"; continue; fi
  uv run python src/diagnostics.py "runs/$RUN/final.pt"
done
