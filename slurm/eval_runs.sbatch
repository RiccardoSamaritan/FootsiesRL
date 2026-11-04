#!/bin/bash
# Evaluate finished runs: checkpoints at ~5M, 10M, 15M and the final one against random,
# noop and back (100 games each), and the final one also against FootsiesBot.
# One CSV per run is appended to results/eval_<run>.csv. Give the runs as arguments:
#   sbatch slurm/eval_runs.sbatch r0_s2 r0_s3 r0_s4 r0_s5
# About 6 minutes per run on THIN, so 4-5 runs fit in one job.
#SBATCH --job-name=eval
#SBATCH --partition=THIN
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

GAMES=${GAMES:-100}
for RUN in "$@"; do
  DIR=runs/$RUN
  if [ ! -f "$DIR/final.pt" ]; then echo "skip $RUN: no final.pt"; continue; fi
  # checkpoint written first after 5M, 10M and 15M steps
  for M in 5 10 15; do
    CK=$(ls "$DIR"/ckpt_*.pt 2>/dev/null | awk -F'[_.]' -v m=$((M * 1000000)) '{s=$(NF-1); if (s >= m) print s, $0}' | sort -n | head -1 | cut -d' ' -f2)
    [ -n "$CK" ] && uv run python src/evaluate.py --agent "$CK" --opponents random noop back --games "$GAMES" --label "$RUN-${M}M"
  done
  uv run python src/evaluate.py --agent "$DIR/final.pt" --opponents random noop back --games "$GAMES" --label "$RUN-final"
  uv run python src/evaluate.py --agent "$DIR/final.pt" --opponents footsies_bot --games "$GAMES" --label "$RUN-final-bot"
done
