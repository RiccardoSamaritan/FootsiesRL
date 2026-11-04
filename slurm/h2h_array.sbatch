#!/bin/bash
# Head-to-head matches between checkpoints, split in 8 shards (one job each).
# Pick the pool with POOL (default: final):
#   final  the final checkpoint of every comparison run: R0, R0 Sched., R1, R2, R3 (seeds 1-5)
#          and r1_floor_s1 = 26 agents, 325 pairs
#   time   seed 1 of each variant at ~5M, 10M, 15M and final = 24 agents, 276 pairs
#          (how every variant progresses, and who beats whom at the same training stage)
#   POOL=final sbatch slurm/h2h_array.sbatch
#   POOL=time  sbatch slurm/h2h_array.sbatch
# Each shard writes results/h2h_<pool>.s<k>of8.csv and skips pairs already played, so a
# shard that hits the time limit can simply be submitted again (sbatch --array=3 ...).
# When all 8 are done, draw the figure and the ranking (needs no game server, e.g. on the Mac):
#   python3 -I src/head_to_head.py --label final --plot-only
#SBATCH --job-name=h2h
#SBATCH --partition=THIN
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-7%5
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs
export OMP_NUM_THREADS=${THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

POOL=${POOL:-final}
GAMES=${GAMES:-100}
case "$POOL" in
  final)
    AGENTS=$(ls runs/r{0,1,2,3}_s{1,2,3,4,5}/final.pt runs/r0_sched_s{1,2,3,4,5}/final.pt runs/r1_floor_s1/final.pt) ;;
  time)
    AGENTS=""
    for RUN in r0_s1 r1_s1 r2_s1 r3_s1 r0_sched_s1 r1_floor_s1; do
      for M in 5 10 15; do
        CK=$(ls runs/$RUN/ckpt_*.pt | awk -F'[_.]' -v m=$((M * 1000000)) '{s=$(NF-1); if (s >= m) print s, $0}' | sort -n | head -1 | cut -d' ' -f2)
        AGENTS="$AGENTS $CK"
      done
      AGENTS="$AGENTS runs/$RUN/final.pt"
    done ;;
  *) echo "unknown POOL $POOL (final or time)"; exit 1 ;;
esac

uv run python src/head_to_head.py --agents $AGENTS --label "$POOL" --games "$GAMES" \
  --shard "$SLURM_ARRAY_TASK_ID/8"
