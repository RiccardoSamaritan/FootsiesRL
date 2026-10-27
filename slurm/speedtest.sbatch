#!/bin/bash
# Speed test on Orfeo: one short PPO run to measure steps/s on an EPYC node.
#   sbatch slurm/speedtest.sbatch
#SBATCH --job-name=footsies-speed
#SBATCH --partition=EPYC            # TODO check the exact partition name on Orfeo
##SBATCH --account=<your_account>   # TODO if required
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # server and training compete for CPU: try 4/8/16
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# the port must be unique among jobs sharing a node
PORT=$((45001 + SLURM_JOB_ID % 1000))

uv run python src/train.py --total-steps 300000 --run-name "speed_${SLURM_JOB_ID}" --port "$PORT"
