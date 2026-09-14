#!/bin/bash
#SBATCH --job-name=ssc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out
# Usage: sbatch slurm/run_benchmark.sh <benchmark> [n]
#   stages: generate -> extract -> score -> evaluate -> judge -> report (all resumable)
set -euo pipefail
BENCH=${1:-nl4opt}
N=${2:-}
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/activate.sh"
NARG=""; if [ -n "$N" ]; then NARG="--n $N"; fi
python scripts/run.py generate --benchmark "$BENCH" $NARG --problem-workers 4 --recipe-workers 4
python scripts/run.py extract  --benchmark "$BENCH" $NARG --workers 6
python scripts/run.py score    --benchmark "$BENCH" $NARG --workers 4 --pool-workers 2
python scripts/run.py evaluate --benchmark "$BENCH" $NARG --workers 4 --pool-workers 2
python scripts/run.py judge    --benchmark "$BENCH" $NARG --workers 4
python scripts/run.py report   --benchmark "$BENCH" $NARG
