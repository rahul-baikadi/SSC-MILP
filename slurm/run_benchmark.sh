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
# Usage: sbatch [--mem=32G --time=24:00:00] slurm/run_benchmark.sh <benchmark> [n]
#   env RUN_NAME=<dir>      write to runs/<dir> instead of runs/<benchmark>
#   env GEN_ARGS="..."      extra generation flags (e.g. "--no-templates --repair-rounds 0")
#   env SKIP_JUDGE=1        skip the LLM-judge baseline
#   stages: generate -> extract -> score -> evaluate -> judge -> report (all resumable)
set -euo pipefail
BENCH=${1:-nl4opt}
N=${2:-}
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/activate.sh"
NARG=""; if [ -n "$N" ]; then NARG="--n $N"; fi
RARG=""; if [ -n "${RUN_NAME:-}" ]; then RARG="--run-name $RUN_NAME"; fi
python scripts/run.py generate --benchmark "$BENCH" $NARG $RARG ${GEN_ARGS:-} --problem-workers 4 --recipe-workers 4
python scripts/run.py extract  --benchmark "$BENCH" $NARG $RARG --workers 6
python scripts/run.py score    --benchmark "$BENCH" $NARG $RARG --workers 4 --pool-workers 2
python scripts/run.py evaluate --benchmark "$BENCH" $NARG $RARG --workers 4 --pool-workers 2
if [ -z "${SKIP_JUDGE:-}" ]; then
  python scripts/run.py judge  --benchmark "$BENCH" $NARG $RARG --workers 4 || echo "judge stage failed (non-fatal)"
fi
python scripts/run.py report   --benchmark "$BENCH" $NARG $RARG
