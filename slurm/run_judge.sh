#!/bin/bash
#SBATCH --job-name=ssc-judge
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
# Usage: sbatch slurm/run_judge.sh <benchmark> [<benchmark> ...]  -- judge baseline + report refresh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/activate.sh"
for BENCH in "$@"; do
  python scripts/run.py judge  --benchmark "$BENCH" --workers 4
  python scripts/run.py report --benchmark "$BENCH"
done
