#!/bin/bash
#SBATCH --job-name=ssc-rescore
#SBATCH --partition=general-compute --qos=general-compute
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=08:00:00 --mem=32G
#SBATCH --output=runs/rescore-%j.out
cd "$SLURM_SUBMIT_DIR" || exit 1
source slurm/activate.sh
for b in nl4opt mamo_easy mamo_complex complexor industryor optibench optmath; do
  echo "=== $b ==="
  python scripts/run.py score    --benchmark "$b" --workers 6 --pool-workers 6
  python scripts/run.py report   --benchmark "$b"
done
echo ALL_DONE
