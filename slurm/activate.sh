# source this file to activate the SSC environment on CCR
PROJ=${PROJ:-/projects/academic/shaojiet/rahul/ssc-milp}
if [ -f "$PROJ/miniforge/etc/profile.d/conda.sh" ]; then source "$PROJ/miniforge/etc/profile.d/conda.sh"; fi
eval "$(conda shell.bash hook)"
conda activate "$PROJ/env"
export DEEPSEEK_API_KEY_FILE=${DEEPSEEK_API_KEY_FILE:-$(ls "$PROJ"/*key* 2>/dev/null | head -1)}
export OMP_NUM_THREADS=1
cd "$PROJ/code"
