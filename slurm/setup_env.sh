#!/bin/bash
# One-time environment setup on UB CCR (run on a login node; needs internet).
# Creates a conda env "ssc" with Python 3.11, Pyomo, GLPK, SCIP/PySCIPOpt, OpenAI client.
set -euo pipefail
PROJ=${PROJ:-/projects/academic/shaojiet/rahul/ssc-milp}
ENV_DIR=${ENV_DIR:-$PROJ/env}

if ! command -v conda >/dev/null 2>&1; then
  if [ ! -x "$PROJ/miniforge/bin/conda" ]; then
    echo "Installing Miniforge into $PROJ/miniforge"
    cd "$PROJ"
    curl -L -o Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3.sh -b -p "$PROJ/miniforge"
    rm -f Miniforge3.sh
  fi
  source "$PROJ/miniforge/etc/profile.d/conda.sh"
fi
eval "$(conda shell.bash hook)"
if [ ! -d "$ENV_DIR" ]; then
  conda create -y -p "$ENV_DIR" -c conda-forge python=3.11 pyomo glpk scip pyscipopt numpy scipy openai pytest
fi
conda activate "$ENV_DIR"
python -c "import pyomo, pyscipopt, openai; print('pyomo', pyomo.version.version, 'pyscipopt', pyscipopt.__version__)"
which glpsol
echo "Environment ready: conda activate $ENV_DIR"
