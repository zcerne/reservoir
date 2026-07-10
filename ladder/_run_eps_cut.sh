#!/usr/bin/env bash
set -e
export MAMBA_EXE=/home/cernez/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/cernez/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export JAX_PLATFORMS=cuda,cpu
cd /home/cernez/resevoir/ladder
python _eps_cut.py
echo "DONE_EPS_CUT"
