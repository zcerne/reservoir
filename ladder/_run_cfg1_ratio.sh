#!/usr/bin/env bash
set -e
export MAMBA_EXE=/home/cernez/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/cernez/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src JAX_PLATFORMS=cuda,cpu
cd /home/cernez/resevoir/ladder
python ladder.py --config 1 --engine both >/dev/null 2>&1
python _cfg1_ratio.py
