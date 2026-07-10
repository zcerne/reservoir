#!/usr/bin/env bash
# Rerun ONLY gpumeep for config 4 (mirrors+air) with the new exact-MEEP Kottke
# subpixel averaging, then compare against the existing MEEP reference monitor.
set -e
export MAMBA_EXE=/home/cernez/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/cernez/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export JAX_PLATFORMS=cuda,cpu
cd /home/cernez/resevoir/ladder
echo "=== gpumeep config 4 (exact Kottke) ==="
python ladder.py --config 4 --engine gpumeep
echo "=== metric vs MEEP reference ==="
python _cmp_metric.py /home/cernez/resevoir/data/ladder/config_4_mirrors_air
echo "DONE_CFG4"
