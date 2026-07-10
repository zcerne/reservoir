#!/usr/bin/env bash
# Isolate ENGINE vs SMOOTHING: run config 4 with subpixel averaging OFF in BOTH
# engines (MEEP_NO_SUBPIXEL + GPUMEEP_NOAVG). Both then point-sample the same
# n^2(x); any residual is a pure engine/grid-registration difference.
set -e
export MAMBA_EXE=/home/cernez/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/cernez/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export JAX_PLATFORMS=cuda,cpu
export MEEP_NO_SUBPIXEL=1
export GPUMEEP_NOAVG=1
cd /home/cernez/resevoir/ladder
echo "=== config 4, subpixel OFF in BOTH engines ==="
python ladder.py --config 4 --engine both
echo "=== metric (both point-sampled) ==="
python _cmp_metric.py /home/cernez/resevoir/data/ladder/config_4_mirrors_air
echo "DONE_NOAVG"
