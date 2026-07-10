#!/usr/bin/env bash
# Decisive convergence test: config 4 at res=80 (layers ~4-8 px wide instead of
# ~2-4). If gpu<->MEEP amplitude ratio moves toward 1.0, the res-40 residual is
# numerical cavity-Q sensitivity (not a code bug).
set -e
export MAMBA_EXE=/home/cernez/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/cernez/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src JAX_PLATFORMS=cuda,cpu
export LADDER_RES=80
cd /home/cernez/resevoir/ladder
echo "=== config 4 @ res=80, both engines ==="
python ladder.py --config 4 --engine both
python _cmp_metric.py /home/cernez/resevoir/data/ladder/config_4_mirrors_air
echo "DONE_RES80"
