#!/usr/bin/env bash
set -e
export MAMBA_EXE=/home/cernez/.local/bin/micromamba MAMBA_ROOT_PREFIX=/home/cernez/micromamba
eval "$($MAMBA_EXE shell hook --shell bash)"; micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src JAX_PLATFORMS=cuda,cpu
cd /home/cernez/resevoir/ladder
echo "=== config 1 (calibration) res40 ==="
python ladder.py --config 1 --engine both >/dev/null 2>&1
python _cfg1_ratio.py 2>&1 | grep -E "central mean|ratio_of_max"
echo "=== config 4 (cavity) res40 ==="
python ladder.py --config 4 --engine both >/dev/null 2>&1
python _cmp_metric.py /home/cernez/resevoir/data/ladder/config_4_mirrors_air
echo DONE_RECAL
