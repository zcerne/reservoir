#!/usr/bin/env bash
# Config 4, subpixel averaging ON in both (MEEP default + gpumeep exact Kottke),
# now with the grid-registration fix (rounded-centered Yee grid). Regenerates the
# averaged MEEP reference (clobbered by the NOAVG probe) and the gpumeep monitor.
set -e
export MAMBA_EXE=/home/cernez/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/cernez/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export JAX_PLATFORMS=cuda,cpu
cd /home/cernez/resevoir/ladder
echo "=== config 4, averaging ON in both, grid-registration fix ==="
python ladder.py --config 4 --engine both
echo "=== metric ==="
python _cmp_metric.py /home/cernez/resevoir/data/ladder/config_4_mirrors_air
echo "=== plot ==="
python plot_config.py --config 4
cp -f /tmp/ladder_config_4_mirrors_air.png /home/cernez/resevoir/ladder/ladder_config_4.png
echo "DONE_CFG4_BOTH"
