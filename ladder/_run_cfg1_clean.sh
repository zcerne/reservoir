#!/usr/bin/env bash
set -e
export MAMBA_EXE=/home/cernez/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/cernez/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src JAX_PLATFORMS=cuda,cpu
find /home/cernez/resevoir /home/cernez/GPUmeep -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cd /home/cernez/resevoir/ladder
# sanity: confirm the edited source term is present in the file that will run
grep -c "envelope-derivative" /home/cernez/resevoir/class_simulation_gpu.py
python ladder.py --config 1 --engine gpumeep >/dev/null 2>&1
python _cfg1_ratio.py 2>&1 | grep -E "central mean|ratio_of_max"
echo DONE_CLEAN
