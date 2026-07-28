#!/bin/bash
# workbox (A1000, ~5x slower): one capacity batch.
cd /home/ziga/Orion/resevoir || exit 1
PY=/home/ziga/micromamba/envs/pmp/bin/python
export SIMPLESIM_PATH=/home/ziga/Orion/SimpleSim
export GPUMEEP_PATH=/home/ziga/Orion/GPUmeep/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_PLATFORMS=cuda,cpu SIMPLESIM_SCRATCH_TAG=ovwb
D03=data/lasing_testing/03b_isotropic_ds
echo "=== [$(date '+%H:%M:%S')] 03b ipc batch 6 ==="
$PY -u data_gen/generate_ipc_data.py --path $D03 --n 400 --scale 10 \
    --out_sensor n2f_map --components Ex,Ey,Ez \
    --batch 6 --batch_size 50 --skip_existing
echo "=== workbox queue DONE ==="
