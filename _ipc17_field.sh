#!/bin/bash
cd /home/cernez/resevoir
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
exec /home/cernez/micromamba/envs/pmp/bin/python data_gen/generate_ipc_data.py \
  --path data/reservoir_clasifications/17_2D_thr_resonator --n 400 \
  --readout field \
  --out data/reservoir_clasifications/17_2D_thr_resonator/datasets/ipc_field.npz \
  --skip_existing >> _ipc17_field.log 2>&1
