#!/bin/bash
cd /home/cernez/resevoir
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export OPT_MAXEVAL=150
exec /home/cernez/micromamba/envs/pmp/bin/python run_2electrode_fingers.py >> _fingers8.log 2>&1
