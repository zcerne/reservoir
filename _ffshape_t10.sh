#!/bin/bash
cd /home/cernez/resevoir
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export THETA0_DEG=10.0
export SIGMA_DEG=3.0
export FF_W=11.1
export FF_OPT_V0=1
export TAG=ffshape_t10
export OPT_MAXEVAL=160
exec /home/cernez/micromamba/envs/pmp/bin/python run_2electrode_ffshape.py >> _ffshape_t10.log 2>&1
