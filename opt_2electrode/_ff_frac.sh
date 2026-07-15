#!/bin/bash
cd /home/cernez/resevoir
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export FF_COST=fraction
export TAG=ff_frac
export SIGMA_DEG=3.0
export OPT_MAXEVAL=160
exec /home/cernez/micromamba/envs/pmp/bin/python opt_2electrode/run_2electrode_farfield.py >> _ff_frac.log 2>&1
