#!/bin/bash
cd /home/cernez/resevoir
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export FF_COST=overlap
export TAG=ff_ovl
export OPT_MAXEVAL=160
exec /home/cernez/micromamba/envs/pmp/bin/python opt_2electrode/run_2electrode_farfield.py >> _ff_ovl.log 2>&1
