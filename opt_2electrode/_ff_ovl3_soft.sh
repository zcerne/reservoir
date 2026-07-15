#!/bin/bash
cd /home/cernez/resevoir
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export FF_COST=overlap
export FF_ANNEAL=0
export SIGMA_DEG=3.0
export FF_ANCHOR=soft
export FF_W=11.1
export VMAX=7.0
export TAG=ff_ovl3_soft
export OPT_MAXEVAL=160
exec /home/cernez/micromamba/envs/pmp/bin/python opt_2electrode/run_2electrode_farfield.py >> _ff_ovl3_soft.log 2>&1
