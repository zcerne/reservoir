#!/bin/bash
# MEEP-backend IPC worker for the smaug CPUs — frees the GPUs for the task sets.
# $1 = design dir, $2 = batch, $3 = batch_size, $4 = mpi ranks (default 8)
cd "$HOME/resevoir" || exit 1
PY=$HOME/micromamba/envs/pmp/bin/python
MPI=$HOME/micromamba/envs/pmp/bin/mpirun
export SIMPLESIM_PATH=$HOME/SimpleSim GPUMEEP_PATH=$HOME/GPUmeep/src
export RESERVOIR_SOLVER=meep JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export SIMPLESIM_SCRATCH_TAG="cpu_$(basename $1)_b$2"
echo "=== $(date '+%H:%M') $1 ipc batch $2 (${4:-8} ranks) ==="
$MPI -np ${4:-8} $PY -u data_gen/generate_ipc_data.py \
    --path "data/lasing_testing/$1" --n 400 --scale 10 \
    --out_sensor n2f_map --components Ex,Ey,Ez \
    --batch "$2" --batch_size "${3:-50}" --skip_existing
echo "=== batch $2 done $(date '+%H:%M') ==="
