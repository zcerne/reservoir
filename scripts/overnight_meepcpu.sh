#!/bin/bash
# smaug1 CPUs via MEEP (8 ranks) — GPU-free, runs alongside the CUDA queues.
cd "$HOME/resevoir" || exit 1
source "$HOME/resevoir/scripts/_overnight_lib.sh"
export SIMPLESIM_SCRATCH_TAG=ovcpu
export RESERVOIR_SOLVER=meep JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
MPI=$HOME/micromamba/envs/pmp/bin/mpirun
banner "03b ipc batch 7 (MEEP backend)"
$MPI -np 8 $PY -u data_gen/generate_ipc_data.py --path $D03 --n 400 --scale 10 \
    --out_sensor n2f_map --components Ex,Ey,Ez \
    --batch 7 --batch_size 50 --skip_existing
banner "meep-cpu queue DONE"
