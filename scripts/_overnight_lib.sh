#!/bin/bash
# Shared helpers for the 2026-07-28 overnight nonlinearity/capacity campaign.
# /home/cernez on smaug == /home/ziga/Orion on workbox (same NFS), so one copy
# of these scripts serves every host.

PY=${PY:-$HOME/micromamba/envs/pmp/bin/python}
export SIMPLESIM_PATH=${SIMPLESIM_PATH:-$HOME/SimpleSim}
export GPUMEEP_PATH=${GPUMEEP_PATH:-$HOME/GPUmeep/src}
# Never preallocate: the first JAX job on a card otherwise grabs ~75% of VRAM
# and every later job dies with a cuSolver/OOM error.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

D04=data/lasing_testing/04_adding_LC
D03=data/lasing_testing/03b_isotropic_ds
AMPFILE=$HOME/resevoir/scripts/chosen_amps.txt

# wait_slot N — block until fewer than N *FDTD* jobs are using the GPU. Counts
# CUDA contexts holding >= 800 MiB: one 2D forward needs ~1.2 GB, while display
# servers and other strays sit at a few hundred MiB and would otherwise eat a
# slot permanently (they did: a 386 MiB process wedged smaug1's queue at 3/3).
wait_slot() {
  local limit=${1:-3}
  while :; do
    local n
    n=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
        2>/dev/null | awk '$1 >= 800' | wc -l)
    [ "$n" -lt "$limit" ] && return 0
    sleep 60
  done
}

# wait_file PATH — block until PATH exists.
wait_file() { while [ ! -e "$1" ]; do sleep 60; done; }

# read_amps — echo the two chosen drive amplitudes.
read_amps() { cat "$AMPFILE"; }

banner() { echo "=== [$(date '+%H:%M:%S')] $* ==="; }
