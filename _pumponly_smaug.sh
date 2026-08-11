#!/usr/bin/env bash
# Pump-only run on a smaug 4090 (gpumeep). Detaches and returns immediately.
#   bash _pumponly_smaug.sh data/reservoir_types/block_iso_gain/02 pumponly
DESIGN=${1:?design rel path}
SUFFIX=${2:-pumponly}
cd "$HOME/resevoir" || exit 1
export RESERVOIR_SOLVER=gpumeep
export GPUMEEP_PATH=$HOME/GPUmeep/src
export SIMPLESIM_PATH=$HOME/SimpleSim
# LCrelax first on PYTHONPATH, it ships helpers that others duplicate
export PYTHONPATH=$HOME/LCrelax
export JAX_PLATFORMS=cuda,cpu
# never preallocate: a second JAX job on the same 4090 dies on cuSolver/OOM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SIMPLESIM_SCRATCH_TAG="pumponly_$(basename "$DESIGN")"

LOG="pumponly_$(basename "$DESIGN")_${SUFFIX}.log"
nohup setsid "$HOME"/micromamba/envs/pmp/bin/python -u _run_pumponly.py \
      "$DESIGN" --suffix "$SUFFIX" ${SNAP:+--snap-interval "$SNAP"} ${PUMPAMP:+--pump-amp "$PUMPAMP"} > "$LOG" 2>&1 < /dev/null &
sleep 2
echo "launched -> $HOME/resevoir/$LOG"
