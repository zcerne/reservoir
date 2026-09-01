#!/bin/bash
# signal_modulation/design01c_sweep64 — NONLINEARITY BY THE ESTABLISHED 64-PAIR
# AMPLITUDE-SWEEP METHOD (data_gen/generate_harmonics_data.py + n4_harmonics_distortion),
# the same protocol as the reservoir designs, so the numbers are comparable to them.
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# WHAT RUNS: 64 static CW runs. Sweep parameter t over one period in 64 steps;
# source_1 gets amplitude cos(3t_j), source_2 cos(5t_j). DFT the 64 outputs at
# the 1.064 um bin: linear -> power only in bins 3 and 5; nonlinear -> harmonics
# (6, 9, 10) and intermods (2, 8, 1, 7).
#
# BATCHED 8 PER TASK, NOT ONE: each run is only 3000 t.u. (~5 min), so a task per
# run would spend a large fraction of the array on process start and JAX cache
# lookup. 8 array tasks x 8 runs amortizes that. --skip_existing makes a
# resubmission cost seconds on finished indices, so a partial array is safe to
# resubmit as-is.
#
# SUBMIT (the user submits; never srun/sbatch from the assistant):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir
#   sbatch --array=0-7%2 scripts/slurm_sweep64_lips.sh
#
# THEN ASSEMBLE (quick, one rank, on the login node or any task):
#   /project/cerneziga/micromamba/envs/opt/bin/python data_gen/generate_harmonics_data.py \
#       --path data/signal_modulation/design01c_sweep64 \
#       --tones 3,5 --channels 0,1 --n_t 64 --amps 40,40 --assemble
#
# THEN READ OUT:
#   python plotting/plot_sweep64_harmonics.py data/signal_modulation/design01c_sweep64 \
#       <design>/figures/sweep64_harmonics.png
#
#SBATCH --job-name=sweep64
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/sweep64_%A_%a.log

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID:-0}
BATCH_SIZE=8

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
DESIGN=data/signal_modulation/design01c_sweep64

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: opt env python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
# LCrelax FIRST on PYTHONPATH, repo ROOT not src/ — SimpleSim's lc_block imports
# `src.class_meep_blocks` from it for ANY 'reservoir' object, isotropic or not.
export PYTHONPATH=/home/cerneziga/LCrelax${PYTHONPATH:+:$PYTHONPATH}
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/project/cerneziga/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export PYTHONUNBUFFERED=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Per-task scratch tag — concurrent tasks share the design's simulation_gpumeep/
# and would otherwise overwrite each other's monitor npz between run() and the
# read-back, silently recording another task's fields.
export SIMPLESIM_SCRATCH_TAG="sweep64_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

cd "$BASE_DIR"
echo "=== sweep64 task $IDX (items $((IDX*BATCH_SIZE))..$((IDX*BATCH_SIZE+BATCH_SIZE-1))) host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) $(date) ==="
$PY -u data_gen/generate_harmonics_data.py \
    --path "$DESIGN" \
    --tones 3,5 --channels 0,1 --n_t 64 --amps 40,40 \
    --components Ey \
    --batch "$IDX" --batch_size "$BATCH_SIZE" --skip_existing
echo "=== task $IDX done $(date) ==="
