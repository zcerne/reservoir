#!/bin/bash
# signal_modulation/design01 — Nd:YAG crystal modulator, PILOT amplitude sweep
# on lips F5-gpu: 3 signal amplitudes, one per array task. gpumeep backend.
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# WHY THIS LADDER + 6000 t.u. (user, 2026-09-01): the design is uncalibrated and
# the steady state (stimulated burn = pump refill) settles on an amplitude-
# dependent timescale. This pilot brackets the ladder in one decade steps and
# runs LONG (6000) so the settling time can be read directly off the
# snap_point trace at the exit; the production sweep then sets
# dft_t_start = t_settle and a much shorter window. Sensors are deliberately
# minimal (two 1Ddft lines + one 0Dsnap point): a 6000 t.u. run with line
# snapshots or the population monitor would be GBs per task.
#
# SUBMIT (the user submits; never srun/sbatch from the assistant):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0-9%2 scripts/slurm_sigmod01_lips.sh
#
#   (%2 = one task per F5-gpu GPU. First task pays the JAX compile; the shared
#    compile cache on /project makes the rest start fast.)
#
# READ OUT after: settling time from snap_point (|Ez|/|Ey| envelope
# flattening), gain from monitor_2/monitor_1 at 1.064 um, per amplitude.
#
#SBATCH --job-name=signalmod01
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/sigmod01_%A_%a.log

set -euo pipefail

# 10-point ladder, deliberately dense between 10 and 100 where the
# pilot put the saturation knee (N3 burn 11.5% -> 88.8% across that
# one jump); 160/250 anchor the plateau past it.
AMPS=(1 5 15 22 32 46 68 100 160 250)
IDX=${SLURM_ARRAY_TASK_ID:-0}
AMP=${AMPS[$IDX]}

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
DESIGN=data/signal_modulation/design01

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: opt env python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
# LCrelax FIRST on PYTHONPATH: SimpleSim's lc_block imports
# `src.class_meep_blocks` from it for ANY 'reservoir' object, isotropic
# or not — omitting this is what killed arrays 29721821/29721842 with
# ModuleNotFoundError before the geometry was even built (2026-09-01).
# The path must be the repo ROOT (which contains src/), not src/.
export PYTHONPATH=/home/cerneziga/LCrelax${PYTHONPATH:+:$PYTHONPATH}
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/project/cerneziga/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export PYTHONUNBUFFERED=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Per-task scratch tag: concurrent tasks otherwise clobber each other's
# monitor npz in the shared simulation_gpumeep/ between run() and read-back.
export SIMPLESIM_SCRATCH_TAG="sigmod${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

cd "$BASE_DIR"
echo "=== sigmod01 task $IDX amp=$AMP host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) $(date) ==="
$PY -u run.py "$DESIGN" --backend gpumeep \
    --signal-amp "$AMP" --suffix "a${AMP}"
echo "=== task $IDX done $(date) ==="
