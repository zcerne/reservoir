#!/bin/bash
# 4-SOURCE saturation sweep on the lips F5-gpu partition (GH200 / gpumeep).
# All 4 input strips driven at the same amplitude (single_source_sweep.py),
# settled single-pass gain read over the design's DFT window; the 4-source
# saturation amplitude = the per-strip level where gain drops to ~half its
# small-signal value. ONE ARRAY TASK PER LEVEL, so the 5 levels run across the
# free F5-gpu GPUs.
#
# Backend is gpumeep (the design's solver), which is MEEP-validated (2% RMS in
# 2D, exact in 3D). NOTE gpumeep ignores rate_32, so this is the emission+pump
# model (rate_32=0) only — correct for the amp sweep (saturation is pump-vs-
# stimulated). Do NOT point this at a rate_32 design expecting recharge; use the
# MEEP CPU script (slurm_sat4src_lips.sh) for that.
#
# $1 = design dir (default design04_4source).
#
# SUBMIT (the user submits; the assistant never runs sbatch):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir && git pull
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0-4%2 scripts/slurm_sat4src_gpu_lips.sh data/signal_modulation/design04_4source
#   squeue --me
#   # once all 5 finish, merge the per-level npz (no FDTD):
#   /project/cerneziga/micromamba/envs/opt/bin/python single_source_sweep.py \
#       --path data/signal_modulation/design04_4source --assemble
#   # -> <design>/datasets/single_source_sweep.npz {levels,out_norm,gain}, sorted.
#
# PARTITION RULE (user, 2026-08-24): only F5 / F5-gpu on lips.
#
#SBATCH --job-name=sat4src_gpu
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/sat4src_gpu_%A_%a.log

set -euo pipefail
LEVELS=(0.01 0.035 0.1 0.35 1)          # per-strip amplitudes, brackets the 0.035 estimate
IDX=${SLURM_ARRAY_TASK_ID:-0}
LV=${LEVELS[$IDX]}
[ -n "${LV:-}" ] || { echo "no level for array idx $IDX (ladder has ${#LEVELS[@]}: 0-$(( ${#LEVELS[@]} - 1 )))"; exit 1; }

D=${1:-data/signal_modulation/design04_4source}
BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
[ -d "$BASE_DIR/$D" ] || { echo "ERROR: $BASE_DIR/$D missing — git pull on lips first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: opt env python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export PYTHONPATH=/home/cerneziga/LCrelax${PYTHONPATH:+:$PYTHONPATH}
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/project/cerneziga/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export PYTHONUNBUFFERED=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export SIMPLESIM_SCRATCH_TAG="sat4gpu_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

cd "$BASE_DIR"
mkdir -p "$D/datasets"
echo "=== sat4src_gpu $D level $LV (array idx $IDX) on $(hostname) $(date) ==="
$PY -u single_source_sweep.py --path "$D" --levels "$LV" \
    --out "$D/datasets/sat4src_${IDX}.npz"
echo "=== idx $IDX (level $LV) done $(date) ==="
