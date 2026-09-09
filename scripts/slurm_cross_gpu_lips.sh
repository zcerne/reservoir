#!/bin/bash
# CROSS-SATURATION test on the lips F5-gpu partition (GH200 / gpumeep).
# One strip is swept over a ladder while the other strips sit FIXED at the
# saturation knee; if the fixed channels' output lobes respond to the swept
# strip, the channels share gain -> genuine input-pattern-dependent mixing
# (the thing the equal-drive amp sweep cannot see). One array task per swept
# level; cross_sweep.py sets the asymmetric drive vector, the design's
# monitors (monitor_2 / field_map / pop) save the per-run profiles.
#
# $1 = design dir           (default data/signal_modulation/05b_ampsweep)
# $2 = swept-level ladder   (default 0,10,30,50,70,90,120 — the 0 rung is the
#                            baseline: fixed strips alone)
# $3 = fixed amplitude      (default 70 = 05b's amp_sat)
# $4 = swept strip, 1-based (default 2 — inner strip, two neighbours)
#
# SUBMIT (the user submits; the assistant never runs sbatch):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir && git pull
#   sbatch --array=0-6%2 scripts/slurm_cross_gpu_lips.sh
#   # array size must match the ladder length (7 defaults -> 0-6).
#
# PARTITION RULE (user, 2026-08-24): only F5 / F5-gpu on lips.
#
#SBATCH --job-name=cross_gpu
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/cross_gpu_%A_%a.log

set -euo pipefail
IFS=',' read -r -a LEVELS <<< "${2:-0,10,30,50,70,90,120}"
IDX=${SLURM_ARRAY_TASK_ID:-0}
LV=${LEVELS[$IDX]}
[ -n "${LV:-}" ] || { echo "no level for array idx $IDX (ladder has ${#LEVELS[@]}: 0-$(( ${#LEVELS[@]} - 1 )))"; exit 1; }

D=${1:-data/signal_modulation/05b_ampsweep}
FIXED=${3:-70}
STRIP=${4:-2}
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
export SIMPLESIM_SCRATCH_TAG="cross_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

cd "$BASE_DIR"
mkdir -p "$D/datasets"
echo "=== cross $D strip $STRIP level $LV (fixed $FIXED, idx $IDX) on $(hostname) $(date) ==="
$PY -u cross_sweep.py --path "$D" --strip "$STRIP" --level "$LV" --fixed "$FIXED"
echo "=== idx $IDX (level $LV) done $(date) ==="
