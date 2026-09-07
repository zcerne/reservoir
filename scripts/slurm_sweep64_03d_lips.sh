#!/bin/bash
# design03d_sweep64 — 64-pair CROSS sweep, lips F5-gpu array. One sample per
# array task (--batch $IDX --batch_size 1), so the GH200 does each in ~40 min
# (amp-50 matched-drive rerun of 03c). All 64 compute fresh — no parts
# carried over, so nothing is skipped.
#
# PARTITION RULE (user, 2026-08-24): only F5 / F5-gpu on lips.
#
# WHY ARRAY not one-process-loop: the smaug run OOM'd a 4090 when 8 samples
# accumulated in one python process (VRAM fragmentation; single run peaks ~9
# GB, 8 accumulated blew 24 GB). One task = one process = clean device, and
# the array also parallelises across every free F5-gpu GPU.
#
# SUBMIT (the user submits; the assistant never runs sbatch):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir && git pull
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0-63%4 scripts/slurm_sweep64_03d_lips.sh
#   # all 64 run (~40 min each on the GH200)
#   # then, once the array is done:
#   BASE_DIR=/home/cerneziga/resevoir /project/cerneziga/micromamba/envs/opt/bin/python \
#       data_gen/generate_harmonics_data.py \
#       --path data/signal_modulation/design03d_sweep64 \
#       --tones 3,5 --channels 0,1 --n_t 64 --amps 50,50 --assemble
#
#SBATCH --job-name=sweep64_03d
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/sweep64_03d_%A_%a.log

set -euo pipefail
IDX=${SLURM_ARRAY_TASK_ID:-0}
BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
D=data/signal_modulation/design03d_sweep64

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
export SIMPLESIM_SCRATCH_TAG="s64d03_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

cd "$BASE_DIR"
echo "=== sweep64_03d task $IDX host $(hostname) $(date) ==="
$PY -u data_gen/generate_harmonics_data.py --path "$D" \
    --tones 3,5 --channels 0,1 --n_t 64 --amps 50,50 \
    --skip_existing --batch "$IDX" --batch_size 1
echo "=== task $IDX done $(date) ==="
