#!/bin/bash
# IPC dataset on WAVELENGTH channels — the 4-tone comb reservoir.
# Channels are the four comb tones (k = -2,-1,+1,+2, spacing 30 GHz) of
# data/frequency_modulation/01_4tone_comb instead of spatial strips: per probe,
# tone i is driven at amplitude scale*u_i (u ~ U[-1,1]; negative = pi phase).
# Readout = monitor_2's 61 comb-aligned bins x exit line: tones, their gain
# tilt, AND the intermods the inversion beats write between the bins.
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# SCALE 20/tone: the measured comb sweet spot (job 29782513) — IMD peaks ~4%
# at 10-20/tone with clean populations; the beat-driven runaway starts ~40
# (L35 already rings at t~21k, L50 diverges). Random |u|<=1 keeps every probe
# at <=20/tone, worst coherent peak 80 << the ~140-200 static ceiling.
#
# COST: run 24000 (window 4000-24000 = 2 beat periods of the 1/Delta_f =
# 10000 t.u. beat — the standing >=2-beat rule for comb-aligned bins),
# ~12.6 min/sample on a GH200 -> 1000 samples at %2 ~ 4.4 days. Use
# --array=0-499%2 first if a 500-probe pilot should gate the full set.
#
# SUBMIT (the user submits; never sbatch from the assistant):
#   cd /home/cerneziga/resevoir
#   sbatch --array=0-999%2 scripts/slurm_ipc_comb_gpu.sh
#
# ASSEMBLE when done (login node, x86 pmp env):
#   /project/cerneziga/mamba_x86/envs/pmp/bin/python \
#       data_gen/generate_ipc_data.py \
#       --path data/frequency_modulation/01_4tone_comb --n 1000 --scale 20 \
#       --out_sensor monitor_2 --components Ex,Ey,Ez \
#       --out data/frequency_modulation/01_4tone_comb/datasets/ipc_wl.npz \
#       --assemble
#
#SBATCH --job-name=ipc_comb_wl
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/ipc_comb_wl_%A_%a.log

set -euo pipefail

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
DESIGN=data/frequency_modulation/01_4tone_comb
OUT=$DESIGN/datasets/ipc_wl.npz

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: opt env python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export RESERVOIR_SOLVER=gpumeep
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/project/cerneziga/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Per-task scratch tag: concurrent tasks share simulation_gpumeep/; without a
# unique tag they clobber each other's monitor npz and silently swap results.
export SIMPLESIM_SCRATCH_TAG="ipcwl_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

cd "$BASE_DIR"
echo "=== ipc comb wavelength-channels task ${SLURM_ARRAY_TASK_ID:-?} host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$PY -u data_gen/generate_ipc_data.py \
    --path "$DESIGN" \
    --n 1000 --scale 20 \
    --out_sensor monitor_2 --components Ex,Ey,Ez \
    --out "$OUT" \
    --skip_existing --index "${SLURM_ARRAY_TASK_ID}"
echo "=== task ${SLURM_ARRAY_TASK_ID:-?} done $(date) ==="
