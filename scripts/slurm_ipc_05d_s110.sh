#!/bin/bash
# IPC dataset #2 for 05d — drive scale 110 (the "stronger medium" run).
# Same 1000 probes as the scale-70 dataset (same default seed -> same U),
# only the physical drive changes: capacity difference is then purely the
# medium's nonlinearity growing, not sampling noise. F5-gpu array, gpumeep.
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# WHY 110: dataset #1 (scale 70) showed deg-3 capacity near-saturated but
# deg-4/5 weak; cross-saturation grows 0.7% -> 11.4% over drives 30 -> 120.
# 110 pushes toward the knee while staying under the instability ceiling.
#
# VALIDATE FIRST (worst case = all four strips at +110, coherent): run one
# rung of the existing sat sweep and check populations stay positive:
#   sbatch --array=0-0 scripts/slurm_sat4src_gpu_lips.sh \
#       data/signal_modulation/05d_ampsweep 110
# (05d was swept clean only to 90; 05b was clean to 120 — expected fine,
#  but confirm N3 >= 0 before burning 1000 runs.)
#
# SUBMIT (the user submits; never srun/sbatch from the assistant):
#   cd /home/cerneziga/resevoir
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0-999%2 scripts/slurm_ipc_05d_s110.sh
#
# ASSEMBLE when done (login node, x86 env — opt is aarch64 and gives
# "Exec format error" on the login node):
#   /project/cerneziga/mamba_x86/envs/pmp/bin/python \
#       data_gen/generate_ipc_data.py \
#       --path data/signal_modulation/05d_ampsweep --n 1000 --scale 110 \
#       --n_sources 4 --out_sensor monitor_2 --components Ex,Ey,Ez \
#       --out data/signal_modulation/05d_ampsweep/datasets/ipc_s110.npz \
#       --assemble
#
#SBATCH --job-name=ipc05d110
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/ipc05d110_%A_%a.log

set -euo pipefail

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
DESIGN=data/signal_modulation/05d_ampsweep
OUT=$DESIGN/datasets/ipc_s110.npz

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
export SIMPLESIM_SCRATCH_TAG="ipc110_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

cd "$BASE_DIR"
echo "=== ipc 05d scale110 task ${SLURM_ARRAY_TASK_ID:-?} host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$PY -u data_gen/generate_ipc_data.py \
    --path "$DESIGN" \
    --n 1000 --scale 110 --n_sources 4 \
    --out_sensor monitor_2 --components Ex,Ey,Ez \
    --out "$OUT" \
    --skip_existing --index "${SLURM_ARRAY_TASK_ID}"
echo "=== task ${SLURM_ARRAY_TASK_ID:-?} done $(date) ==="
