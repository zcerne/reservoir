#!/bin/bash
# IPC dataset for the res_sigmoid_gate reservoir type on lips — GPU array on
# the F5-gpu partition (2x GH200 nodes), gpumeep backend.
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# WHY GPU AND NOT THE F5 CPU PARTITION (first version of this script,
# 2026-08-24): the 19i-b element uses PER-TRANSITION DIPOLE ORIENTATIONS
# (45-deg converter, z-amp) and SimpleSim's MEEP backend raises
# "per-transition orientation requires the gpumeep backend" — real MEEP's
# MultilevelAtom has no oriented transitions. The design is gpumeep-only.
#
# Dataset spec (2026-08-24): 1000 samples, 4 input strips, drive scale 30
# (u ~ U[-1,1] per strip; per-strip amplitude 30*u puts the mean total input
# power mid-transition of the measured 19i-b sigmoid — between the decision
# knee S_in 6.3e5 and the ON crossing 4.4e6). All three components recorded
# at monitor_2; parts store the RAW COMPLEX FIELD (readout applied at
# assemble).
#
# SUBMIT (the user submits; never srun/sbatch from the assistant):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0-999%2 scripts/slurm_ipc_lips.sh
#
#   (For ipc, work items = samples, so the array is simply 0..n-1 — no --count
#    step. NOTE the count/assemble helpers CANNOT run on the login node with
#    the opt env: opt is aarch64 (GH200) and the login node is x86_64 ->
#    "Exec format error". Use the x86 env for login-node helpers:
#    /project/cerneziga/mamba_x86/envs/pmp/bin/python.
#    %2 = one task per F5-gpu GPU. First tasks pay the JAX compile; the shared
#    compile cache on /project makes the rest start fast.)
#
# ASSEMBLE when the array is done (login node, 1 rank, seconds — x86 env,
# NOT opt, see note above):
#   /project/cerneziga/mamba_x86/envs/pmp/bin/python \
#       data_gen/generate_ipc_data.py \
#       --path data/reservoir_types/res_sigmoid_gate --n 1000 --scale 30 \
#       --n_sources 4 --out_sensor monitor_2 --components Ex,Ey,Ez --assemble
#   -> data/reservoir_types/res_sigmoid_gate/datasets/ipc.npz {inputs, outputs}
#
# Prereqs on lips:
#   * repo at /home/cerneziga/resevoir with this design pulled  (git pull runs
#     ONLY from workbox through the /home/ziga/Lips sshfs mount — never here)
#   * aarch64 env /project/cerneziga/micromamba/envs/opt (jax + CUDA12 — the
#     GH200 env; gpumeep runs there, verified 2026-07-29)
#   * design is isotropic: no LC relax cache needed
#
#SBATCH --job-name=ipc_sigmoid
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/ipc_sigmoid_%A_%a.log

set -euo pipefail

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
DESIGN=data/reservoir_types/res_sigmoid_gate

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
# Per-task scratch tag: concurrent tasks share the design's simulation_gpumeep/
# dir; with one tag they clobber each other's monitor npz between run() and
# read-back and silently swap results (see slurm_char_array.sh history).
export SIMPLESIM_SCRATCH_TAG="ipcsig${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

cd "$BASE_DIR"
echo "=== ipc res_sigmoid_gate task ${SLURM_ARRAY_TASK_ID:-?} host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$PY -u data_gen/generate_ipc_data.py \
    --path "$DESIGN" \
    --n 1000 --scale 30 --n_sources 4 \
    --out_sensor monitor_2 --components Ex,Ey,Ez \
    --skip_existing --index "${SLURM_ARRAY_TASK_ID}"
echo "=== task ${SLURM_ARRAY_TASK_ID:-?} done $(date) ==="
