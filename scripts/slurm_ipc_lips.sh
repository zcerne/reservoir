#!/bin/bash
# IPC dataset for the res_sigmoid_gate reservoir type on the lips F5 CPU
# partition (x86_64 zen-4 nodes f02..f07) — Slurm ARRAY, one forward MEEP run
# per task, one FULL node per run (the sim is compute-bound; same rationale as
# scripts/slurm_char_array.sh on Orion).
#
# Dataset spec (2026-08-24): 1000 samples, 4 input strips, drive scale 30
# (u ~ U[-1,1] per strip; per-strip amplitude 30*u puts the mean total input
# power mid-transition of the measured 19i-b sigmoid — between the decision
# knee S_in 6.3e5 and the ON crossing 4.4e6 — max excursions reach saturation,
# small |u| samples the linear foot). All three components recorded at
# monitor_2; parts store the RAW COMPLEX FIELD (readout applied at assemble).
#
# SUBMIT (the user submits; never srun/sbatch from the assistant):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   N=$(/project/cerneziga/mamba_x86/envs/pmp/bin/python \
#         data_gen/generate_ipc_data.py \
#         --path data/reservoir_types/res_sigmoid_gate --n 1000 --count)
#   sbatch --array=0-$((N-1))%6 scripts/slurm_ipc_lips.sh
#
#   (%6 = one task per F5 CPU node; f02..f07 is 6 nodes. Raise cpus-per-task
#    with --cpus-per-task=<cores> at submit time if the nodes have more.)
#
# ASSEMBLE when the array is done (login node, 1 rank, seconds):
#   /project/cerneziga/mamba_x86/envs/pmp/bin/python \
#       data_gen/generate_ipc_data.py \
#       --path data/reservoir_types/res_sigmoid_gate --n 1000 --scale 30 \
#       --n_sources 4 --out_sensor monitor_2 --components Ex,Ey,Ez --assemble
#   -> data/reservoir_types/res_sigmoid_gate/datasets/ipc.npz {inputs, outputs}
#
# Prereqs on lips (all already in place per About servers.md unless noted):
#   * repo at /home/cerneziga/resevoir with this design pulled  (git pull runs
#     ONLY from workbox through the /home/ziga/Lips sshfs mount — never here)
#   * x86 MEEP env /project/cerneziga/mamba_x86/envs/pmp (mpi_mpich pymeep;
#     built by scripts/lips_build_pmp_env.sh — NOT the aarch64 `opt` env)
#   * design is isotropic: no LC relax cache needed
#
#SBATCH --job-name=ipc_sigmoid
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/ipc_sigmoid_%A_%a.log

set -euo pipefail

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
ENVBIN=/project/cerneziga/mamba_x86/envs/pmp/bin
PY=$ENVBIN/python
MPI=$ENVBIN/mpirun
DESIGN=data/reservoir_types/res_sigmoid_gate

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: x86 pmp env missing — run scripts/lips_build_pmp_env.sh first"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export RESERVOIR_SOLVER=meep          # F5 CPU nodes have no GPU
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
# Per-task scratch tag: concurrent tasks share the design's simulation_meep/
# dir; with one tag they clobber each other's monitor npz between run() and
# read-back and silently swap results (see slurm_char_array.sh history).
export SIMPLESIM_SCRATCH_TAG="ipcsig${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

NRANK=${SLURM_CPUS_PER_TASK:-64}

cd "$BASE_DIR"
echo "=== ipc res_sigmoid_gate task ${SLURM_ARRAY_TASK_ID:-?} host $(hostname) ranks $NRANK tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$MPI -np "$NRANK" $PY -u data_gen/generate_ipc_data.py \
    --path "$DESIGN" \
    --n 1000 --scale 30 --n_sources 4 \
    --out_sensor monitor_2 --components Ex,Ey,Ez \
    --skip_existing --index "${SLURM_ARRAY_TASK_ID}"
echo "=== task ${SLURM_ARRAY_TASK_ID:-?} done $(date) ==="
