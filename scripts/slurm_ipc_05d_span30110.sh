#!/bin/bash
# IPC dataset #4 for 05d — amplitude SPAN AROUND I_SAT: amp = 70 + 40*u,
# i.e. 30..110 per strip (amp_sat = 73). The bias parks every probe at the
# knee where the gain curvature is maximal, the +-40 swing keeps degree-k
# products alive, and the all-positive drive breaks the u -> -u parity
# symmetry (no more odd/even walls). Same 1000 probes as the other datasets
# (same seed). F5 CPU array (GPUs are busy with the comb dataset): one
# 64-rank MPI MEEP run per array task; amplitude encoding is backend-agnostic
# (gpumeep/MEEP cross-validated 0.3% on 05d).
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# WHY: the phase-encoding run (2026-09-11) proved the medium computes via
# intensity-driven saturation — this run modulates intensity ACROSS the knee
# every probe. Max drive 110 all-strips validated clean (job 29782666).
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
#       --path data/signal_modulation/05d_ampsweep --n 1000 --scale 40 --offset 70 \
#       --out_sensor monitor_2 --components Ex,Ey,Ez \
#       --out data/signal_modulation/05d_ampsweep/datasets/ipc_span30_110.npz \
#       --assemble
#
#SBATCH --job-name=ipc05dspan
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/ipc05dspan_%A_%a.log

set -euo pipefail

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/mamba_x86/envs/pmp/bin/python
MPI=$(dirname "$PY")/mpirun
DESIGN=data/signal_modulation/05d_ampsweep
OUT=$DESIGN/datasets/ipc_span30_110.npz

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: x86 pmp python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export RESERVOIR_SOLVER=meep
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Per-task scratch tag: concurrent tasks share simulation_gpumeep/; without a
# unique tag they clobber each other's monitor npz and silently swap results.
export SIMPLESIM_SCRATCH_TAG="ipcspan_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

cd "$BASE_DIR"
echo "=== ipc 05d span30-110 task ${SLURM_ARRAY_TASK_ID:-?} host $(hostname) tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$MPI -np "${SLURM_NTASKS:-64}" $PY -u data_gen/generate_ipc_data.py \
    --path "$DESIGN" \
    --n 1000 --scale 40 --offset 70 \
    --out_sensor monitor_2 --components Ex,Ey,Ez \
    --out "$OUT" \
    --skip_existing --index "${SLURM_ARRAY_TASK_ID}"
echo "=== task ${SLURM_ARRAY_TASK_ID:-?} done $(date) ==="
