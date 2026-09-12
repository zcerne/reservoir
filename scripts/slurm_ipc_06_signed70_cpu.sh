#!/usr/bin/bash -l
# IPC dataset on 06_recurrent_R01 — the DOUBLE-PASS capacity measurement.
# Signed amplitude +-70 (the sign-task champion encoding), SAME 1000 probes
# as every 05d dataset: the Dambre delta vs ipc_amp70 isolates what the
# R~0.09 exit mirror (+ the standing-wave population grating it writes) buys.
# run 8000, window 5000-8000 (design), MEEP on F5 CPUs (GPUs busy with comb;
# amplitude encoding is backend-agnostic, 0.3% RMS cross-validated).
# Stationarity of the mirrored design verified 2026-09-12 (settle 2420).
#
# SUBMIT (user only):
#   cd /home/cerneziga/resevoir
#   sbatch --array=0-999%7 scripts/slurm_ipc_06_signed70_cpu.sh
#
# ASSEMBLE (login node, x86 pmp env):
#   /project/cerneziga/mamba_x86/envs/pmp/bin/python \
#       data_gen/generate_ipc_data.py \
#       --path data/signal_modulation/06_recurrent_R01 --n 1000 --scale 70 \
#       --out_sensor monitor_2 --components Ex,Ey,Ez \
#       --out data/signal_modulation/06_recurrent_R01/datasets/ipc_signed70.npz \
#       --assemble
#
#SBATCH --job-name=ipc06rec
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/ipc06rec_%A_%a.log

set -euo pipefail

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/mamba_x86/envs/pmp/bin/python
MPI=$(dirname "$PY")/mpirun
DESIGN=data/signal_modulation/06_recurrent_R01
OUT=$DESIGN/datasets/ipc_signed70.npz

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: x86 pmp python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export RESERVOIR_SOLVER=meep
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Per-task scratch tag: concurrent tasks share simulation_gpumeep/; without a
# unique tag they clobber each other's monitor npz and silently swap results.
export SIMPLESIM_SCRATCH_TAG="ipc06_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

cd "$BASE_DIR"
echo "=== ipc 05d span30-110 task ${SLURM_ARRAY_TASK_ID:-?} host $(hostname) tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$MPI -np "${SLURM_NTASKS:-64}" $PY -u data_gen/generate_ipc_data.py \
    --path "$DESIGN" \
    --n 1000 --scale 70 \
    --out_sensor monitor_2 --components Ex,Ey,Ez \
    --out "$OUT" \
    --skip_existing --index "${SLURM_ARRAY_TASK_ID}"
echo "=== task ${SLURM_ARRAY_TASK_ID:-?} done $(date) ==="
