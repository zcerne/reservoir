#!/bin/bash
# Amplitude sweep on Orion with SMALL MPI jobs: 8 ranks per task, many array
# tasks in parallel — the opposite trade to slurm_char_array.sh (one full node
# per task). Use when the per-run cell is small (e.g. the mirror-free 30x10
# block designs at run_until=100) and node availability, not per-run speed, is
# the bottleneck.
#
# CAVEAT, known from 2026-07: multiple MEEP jobs CO-LOCATED on one node have
# caused PMI-init failures and large slowdowns (they fight for memory
# bandwidth). Requested explicitly anyway 2026-08-06 for the block ampsweep —
# the cells are small. If tasks start dying at MPI init or each run takes
# many times the single-job time, fall back to slurm_char_array.sh, or add
# --exclusive to give each 8-rank task a node of its own.
#
# Usage (from /home/cernez/resevoir on orion):
#   N=$(~/micromamba/envs/pmp/bin/python data_gen/generate_amplitude_sweep_data.py \
#         --path data/reservoir_types/block_iso_gain \
#         --levels 1,2,5,10,20,50,100,200,350,500 --n_probes 12 --count)
#   sbatch --array=0-$((N-1))%12 scripts/slurm_orion_amp8.sh \
#         ampsweep data/reservoir_types/block_iso_gain \
#         --levels 1,2,5,10,20,50,100,200,350,500 --n_probes 12 \
#         --out_sensor n2f_map --components Ex,Ey,Ez
#   # afterwards (1 rank, quick):
#   ~/micromamba/envs/pmp/bin/python data_gen/generate_amplitude_sweep_data.py \
#         --path data/reservoir_types/block_iso_gain \
#         --levels 1,2,5,10,20,50,100,200,350,500 --n_probes 12 \
#         --out_sensor n2f_map --components Ex,Ey,Ez --assemble
#
# Isotropic designs need no LC cache; LC designs must have
# simulation/lc_fields_*.npz cached first or every task redoes the relax.
#SBATCH --job-name=amp8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --partition=of
#SBATCH --qos=soft
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/cernez/resevoir/slurm_amp8_%A_%a.log

set -e

BASE_DIR="/home/cernez/resevoir"
PYTHON_MEEP=/home/cernez/micromamba/envs/pmp/bin/python
MPIRUN=/home/cernez/micromamba/envs/pmp/bin/mpirun
NRANK=${SLURM_NTASKS:-8}

export SIMPLESIM_PATH=/home/cernez/SimpleSim
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export RESERVOIR_SOLVER=meep      # design JSONs say gpumeep; no GPU on orion
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
# per-task scratch — without it concurrent tasks clobber each other's monitor
# npz in the shared simulation_meep/ dir and silently swap results
export SIMPLESIM_SCRATCH_TAG="amp8_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

METHOD=${1:?usage: sbatch --array=0-(N-1)%%K scripts/slurm_orion_amp8.sh <method> <design> [args]}
PATH_ARG=${2:?usage: ... <method> <design> [args]}
shift 2
EXTRA=("$@")

case "$METHOD" in
    superposition) GEN=data_gen/generate_superposition_data.py ;;
    harmonics)     GEN=data_gen/generate_harmonics_data.py ;;
    ampsweep)      GEN=data_gen/generate_amplitude_sweep_data.py ;;
    ipc)           GEN=data_gen/generate_ipc_data.py ;;
    gr)            GEN=data_gen/generate_gr_data.py ;;
    balance)       GEN=data_gen/generate_balance_scale_data.py ;;
    *) echo "unknown method '$METHOD'"; exit 1 ;;
esac

cd "$BASE_DIR"
echo "=== $METHOD $PATH_ARG task ${SLURM_ARRAY_TASK_ID} host $(hostname) ranks $NRANK tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$MPIRUN -np "$NRANK" $PYTHON_MEEP "$GEN" --path "$PATH_ARG" "${EXTRA[@]}" \
        --skip_existing --index "$SLURM_ARRAY_TASK_ID"
echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
