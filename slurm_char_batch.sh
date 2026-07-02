#!/bin/bash
# Characterization data gen — BATCH job: one job runs a batch of forward runs
# sequentially on 8 cores. Submit multiple jobs (one per batch id) to cover all items.
# Small 8-core jobs backfill into scattered idle cores (better cluster time under qos=soft)
# than full-node jobs. Each forward run writes its own part immediately (incremental).
#
# Usage:
#   sbatch slurm_char_batch.sh <method> <design> <batch_id> [--batch_size N] [gen args]
#     method   = superposition | harmonics | ampsweep | ipc
#     batch_id = 0,1,2,...  → items [batch_id*batch_size, +batch_size)
#
# How many batches?  N=$(PY <gen>.py --path <design> [args] --count); nbatch = ceil(N/batch_size).
# Example (superposition, 48 items, batch 12 → ids 0..3):
#   for b in 0 1 2 3; do
#     sbatch slurm_char_batch.sh superposition data/test2D $b --batch_size 12 --n_base 4 --n_trials 8
#   done
# Then assemble: PY <gen>.py --path <design> [args] --assemble

#SBATCH --nodes=1
#SBATCH --partition=of
#SBATCH --qos=soft
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_cbatch_%j.log

set -e

BASE_DIR="/home/cernez/resevoir"
PYTHON_MEEP=/home/cernez/miniconda3/envs/pmp/bin/python
MPIRUN=/home/cernez/miniconda3/envs/pmp/bin/mpirun
N=8   # MPI ranks per forward run (matches --cpus-per-task)

METHOD=${1:?usage: sbatch slurm_char_batch.sh <method> <design> <batch_id> [args]}
PATH_ARG=${2:?usage: sbatch slurm_char_batch.sh <method> <design> <batch_id> [args]}
BATCH_ID=${3:?usage: sbatch slurm_char_batch.sh <method> <design> <batch_id> [args]}
shift 3
EXTRA=("$@")

case "$METHOD" in
    superposition) GEN=data_gen/generate_superposition_data.py ;;
    harmonics)     GEN=data_gen/generate_harmonics_data.py ;;
    ampsweep)      GEN=data_gen/generate_amplitude_sweep_data.py ;;
    ipc)           GEN=data_gen/generate_ipc_data.py ;;
    *) echo "unknown method '$METHOD'"; exit 1 ;;
esac

cd "$BASE_DIR"
echo "=== $METHOD $PATH_ARG  batch $BATCH_ID  host $(hostname)  $(date)  args: ${EXTRA[*]} ==="
$MPIRUN -np $N $PYTHON_MEEP "$GEN" --path "$PATH_ARG" "${EXTRA[@]}" --batch "$BATCH_ID"
echo "=== batch $BATCH_ID done $(date) ==="
