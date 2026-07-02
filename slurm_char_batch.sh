#!/bin/bash
# Characterization data gen — BATCH job: one job runs a batch of forward runs
# sequentially on 8 cores. Submit multiple jobs (one per batch id) to cover all items.
# Small 8-core jobs backfill into scattered idle cores (better cluster time under qos=soft)
# than full-node jobs. Each forward run writes its own part immediately (incremental).
#
# Batch id comes from $SLURM_ARRAY_TASK_ID (array mode, preferred) or a positional arg.
#
# Preferred — ONE array submission covers all batches (batch id = array task id):
#   sbatch --array=0-9 slurm_char_batch.sh <method> <design> [--batch_size N] [gen args]
#   e.g. sbatch --array=0-9 slurm_char_batch.sh superposition data/reservoir_clasifications/01_2D_director --batch_size 5 --n_base 10 --n_trials 40
#   (nbatch = ceil(N_items / batch_size);  N_items = $(PY <gen>.py --path <design> [args] --count))
#
# Or individual jobs — batch id as positional $3:
#   sbatch slurm_char_batch.sh <method> <design> <batch_id> [--batch_size N] [gen args]
#
# Then assemble once all batches done: PY <gen>.py --path <design> [args] --assemble

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

METHOD=${1:?usage: sbatch [--array=0-K] slurm_char_batch.sh <method> <design> [batch_id] [args]}
PATH_ARG=${2:?usage: sbatch [--array=0-K] slurm_char_batch.sh <method> <design> [batch_id] [args]}
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    BATCH_ID=$SLURM_ARRAY_TASK_ID       # array mode: batch id = array task id
    shift 2
else
    BATCH_ID=${3:?need a batch_id (positional) or submit with --array}
    shift 3
fi
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
