#!/bin/bash
# Slurm ARRAY job for reservoir characterization data gen on Orion — one forward run
# per array task (parallel; wall-clock ≈ one MEEP run, not N of them). Each task writes
# its own part file (incremental save); a final --assemble combines them.
#
# Usage (2 steps):
#   1) find the number of work items N, submit the array 0..N-1:
#        N=$(/home/cernez/micromamba/envs/pmp/bin/python data_gen/generate_<gen>.py --path <design> [args] --count)
#        sbatch --array=0-$((N-1))%<concurrency> slurm_char_array.sh <method> <design> [args]
#   2) after the array finishes, assemble (quick, 1 rank):
#        /home/cernez/micromamba/envs/pmp/bin/python data_gen/generate_<gen>.py --path <design> [args] --assemble
#
#   method = superposition | harmonics | ampsweep | ipc
# Example:
#   N=$(.../python data_gen/generate_ipc_data.py --path data/reservoir_clasifications/01_2D_director --n 400 --count)   # -> 400
#   sbatch --array=0-399%40 slurm_char_array.sh ipc data/reservoir_clasifications/01_2D_director --n 400 --readout intensity
#   .../python data_gen/generate_ipc_data.py --path data/reservoir_clasifications/01_2D_director --n 400 --readout intensity --assemble
#
# Prereq: relaxed LC at the design dir (python class_reservoir.py --path <design>).

# One FULL node per task. The MEEP sim is compute-bound and genuinely uses all 96
# cores (~10 min); packing many small-core MPI jobs on a node caused PMI-init failures
# + ~26x slowdown (they don't co-locate). So: one sim per node, run as many in
# parallel as there are free nodes. Reduce the WORKLOAD (fewer probes / run_until),
# not the cores-per-task, to make a run tractable.
#SBATCH --nodes=1
#SBATCH --partition=of
#SBATCH --qos=soft
#SBATCH --time=2:00:00
#SBATCH --mem=1900GB
#SBATCH --cpus-per-task=96
#SBATCH --output=slurm_char_%A_%a.log

set -e

BASE_DIR="/home/cernez/resevoir"
PYTHON_MEEP=/home/cernez/micromamba/envs/pmp/bin/python
MPIRUN=/home/cernez/micromamba/envs/pmp/bin/mpirun
N=96  # MPI ranks per forward run = full node (compute-bound sim; don't co-locate)

METHOD=${1:?usage: sbatch --array=0-(N-1) slurm_char_array.sh <method> <design> [args]}
PATH_ARG=${2:?usage: sbatch --array=0-(N-1) slurm_char_array.sh <method> <design> [args]}
shift 2
EXTRA=("$@")

case "$METHOD" in
    superposition) GEN=data_gen/generate_superposition_data.py ;;
    harmonics)     GEN=data_gen/generate_harmonics_data.py ;;
    ampsweep)      GEN=data_gen/generate_amplitude_sweep_data.py ;;
    ipc)           GEN=data_gen/generate_ipc_data.py ;;
    *) echo "unknown method '$METHOD'"; exit 1 ;;
esac

cd "$BASE_DIR"
echo "=== $METHOD $PATH_ARG  task ${SLURM_ARRAY_TASK_ID}  host $(hostname)  $(date) ==="
$MPIRUN -np $N $PYTHON_MEEP "$GEN" --path "$PATH_ARG" "${EXTRA[@]}" --index "$SLURM_ARRAY_TASK_ID"
echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
