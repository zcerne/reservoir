#!/bin/bash
# Slurm job for reservoir CHARACTERIZATION data generation on Orion HPC.
# Runs one of the data_gen/ generators (each loops many MEEP forward runs, MPI-parallel).
#
# Usage:
#   sbatch slurm_characterization.sh <method> <design_path> [extra args ...]
#     method       = superposition | harmonics | ampsweep | ipc
#     design_path  = e.g. data/test2D  (must already have simulation_data.json + relaxed LC)
#
# Examples:
#   sbatch slurm_characterization.sh superposition data/test2D --n_base 8 --n_trials 40
#   sbatch slurm_characterization.sh harmonics     data/test2D --tones 5,7 --n_t 128
#   sbatch slurm_characterization.sh ampsweep      data/test2D --levels 0.1,0.3,1,3,10 --n_probes 12
#   sbatch slurm_characterization.sh ipc           data/test2D --n 400 --readout intensity
#
# Prerequisite: relax the LC first (locally or via run_sim.sh):
#   python class_reservoir.py --path data/test2D
#
# The generators are MPI-safe (writes guarded by meep am_master), so they run under
# mpirun -np $N: each forward run is one N-rank MEEP sim, looped over trials by the script.
# Output npz + log land under the design dir.

#SBATCH --nodes=1
#SBATCH --partition=of
#SBATCH --qos=soft
#SBATCH --time=24:00:00
#SBATCH --mem=1900GB          # full node memory
#SBATCH --cpus-per-task=96    # all cores on node

set -e

BASE_DIR="/home/cernez/resevoir"
PYTHON_MEEP=/home/cernez/miniconda3/envs/pmp/bin/python
MPIRUN=/home/cernez/miniconda3/envs/pmp/bin/mpirun
N=96

METHOD=${1:?usage: sbatch slurm_characterization.sh <method> <design_path> [args]}
PATH_ARG=${2:?usage: sbatch slurm_characterization.sh <method> <design_path> [args]}
shift 2
EXTRA=("$@")

case "$METHOD" in
    superposition) GEN=data_gen/generate_superposition_data.py ;;
    harmonics)     GEN=data_gen/generate_harmonics_data.py ;;
    ampsweep)      GEN=data_gen/generate_amplitude_sweep_data.py ;;
    ipc)           GEN=data_gen/generate_ipc_data.py ;;
    *) echo "unknown method '$METHOD' (superposition|harmonics|ampsweep|ipc)"; exit 1 ;;
esac

LOG_DIR="$BASE_DIR/$PATH_ARG"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gen_${METHOD}.log"

cd "$BASE_DIR"
echo "=== characterization gen: method=$METHOD path=$PATH_ARG ($N ranks) ===" | tee "$LOG"
echo "Host: $(hostname)  Date: $(date)  args: ${EXTRA[*]}" | tee -a "$LOG"

$MPIRUN -np $N $PYTHON_MEEP "$GEN" --path "$PATH_ARG" "${EXTRA[@]}" >> "$LOG" 2>&1

echo "=== Done: $METHOD $PATH_ARG at $(date) ===" | tee -a "$LOG"
