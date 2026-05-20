#!/bin/bash
# Slurm job script for LC reservoir simulation on Orion HPC.
# Usage: sbatch slurm.sh data/test33D
#        sbatch slurm.sh              (defaults to data/test33D)
#        sbatch slurm.sh data/test2D build-T   (also build T matrix after LC+empty runs)
#
# Prerequisite: run LC minimization locally before submitting:
#   python class_reservoir.py --path data/test2D
#
# Runs 2 (or 3) sequential steps:
#   1. MEEP LC simulation        — 96 MPI processes, pmp conda env
#   2. MEEP air reference run    — 96 MPI processes, pmp conda env
#   3. (optional) T matrix build — 96 MPI processes, pmp conda env (--build-T flag)
#
# Logs: $PATH_ARG/simulation/meep_lc.log, meep_empty.log
#       $PATH_ARG/simulation_T/build_T.log  (if --build-T)

#SBATCH --nodes=1
#SBATCH --partition=of
#SBATCH --qos=soft
#SBATCH --time=24:00:00
#SBATCH --mem=1900GB          # full node memory
#SBATCH --cpus-per-task=96   # all cores on node

set -e

BASE_DIR="/home/cernez/resevoir"
PATH_ARG=${1:-data/test33D}
BUILD_T=${2:-}
N=96

SIM_DIR="$BASE_DIR/$PATH_ARG/simulation"
mkdir -p "$SIM_DIR"

cd "$BASE_DIR"

# Step 1: MEEP simulation with LC director field loaded from lc_fields.npz
echo "=== MEEP LC run: $PATH_ARG ($N processes) ===" | tee "$SIM_DIR/meep_lc.log"
mpirun -np $N python class_simulation.py --path "$PATH_ARG" --lc-only \
    >> "$SIM_DIR/meep_lc.log" 2>&1

# Step 2: MEEP simulation with air (no LC geometry) — reference for transmission %
echo "=== MEEP empty run: $PATH_ARG ($N processes) ===" | tee "$SIM_DIR/meep_empty.log"
mpirun -np $N python class_simulation.py --path "$PATH_ARG" --empty-only \
    >> "$SIM_DIR/meep_empty.log" 2>&1

# Step 3 (optional): T matrix build — N basis MEEP runs
if [ -n "$BUILD_T" ]; then
    T_DIR="$BASE_DIR/$PATH_ARG/simulation_T"
    mkdir -p "$T_DIR"
    echo "=== T matrix build: $PATH_ARG ($N processes) ===" | tee "$T_DIR/build_T.log"
    mpirun -np $N python class_simulation_T.py --path "$PATH_ARG" --build-T \
        >> "$T_DIR/build_T.log" 2>&1
fi

echo "=== Done: $PATH_ARG ==="
