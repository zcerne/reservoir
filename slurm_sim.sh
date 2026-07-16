#!/bin/bash
# ONE slurm script for reservoir design simulation on Orion — pick the stages
# with flags, design folder via --path.
#
# Usage:
#   sbatch slurm_sim.sh --path data/<design> [--lcrelax] [--gpumeep] [--meep]
#
#   --path      design folder (REQUIRED; the only required arg)
#   --lcrelax   LC relaxation (class_reservoir) -> simulation/lc_fields.npz
#   --gpumeep   gpumeep engine forward (class_simulation_gpu, single rank)
#   --meep      MEEP forward (class_simulation, full-node MPI)
#   --empty     ALSO run the empty (air reference) variant for each selected
#               engine -> simulation_empty/ (transmission normalisation)
#
# All stage flags optional; NO stage flags -> --meep is the default.
# Stages run in the order lcrelax -> gpumeep -> meep (any subset; e.g.
#   sbatch slurm_sim.sh --path data/lasing_testing/01_basic_test --lcrelax --gpumeep
# ). Isotropic designs (reservoir.isotropic=true) don't need --lcrelax.
#
# Precision: JSON top-level "fp": "fp32"|"fp64" (default fp64) — read by
# class_simulation_gpu before jax import. Runs as-is on smaug too
# (./slurm_sim.sh ...): #SBATCH lines are ignored, N=nproc adapts, and the
# 4090 is used automatically for --gpumeep/--lcrelax when present.
#
# Logs: <design>/simulation/{lcrelax,gpumeep,meep}.log  (tee'd, absolute paths —
# never rely on #SBATCH --output with relative paths: Slurm kills the job
# silently if the dir doesn't exist at submission).

#SBATCH --nodes=1
#SBATCH --partition=of
#SBATCH --qos=soft
#SBATCH --time=12:00:00
#SBATCH --mem=1900GB
#SBATCH --cpus-per-task=96
#SBATCH --output=/home/cernez/resevoir/slurm_sim_%j.log

set -e

BASE_DIR="/home/cernez/resevoir"
PMP="/home/cernez/micromamba/envs/pmp/bin"
PY="$PMP/python"
MPIRUN="$PMP/mpirun"
N=$(nproc)                              # MEEP ranks = all cores of THIS host
export GPUMEEP_PATH="/home/cernez/GPUmeep/src"
# HOST-AWARE jax platform: use the GPU when one exists (smaug 4090 via
# ./slurm_sim.sh), CPU otherwise (orion nodes via sbatch). Applies to the
# --gpumeep stage and the jax-based LC relax.
if nvidia-smi -L >/dev/null 2>&1; then
    export JAX_PLATFORMS=cuda,cpu
else
    export JAX_PLATFORMS=cpu
fi

# ---- parse flags ------------------------------------------------------
DESIGN=""
DO_LCRELAX=""; DO_GPUMEEP=""; DO_MEEP=""; DO_EMPTY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --path)     DESIGN="$2"; shift 2 ;;
        --lcrelax)  DO_LCRELAX=1; shift ;;
        --gpumeep)  DO_GPUMEEP=1; shift ;;
        --meep)     DO_MEEP=1; shift ;;
        --empty)    DO_EMPTY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$DESIGN" ]; then
    echo "usage: sbatch slurm_sim.sh --path data/<design> [--lcrelax] [--gpumeep] [--meep]" >&2
    exit 2
fi
if [ -z "$DO_LCRELAX$DO_GPUMEEP$DO_MEEP" ]; then
    DO_MEEP=1                           # default stage
fi

cd "$BASE_DIR"
SIM_DIR="$BASE_DIR/$DESIGN/simulation"
mkdir -p "$SIM_DIR"
echo "=== slurm_sim: $DESIGN  (lcrelax=${DO_LCRELAX:-0} gpumeep=${DO_GPUMEEP:-0} meep=${DO_MEEP:-0}) ==="

# ---- 1) LC relaxation (single rank) -----------------------------------
if [ -n "$DO_LCRELAX" ]; then
    echo "=== LC relax: $DESIGN ===" | tee "$SIM_DIR/lcrelax.log"
    $PY class_reservoir.py --path "$DESIGN" 2>&1 | tee -a "$SIM_DIR/lcrelax.log"
fi

# ---- 2) gpumeep forward (single rank; GPU when present) -----------------
if [ -n "$DO_GPUMEEP" ]; then
    echo "=== gpumeep run: $DESIGN ===" | tee "$SIM_DIR/gpumeep.log"
    $PY class_simulation_gpu.py --path "$DESIGN" 2>&1 | tee -a "$SIM_DIR/gpumeep.log"
    if [ -n "$DO_EMPTY" ]; then
        echo "=== gpumeep EMPTY run: $DESIGN ===" | tee "$SIM_DIR/gpumeep_empty.log"
        $PY class_simulation_gpu.py --path "$DESIGN" --empty \
            2>&1 | tee -a "$SIM_DIR/gpumeep_empty.log"
    fi
fi

# ---- 3) MEEP forward (full-node MPI) -----------------------------------
if [ -n "$DO_MEEP" ]; then
    echo "=== MEEP run: $DESIGN ($N ranks) ===" | tee "$SIM_DIR/meep.log"
    $MPIRUN -np $N $PY class_simulation.py --path "$DESIGN" --lc-only \
        2>&1 | tee -a "$SIM_DIR/meep.log"
    if [ -n "$DO_EMPTY" ]; then
        echo "=== MEEP EMPTY run: $DESIGN ($N ranks) ===" | tee "$SIM_DIR/meep_empty.log"
        $MPIRUN -np $N $PY class_simulation.py --path "$DESIGN" --empty-only \
            2>&1 | tee -a "$SIM_DIR/meep_empty.log"
    fi
fi

echo "=== slurm_sim done: $DESIGN ==="
