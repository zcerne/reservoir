#!/bin/bash
# Slurm ARRAY job for reservoir data generation on Orion — one forward run per
# array task, so wall-clock is one MEEP run rather than N of them. Each task
# writes its own part file; a final --assemble combines them.
#
# Restored and fixed 2026-07-29 (the original was deleted in the repo strip-down
# and predates SimpleSim). Four things it now does that it did not before:
#   * exports SIMPLESIM_PATH / GPUMEEP_PATH — open_reservoir imports SimpleSim
#     through run._ensure_simplesim and dies with ModuleNotFoundError without them
#   * forces RESERVOIR_SOLVER=meep and JAX_PLATFORMS=cpu — the design JSONs say
#     solver "gpumeep", but Orion compute nodes have no GPU
#   * gives every array task its OWN SIMPLESIM_SCRATCH_TAG. This one is critical:
#     concurrent tasks share the design's simulation_<backend>/ scratch dir, so
#     with one tag they overwrite each other's monitor npz between run() and the
#     read-back, and every task silently records some other task's fields.
#   * passes --skip_existing, so a resubmitted array costs seconds on finished
#     indices instead of recomputing them
#
# Usage (2 steps):
#   1) count the work items and submit:
#        cd /home/cernez/resevoir
#        N=$(~/micromamba/envs/pmp/bin/python data_gen/generate_ipc_data.py \
#              --path data/lasing_testing/04_LC_4src --n 400 --count)
#        sbatch --array=0-$((N-1))%40 scripts/slurm_char_array.sh \
#              ipc data/lasing_testing/04_LC_4src \
#              --n 400 --scale 10 --out_sensor n2f_map --components Ex,Ey,Ez
#   2) when the array finishes, assemble (1 rank, quick):
#        ~/micromamba/envs/pmp/bin/python data_gen/generate_ipc_data.py \
#              --path data/lasing_testing/04_LC_4src --n 400 --scale 10 \
#              --out_sensor n2f_map --components Ex,Ey,Ez --assemble
#
#   method = superposition | harmonics | ampsweep | ipc | balance
#
# Prereq: the design's LC relaxation must already be cached
# (simulation/lc_fields_*.npz), or every task redoes it. Isotropic designs need
# nothing. Check with:  ls data/lasing_testing/<design>/simulation/

# One FULL node per task. The MEEP sim is compute-bound and genuinely uses all 96
# cores; packing many small-core MPI jobs on a node caused PMI-init failures and
# ~26x slowdown (they don't co-locate). So: one sim per node, as many in parallel
# as there are free nodes. Reduce the WORKLOAD (fewer probes / run_until), not
# the cores per task.
#SBATCH --nodes=1
#SBATCH --partition=of,xaos
#SBATCH --qos=soft
#SBATCH --time=2-00:00:00
#SBATCH --mem=180GB
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm_char_%A_%a.log

set -e

BASE_DIR="/home/cernez/resevoir"
PYTHON_MEEP=/home/cernez/micromamba/envs/pmp/bin/python
MPIRUN=/home/cernez/micromamba/envs/pmp/bin/mpirun
NRANK=72                      # MPI ranks per forward run. 72 not 96 so the job
                              # is schedulable on BOTH partitions: taurus (of) has
                              # 96 cores/2 TB, xaos has 72 cores/191 GB. Asking 96
                              # or 1900 GB silently excludes every xaos node.

export SIMPLESIM_PATH=/home/cernez/SimpleSim
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export RESERVOIR_SOLVER=meep  # compute nodes have no GPU
export JAX_PLATFORMS=cpu
# per-task scratch: without this, concurrent tasks clobber each other's
# simulation_meep/*.npz and silently swap results between runs
export SIMPLESIM_SCRATCH_TAG="arr${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

METHOD=${1:?usage: sbatch --array=0-(N-1) scripts/slurm_char_array.sh <method> <design> [args]}
PATH_ARG=${2:?usage: sbatch --array=0-(N-1) scripts/slurm_char_array.sh <method> <design> [args]}
shift 2
EXTRA=("$@")

case "$METHOD" in
    superposition) GEN=data_gen/generate_superposition_data.py ;;
    harmonics)     GEN=data_gen/generate_harmonics_data.py ;;
    ampsweep)      GEN=data_gen/generate_amplitude_sweep_data.py ;;
    ipc)           GEN=data_gen/generate_ipc_data.py ;;
    balance)       GEN=data_gen/generate_balance_scale_data.py ;;
    *) echo "unknown method '$METHOD'"; exit 1 ;;
esac

cd "$BASE_DIR"
echo "=== $METHOD $PATH_ARG task ${SLURM_ARRAY_TASK_ID} host $(hostname) tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$MPIRUN -np $NRANK $PYTHON_MEEP "$GEN" --path "$PATH_ARG" "${EXTRA[@]}" \
        --skip_existing --index "$SLURM_ARRAY_TASK_ID"
echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
