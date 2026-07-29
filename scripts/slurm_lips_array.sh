#!/usr/bin/bash -l
#SBATCH --job-name=res_gen
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/slurm_%A_%a.log
#
# Reservoir data generation on lips (IJS), GH200 / aarch64, one forward run per
# array task. Follows the split the user asked for: CODE on /home/cerneziga,
# RESULTS AND DATASETS on /project/cerneziga.
#
#   sbatch --array=0-399%8 scripts/slurm_lips_array.sh \
#       ipc /project/cerneziga/reservoir_runs/04_LC_4src \
#       --n 400 --scale 10 --out_sensor n2f_map --components Ex,Ey,Ez
#
#   method = superposition | harmonics | ampsweep | ipc | balance
#
# Partition: `grace` has 8 GH200 nodes and is usually less contended than
# `F5-gpu` (2 nodes, where the 3D BlockOpt runs live — don't compete with them).
#
# !! UNVERIFIED ON LIPS !! I could not test this: ssh to lips refuses my key
# (`Permission denied (publickey)`), so I could only write files over the sshfs
# mount. Specifically unconfirmed:
#   * that the `opt` env has a working jax+CUDA on aarch64 for gpumeep
#   * that gpumeep runs at all on GH200 (never executed there)
# If gpumeep fails, fall back to the MEEP backend by setting
# RESERVOIR_SOLVER=meep below — CPU only, but it needs no GPU stack.

set -e

CODE=/home/cerneziga/resevoir
export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export LCRELAX_PATH=/home/cerneziga/LCrelax
# LCrelax must precede anything else that ships duplicate helpers
export PYTHONPATH="$LCRELAX_PATH:$PYTHONPATH"
export JAX_PLATFORMS=cuda,cpu
# never preallocate: several array tasks may land on one node and the first
# would otherwise take ~75% of the GPU and starve the rest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# per-task scratch, or concurrent tasks overwrite each other's monitor npz
export SIMPLESIM_SCRATCH_TAG="lips${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

PY=/project/cerneziga/micromamba/envs/opt/bin/python

METHOD=${1:?usage: sbatch --array=0-(N-1) scripts/slurm_lips_array.sh <method> <design_dir_on_project> [args]}
DESIGN=${2:?usage: ... <method> <design_dir_on_project> [args]}
shift 2

case "$METHOD" in
    superposition) GEN=data_gen/generate_superposition_data.py ;;
    harmonics)     GEN=data_gen/generate_harmonics_data.py ;;
    ampsweep)      GEN=data_gen/generate_amplitude_sweep_data.py ;;
    ipc)           GEN=data_gen/generate_ipc_data.py ;;
    balance)       GEN=data_gen/generate_balance_scale_data.py ;;
    *) echo "unknown method '$METHOD'"; exit 1 ;;
esac

cd "$CODE"
echo "=== $METHOD $DESIGN task ${SLURM_ARRAY_TASK_ID} host $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU visible)"
$PY -u "$GEN" --path "$DESIGN" "$@" --skip_existing --index "$SLURM_ARRAY_TASK_ID"
echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
