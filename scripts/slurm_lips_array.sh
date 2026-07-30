#!/usr/bin/bash -l
#SBATCH --job-name=res_gen
#SBATCH --partition=F5-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/slurm_%A_%a.log
#
# Reservoir data generation on lips (IJS) F5 partition — 2 x H200 — one forward run per
# array task. Follows the split the user asked for: CODE on /home/cerneziga,
# RESULTS AND DATASETS on /project/cerneziga.
#
#   sbatch --array=0-399%2 scripts/slurm_lips_array.sh \
#       ipc /project/cerneziga/reservoir_runs/04_LC_4src \
#       --n 400 --scale 10 --out_sensor n2f_map --components Ex,Ey,Ez
#
#   method = superposition | harmonics | ampsweep | ipc | balance
#
# Partition: F5 — 2 x H200 nodes (user, 2026-07-29; not documented anywhere yet,
# and NOT the GH200/Grace hardware my older notes describe). Only 2 nodes, so
# cap array concurrency at %2; anything higher just queues. H200 is a normal
# x86_64 host, so ordinary x86 CUDA wheels apply — none of the aarch64 caveats
# that apply to the grace partition.
#
# !! UNVERIFIED ON LIPS !! I could not test this: ssh to lips refuses my key
# (`Permission denied (publickey)`), so I could only write files over the sshfs
# mount. Specifically unconfirmed:
#   * the python env path below (is `opt` the right env on an x86_64 H200 node,
#     or does F5 need its own?)
#   * that gpumeep runs on H200 at all (never executed there)
#   * the exact partition name — `F5-gpu` is what the older sinfo showed
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

# gpumeep compiles the whole FDTD as one lax.scan, and forward() rebuilds the
# Simulation for every sample, so without a persistent cache that 20+ minute
# compile is repaid on EVERY sample (measured 2026-07-30: 2696 s per forward, of
# which the FDTD itself was ~40 s). The cache lives on /project so it is shared
# by every task and both F5 nodes: the first task compiles, the rest load it.
#
# Submit the FIRST batch with %1 so exactly one task warms the cache instead of
# several paying for the same compile in parallel:
#   sbatch --array=0-7%1 --export=ALL,BATCH_SIZE=50 scripts/slurm_lips_array.sh …
# Once /project/cerneziga/.jax_cache is populated, %2 is fine.
export JAX_COMPILATION_CACHE_DIR=/project/cerneziga/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
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

# BATCH_SIZE set (via --export=ALL,BATCH_SIZE=50) -> each array task runs a
# CONTIGUOUS BLOCK of samples in one process instead of a single --index.
# That matters here: python + JAX + CUDA init and the design load cost roughly
# as much as one forward run, so one-sample-per-task throws away about half the
# GPU. With 625 samples, 13 tasks of 50 is far better than 625 tasks of 1.
cd "$CODE"
echo "=== $METHOD $DESIGN task ${SLURM_ARRAY_TASK_ID} host $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU visible)"
if [ -n "$BATCH_SIZE" ]; then
    $PY -u "$GEN" --path "$DESIGN" "$@" --skip_existing \
        --batch "$SLURM_ARRAY_TASK_ID" --batch_size "$BATCH_SIZE"
else
    $PY -u "$GEN" --path "$DESIGN" "$@" --skip_existing --index "$SLURM_ARRAY_TASK_ID"
fi
echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
