#!/usr/bin/bash -l
#SBATCH --job-name=res_run
#SBATCH --partition=F5-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/slurm_run_%j.log
#
# ONE plain run.py on a lips GPU node: relax (or load the LC cache) + FDTD +
# plots for a single design. This is the single-simulation counterpart of
# scripts/slurm_lips_array.sh, which runs the data-generation campaigns.
#
#   sbatch scripts/slurm_lips_run.sh \
#       /project/cerneziga/reservoir_runs/reservoir_types/res_lc_gain/05b
#
# Any further arguments are passed straight to run.py, so the usual flags work:
#
#   sbatch scripts/slurm_lips_run.sh <design> --relax-only
#   sbatch scripts/slurm_lips_run.sh <design> --empty          # no reservoir, baseline
#   sbatch scripts/slurm_lips_run.sh <design> --suffix v2      # tag outputs
#   sbatch scripts/slurm_lips_run.sh <design> --backend meep   # CPU solver instead
#
# Partition: F5-gpu (h01/h02, 1 GH200 each). These are **aarch64** Grace Hopper
# nodes, which is why the env is the aarch64 `opt` build — see the longer note in
# scripts/slurm_lips_array.sh. The plain `F5` partition is x86 and GPU-less and
# needs a different env entirely (scripts/slurm_lips_cpu.sh).
#
# As of 2026-08-04 16:26 both h01 and h02 were DOWN — squeue showed
# `ReqNodeNotAvail, UnavailableNodes:h[01-02]`, so a job here stays PENDING
# until they come back. That is a node outage, not a queue wait.

set -e

CODE=/home/cerneziga/resevoir
export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export LCRELAX_PATH=/home/cerneziga/LCrelax
# LCrelax must precede anything else that ships duplicate helpers
export PYTHONPATH="$LCRELAX_PATH:$PYTHONPATH"
export JAX_PLATFORMS=cuda,cpu
# Shared persistent JAX cache: gpumeep compiles the whole FDTD as one lax.scan
# (20+ min cold). With the cache warm from the campaign jobs this run starts
# almost immediately.
export JAX_COMPILATION_CACHE_DIR=/project/cerneziga/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# distinct scratch tag so a manual run never collides with a running campaign's
# monitor npz files
export SIMPLESIM_SCRATCH_TAG="run${SLURM_JOB_ID:-0}"

PY=/project/cerneziga/micromamba/envs/opt/bin/python

DESIGN=${1:?usage: sbatch scripts/slurm_lips_run.sh <design_dir_on_project> [run.py args...]}
shift

cd "$CODE"
echo "=== run.py $DESIGN job ${SLURM_JOB_ID:-?} host $(hostname) $(date) ==="
echo "--- arch $(uname -m), partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU visible)"
# Provenance: the dye/gain update path changed on 2026-08-04, so record exactly
# which checkouts produced this run.
for R in "$CODE" "$SIMPLESIM_PATH" "$LCRELAX_PATH" "$(dirname "$GPUMEEP_PATH")"; do
    printf -- "--- %-34s %s\n" "$(basename "$R")" \
        "$(git -C "$R" log -1 --format='%h %ad %s' --date=short 2>/dev/null || echo 'not a git checkout')"
done

$PY -u run.py "$DESIGN" --backend gpumeep "$@"
echo "=== done $(date) ==="
