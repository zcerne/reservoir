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
# GPUmeep data generation on lips (IJS) GPU partitions — one contiguous block of
# samples per array task. Follows the split the user asked for: CODE on
# /home/cerneziga, RESULTS AND DATASETS on /project/cerneziga.
#
#   # 1000-sample IPC on 05b, 20 tasks of 50. FIRST submit with %1 to warm the
#   # JAX cache (see below), then resubmit the rest at full width:
#   sbatch --array=0%1     --export=ALL,BATCH_SIZE=50 scripts/slurm_lips_array.sh \
#       ipc /project/cerneziga/reservoir_runs/reservoir_types/res_lc_gain/05b \
#       --n 1000 --scale 50 --out_sensor n2f_map --components Ex,Ey,Ez
#   sbatch --array=1-19%2  --export=ALL,BATCH_SIZE=50 scripts/slurm_lips_array.sh  … (same args)
#
#   method = superposition | harmonics | ampsweep | ipc | balance
#
# ---------------------------------------------------------------- the hardware
# `F5-gpu` — 2 nodes, h01/h02, 1 GPU each (confirmed 2026-07-31 from a running
# job's log: "NVIDIA GH20…"). Only 2 GPUs, so %2 is the useful ceiling.
# `grace`  — 8 nodes, 1 GH200 each, historically less contended. For a
# 20-task campaign this is 4x the throughput of F5-gpu; override at submit:
#   sbatch --partition=grace --array=1-19%8 --export=ALL,BATCH_SIZE=50 …
#
# These GPU nodes are **GH200 Grace Hopper = aarch64**, NOT x86_64. That is why
# the env below is /project/cerneziga/micromamba/envs/opt: it is an aarch64
# build (verified from its ELF header, e_machine=0xb7 — see the note in
# scripts/slurm_lips_cpu.sh, which needs a *different*, x86_64 env because the
# plain `F5` CPU partition is AMD zen-4). Do not cross the two envs.
#
# Do NOT confuse `F5-gpu` with the plain `F5` partition: f02..f07, six
# 128-thread AMD nodes with Gres=(null), no GPUs at all. Those are for MEEP/CPU
# work and have their own script, scripts/slurm_lips_cpu.sh.
#
# ------------------------------------------------------------- BEFORE YOU RUN
# GPUmeep on lips must be at the same commit as the machine that produced the
# reference data, or the gain physics differs. As of 2026-08-04 lips was at
# 388757e (2026-07-30) while canonical was b553f83 — 16 commits behind,
# including two that change the dye/gain update path:
#   a2fe42a  drive polarizations with W fields (MEEP update_pols parity)
#   56bb649  get_array far-wall zero ghost; MEEP GammaInv-matmul parity in N update
# Any pumped/gain design (all of res_lc_gain, incl. 05b) is affected. The script
# echoes the GPUmeep commit into every log so a dataset can always be traced to
# the code that made it — check it in the first lines of the log.
#
# !! NEVER EXECUTED ON A LIPS GPU !! ssh to lips refuses my key
# (`Permission denied (publickey)`), so this was written over the sshfs mount and
# never run. Unconfirmed: that gpumeep's JAX/CUDA stack works inside the aarch64
# `opt` env on GH200 (that env was built for FDTDX, which does use jax+cuda, so
# it is plausible but untested). If it fails, the fallback is the CPU partition
# via scripts/slurm_lips_cpu.sh (RESERVOIR_SOLVER=meep, no GPU stack needed).

set -e

CODE=/home/cerneziga/resevoir
export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export LCRELAX_PATH=/home/cerneziga/LCrelax
# LCrelax must precede anything else that ships duplicate helpers
export PYTHONPATH="$LCRELAX_PATH:$PYTHONPATH"
export JAX_PLATFORMS=cuda,cpu
# Be explicit rather than trusting the design JSON: 05b's json says gpumeep, but
# other designs in the tree say meep, and this script only makes sense on GPU.
export RESERVOIR_SOLVER=gpumeep

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
echo "--- arch $(uname -m), partition ${SLURM_JOB_PARTITION:-?}, solver $RESERVOIR_SOLVER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU visible)"
# Provenance: which code produced this dataset. The gain/dye update path changed
# on 2026-08-04 (see the header), so a bare "gpumeep" label is not enough to
# compare datasets across machines.
for R in "$CODE" "$SIMPLESIM_PATH" "$LCRELAX_PATH" "$(dirname "$GPUMEEP_PATH")"; do
    printf -- "--- %-34s %s\n" "$(basename "$R")" \
        "$(git -C "$R" log -1 --format='%h %ad %s' --date=short 2>/dev/null || echo 'not a git checkout')"
done
if [ -n "$BATCH_SIZE" ]; then
    $PY -u "$GEN" --path "$DESIGN" "$@" --skip_existing \
        --batch "$SLURM_ARRAY_TASK_ID" --batch_size "$BATCH_SIZE"
else
    $PY -u "$GEN" --path "$DESIGN" "$@" --skip_existing --index "$SLURM_ARRAY_TASK_ID"
fi
echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
