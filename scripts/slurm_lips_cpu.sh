#!/usr/bin/bash -l
#SBATCH --job-name=res_cpu
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/slurm_cpu_%A_%a.log
#
# MEEP (CPU) data generation on the lips F5 partition — the big idle AMD nodes,
# NOT the GPU partition. One node per array task, one MPI-parallel FDTD at a
# time on it, a contiguous block of samples per task.
#
#   sbatch --array=0-7%6 --export=ALL,BATCH_SIZE=50 scripts/slurm_lips_cpu.sh \
#       ipc /project/cerneziga/reservoir_runs/05_cav_4src \
#       --n 400 --scale 10 --out_sensor n2f_map --components Ex,Ey,Ez
#
#   method = superposition | harmonics | ampsweep | ipc | balance
#
# ---------------------------------------------------------------- the hardware
# f02..f07, six nodes, from scontrol 2026-07-31:
#   CPUTot=128 but Sockets=2 x CoresPerSocket=32 x ThreadsPerCore=2
#   -> 64 PHYSICAL cores, 128 hardware threads
#   RealMemory=750000 (750 GB), Gres=(null)  -> no GPUs on this partition
# So this partition is for MEEP, never for gpumeep. The GPU work lives on
# F5-gpu (h01/h02, GH200) — different nodes entirely; do not confuse the two.
#
# 64 tasks, not 128: FDTD is memory-bandwidth bound, and SMT siblings share a
# core's load/store path, so the second thread per core buys almost nothing and
# often costs. Measured on smaug (same solver, smaller box): 7 ranks 0.0338
# s/step, 14 ranks 0.0153 — good scaling while ranks map to real cores.
#
# ------------------------------------------------------------------ must-haves
# JAX_PLATFORMS=cpu is NOT optional. Every rank imports jax (LC relax); on a
# GPU-less node an unset JAX_PLATFORMS still probes CUDA and each of the 64
# ranks logs an initialisation failure before falling back.
#
# One thread per rank. JAX-CPU sizes its intra-op pool to the FULL core count,
# so 64 ranks x 64 threads oversubscribes the node ~64x and each evaluation
# slows by an order of magnitude (measured 2026-07-15: 5 s -> 66-110 s/eval).
#
# MPI MEEP needs the fix in data_gen/_gen_common.py (commit 991386d): MEEP
# writes monitor/near2far output from the master rank only, so forward() must
# return None on the others. Without it every non-master rank raises
# FileNotFoundError, some die, the survivors block in the next collective and
# the job deadlocks — one item in 2h44m with 16 ranks pinned at 100%.

set -e

CODE=/home/cerneziga/resevoir
export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export LCRELAX_PATH=/home/cerneziga/LCrelax
export PYTHONPATH="$LCRELAX_PATH:$PYTHONPATH"

export RESERVOIR_SOLVER=meep
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# per-task scratch, or concurrent array tasks overwrite each other's monitor npz
export SIMPLESIM_SCRATCH_TAG="cpu${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

# !! THIS PARTITION NEEDS ITS OWN ENV !!
# /project/cerneziga/micromamba/envs/opt is an **aarch64** build (verified from
# its ELF header, e_machine=0xb7) because it was made for the GH200 nodes on
# F5-gpu. f02..f07 are x86_64 AMD zen-4, so that env cannot execute here at all
# -- not a version mismatch, a different instruction set.
#
# Build an x86_64 env once, on /project so every node sees it:
#
#   sbatch scripts/lips_build_pmp_env.sh
#
# That is a batch job, not an interactive session: it must run ON an F5 node so
# the micromamba binary and every package resolve for x86_64, and it verifies
# the result under mpirun before declaring success.
# Then either export RES_PY to its python, or edit the default below.
PY=${RES_PY:-/project/cerneziga/mamba_x86/envs/pmp/bin/python}
MPI=${RES_MPI:-mpirun}
if [ ! -x "$PY" ]; then
    echo "no x86_64 python at $PY -- see the env-build notes at the top of this"
    echo "script; the aarch64 'opt' env from F5-gpu will not run on this node."
    exit 1
fi

METHOD=${1:?usage: sbatch --array=0-(N-1) scripts/slurm_lips_cpu.sh <method> <design_dir> [args]}
DESIGN=${2:?usage: ... <method> <design_dir> [args]}
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
NP=${SLURM_NTASKS:-64}
echo "=== $METHOD $DESIGN task ${SLURM_ARRAY_TASK_ID} on $(hostname), $NP MPI ranks, $(date) ==="

if [ -n "$BATCH_SIZE" ]; then
    $MPI -np "$NP" $PY -u "$GEN" --path "$DESIGN" "$@" --skip_existing \
        --batch "$SLURM_ARRAY_TASK_ID" --batch_size "$BATCH_SIZE"
else
    $MPI -np "$NP" $PY -u "$GEN" --path "$DESIGN" "$@" --skip_existing \
        --index "$SLURM_ARRAY_TASK_ID"
fi
echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
