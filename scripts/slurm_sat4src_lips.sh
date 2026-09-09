#!/usr/bin/bash -l
#SBATCH --job-name=sat4src
#SBATCH --partition=F5
#SBATCH --nodes=1
# F5 nodes: 2x32 physical cores + SMT = 128 threads, 750 GB. FDTD is memory-BW
# bound so SMT siblings buy ~nothing (slurm_lips_cpu.sh measurement); map ranks
# to the 64 PHYSICAL cores.
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/sat4src_%A_%a.log
#
# 4-SOURCE saturation ladder on the lips F5 (CPU/MEEP) partition. All 4 input
# strips driven at the SAME amplitude (single_source_sweep.py), settled gain
# read over dft 6000-10000; the 4-source saturation amplitude = the per-strip
# level where gain drops to ~half its small-signal value. ONE ARRAY TASK PER
# LEVEL so the 5 levels run in parallel across F5 nodes (wall time = one level,
# not the ~12 h the serial smaug/gpu run would have taken).
#
# SUBMIT (the user submits; the assistant never runs sbatch):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir && git pull            # get design04_4source + this script
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0-4 scripts/slurm_sat4src_lips.sh
#   squeue --me
#   # once all 5 tasks finish, merge the per-level npz into one curve (no FDTD):
#   /project/cerneziga/mamba_x86/envs/pmp/bin/python single_source_sweep.py \
#       --path data/signal_modulation/design04_4source --assemble
#   # -> datasets/single_source_sweep.npz {levels,out_norm,gain}, sorted.
#
# NEEDS the x86_64 pmp env built once on /project (aarch64 'opt' from F5-gpu
# will NOT run on these AMD nodes): sbatch scripts/lips_build_pmp_env.sh
set -e

LEVELS=(0.01 0.035 0.1 0.35 1)          # per-strip amplitudes, brackets the 0.035 estimate
IDX=${SLURM_ARRAY_TASK_ID:-0}
LV=${LEVELS[$IDX]}
[ -n "$LV" ] || { echo "no level for array idx $IDX (ladder has ${#LEVELS[@]} entries: 0-$(( ${#LEVELS[@]} - 1 )))"; exit 1; }

CODE=/home/cerneziga/resevoir
export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export LCRELAX_PATH=/home/cerneziga/LCrelax
export PYTHONPATH="$LCRELAX_PATH${PYTHONPATH:+:$PYTHONPATH}"
export RESERVOIR_SOLVER=meep
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# per-task scratch tag, or concurrent array tasks clobber each other's monitor npz
export SIMPLESIM_SCRATCH_TAG="sat4_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

PY=${RES_PY:-/project/cerneziga/mamba_x86/envs/pmp/bin/python}
MPI=${RES_MPI:-$(dirname "$PY")/mpirun}
if [ ! -x "$PY" ]; then
    echo "no x86_64 python at $PY -- build it once: sbatch scripts/lips_build_pmp_env.sh"
    exit 1
fi
if [ ! -x "$MPI" ]; then
    echo "no mpirun at $MPI -- expected beside python in the same env"; exit 1
fi

cd "$CODE"
D=data/signal_modulation/design04_4source
[ -d "$D" ] || { echo "missing $D -- git pull on lips first"; exit 1; }
NP=${SLURM_NTASKS:-64}
mkdir -p "$D/datasets"
echo "=== sat4src level $LV (array idx $IDX) $NP MPI ranks on $(hostname) $(date) ==="
$MPI -np "$NP" $PY -u single_source_sweep.py \
    --path "$D" --levels "$LV" \
    --out "$D/datasets/sat4src_${IDX}.npz"
echo "=== idx $IDX (level $LV) done $(date) ==="
