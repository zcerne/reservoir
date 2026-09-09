#!/usr/bin/bash -l
#SBATCH --job-name=cross_cpu
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/cross_cpu_%A_%a.log
#
# CROSS-saturation test on the lips F5 CPU partition — the MEEP (reference)
# twin of scripts/slurm_cross_gpu_lips.sh, for backend cross-validation: the
# same asymmetric-drive ladder run with RESERVOIR_SOLVER=meep. The gpumeep and
# MEEP nonlinear-residual curves should agree closely (gpumeep 2D TE is ~2% RMS
# vs MEEP with matched dt, which the shared design JSON guarantees).
#
# Outputs are suffixed _meep (summary npz cross_meep_s<strip>_L<level>.npz and
# per-task scratch tags crosscpu_*) so nothing overwrites the GPU run's files.
#
# $1 = design dir           (default data/signal_modulation/05b_ampsweep)
# $2 = swept-level ladder   (default 0,10,30,50,70,90,120)
# $3 = fixed amplitude      (default 70)
# $4 = swept strip, 1-based (default 2)
#
# SUBMIT (the user submits; the assistant never runs sbatch):
#   sbatch --array=0-6 scripts/slurm_cross_cpu_lips.sh
#   # array size must match the ladder length (7 defaults -> 0-6).
set -e

IFS=',' read -r -a LEVELS <<< "${2:-0,10,30,50,70,90,120}"
IDX=${SLURM_ARRAY_TASK_ID:-0}
LV=${LEVELS[$IDX]}
[ -n "${LV:-}" ] || { echo "no level for array idx $IDX (ladder has ${#LEVELS[@]}: 0-$(( ${#LEVELS[@]} - 1 )))"; exit 1; }

D=${1:-data/signal_modulation/05b_ampsweep}
FIXED=${3:-70}
STRIP=${4:-2}
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
export SIMPLESIM_SCRATCH_TAG="crosscpu_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

PY=${RES_PY:-/project/cerneziga/mamba_x86/envs/pmp/bin/python}
MPI=${RES_MPI:-$(dirname "$PY")/mpirun}
[ -x "$PY" ] || { echo "no x86_64 python at $PY (build: sbatch scripts/lips_build_pmp_env.sh)"; exit 1; }
[ -x "$MPI" ] || { echo "no mpirun at $MPI"; exit 1; }

cd "$CODE"
[ -d "$D" ] || { echo "missing $D -- git pull on lips first"; exit 1; }
NP=${SLURM_NTASKS:-64}
mkdir -p "$D/datasets"
echo "=== cross_cpu $D strip $STRIP level $LV (fixed $FIXED, idx $IDX) $NP ranks on $(hostname) $(date) ==="
$MPI -np "$NP" $PY -u cross_sweep.py --path "$D" --strip "$STRIP" --level "$LV" --fixed "$FIXED" \
    --out "$D/datasets/cross_meep_s${STRIP}_L${LV}.npz"
echo "=== idx $IDX (level $LV) done $(date) ==="
