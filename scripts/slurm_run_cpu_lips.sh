#!/usr/bin/bash -l
#SBATCH --job-name=res_run_cpu
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/run_cpu_%j.log
#
# ONE plain run.py on the lips F5 CPU partition with MPI MEEP — the CPU
# counterpart of scripts/slurm_lips_run.sh (which is F5-gpu/gpumeep). For 3D
# designs that need native MEEP features (3D near2far) or exceed GPU memory.
#
#   sbatch scripts/slurm_run_cpu_lips.sh data/prism_reservoir/3d_cyl_lens_test
#   # extra args pass through to run.py after the design path.
set -e
D=${1:?usage: sbatch scripts/slurm_run_cpu_lips.sh <design dir> [run.py args]}
shift || true
CODE=/home/cerneziga/resevoir
export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export LCRELAX_PATH=/home/cerneziga/LCrelax
export PYTHONPATH="$LCRELAX_PATH${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY=${RES_PY:-/project/cerneziga/mamba_x86/envs/pmp/bin/python}
MPI=${RES_MPI:-$(dirname "$PY")/mpirun}
[ -x "$PY" ] || { echo "no x86_64 pmp python at $PY"; exit 1; }
cd "$CODE"; [ -d "$D" ] || { echo "missing $D — git pull on lips first"; exit 1; }
NP=${SLURM_NTASKS:-64}
echo "=== run_cpu $D ($NP ranks) on $(hostname) $(date) ==="
$MPI -np "$NP" $PY -u run.py "$D" --backend meep "$@"
echo "=== done $(date) ==="
