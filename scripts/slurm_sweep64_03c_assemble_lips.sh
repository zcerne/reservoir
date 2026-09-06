#!/bin/bash
# design03c_sweep64 — assemble the 64 parts into harmonics.npz. CPU only (just
# concatenates npz), so F5, no GPU. Submit with a dependency on the array:
#   sbatch --dependency=afterok:<ARRAY_JOBID> scripts/slurm_sweep64_03c_assemble_lips.sh
#SBATCH --job-name=s64c_assemble
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/sweep64_03c_assemble_%j.log
set -euo pipefail
BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export PYTHONPATH=/home/cerneziga/LCrelax${PYTHONPATH:+:$PYTHONPATH}
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
cd "$BASE_DIR"
echo "=== assemble $(date) ==="
$PY -u data_gen/generate_harmonics_data.py \
    --path data/signal_modulation/design03c_sweep64 \
    --tones 3,5 --channels 0,1 --n_t 64 --amps 20,20 --assemble
echo "=== assemble done $(date) ==="
