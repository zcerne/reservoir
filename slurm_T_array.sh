#!/bin/bash
# Slurm job array for T-matrix build.
# Each array task runs one basis vector (one pixel) independently on its own node.
#
# Submit:
#   sbatch --array=0-195 slurm_T_array.sh data/source_mnist
#   sbatch --array=0-3   slurm_T_array.sh data/test2D
#
# After all tasks complete, run assembly (local, no MPI):
#   python class_simulation_T.py --path data/source_mnist --assemble

#SBATCH --nodes=1
#SBATCH --partition=of
#SBATCH --qos=soft
#SBATCH --time=4:00:00
#SBATCH --mem=1900GB
#SBATCH --cpus-per-task=96
#SBATCH --output=slurm_T_%a.log

BASE_DIR=/home/cernez/resevoir
PYTHON_MEEP=/home/cernez/miniconda3/envs/pmp/bin/python
MPIRUN=/home/cernez/miniconda3/envs/pmp/bin/mpirun
PATH_ARG=${1:-data/source_mnist}

mkdir -p $BASE_DIR/$PATH_ARG/simulation_T

echo "=== Array task ${SLURM_ARRAY_TASK_ID} ==="
echo "Host: $(hostname)  Date: $(date)"
echo "Path: ${PATH_ARG}"

cd $BASE_DIR
mpirun -np $SLURM_CPUS_PER_TASK \
    python class_simulation_T.py \
    --path $PATH_ARG \
    --basis-idx $SLURM_ARRAY_TASK_ID

echo "=== Done at $(date) ==="
