#!/bin/bash
# ONE plain run.py on an Orion taurus node with MPI MEEP — the Orion
# counterpart of scripts/slurm_run_cpu_lips.sh, for the big 3D designs.
# Taurus nodes (partition `of`): 96 cores, 2 TB RAM — one node holds the
# 04_sawtooth3d_z100 cell (1.7e9 cells at res 25, ~350 GB) with headroom.
#
# QoS RULE (About servers.md / orion assoc, 2026-08-06): user cernez may
# request qos normal/soft/valhala/valhala-suspend; partition `of` needs
# `--qos=soft`. If `of` is saturated, fall back to
# `--partition=valhala --qos=valhala` (72+ cores, 190 GB — too small for the
# full-res 3D sawtooth; drop resolution to 20 first) — never xaos.
#
# SUBMIT (the user submits; from a real orion shell):
#   ssh cernez@orion-login03.of.fmf.uni-lj.si
#   cd /home/cernez/resevoir
#   sbatch scripts/slurm_orion_run3d.sh data/prism_reservoir/04_sawtooth3d_z100
#   # extra args pass through to run.py after the design path.
#
# Repo sync happens from workbox through the /home/ziga/Orion mount (git on
# the cluster hosts has no GitHub access — never put git in this script).
#
#SBATCH --job-name=res_run3d
#SBATCH --partition=of
#SBATCH --qos=soft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cernez/resevoir/slurm_run3d_%j.log

set -e
D=${1:?usage: sbatch scripts/slurm_orion_run3d.sh <design dir> [run.py args]}
shift || true
BASE_DIR=/home/cernez/resevoir
PY=/home/cernez/micromamba/envs/pmp/bin/python
MPI=/home/cernez/micromamba/envs/pmp/bin/mpirun

export SIMPLESIM_PATH=/home/cernez/SimpleSim
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
export RESERVOIR_SOLVER=meep
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "$BASE_DIR"
[ -d "$D" ] || { echo "missing $D — sync via the workbox Orion mount first"; exit 1; }
NP=${SLURM_NTASKS:-96}
echo "=== orion run3d $D ($NP ranks) on $(hostname) $(date) ==="
$MPI -np "$NP" $PY -u run.py "$D" --backend meep "$@"
echo "=== done $(date) ==="
