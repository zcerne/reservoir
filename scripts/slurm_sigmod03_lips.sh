#!/bin/bash
# signal_modulation/design03_realscale — Nd:YAG at REAL 1 at.% doping, 600 um
# crystal. PILOT AMPLITUDE SWEEP on lips F5-gpu, one amplitude per array task.
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# WHY THIS LADDER. Stimulated rate goes as SGMA*I. SGMA is 30x smaller than
# design01's, so the same saturation needs 30x the intensity = sqrt(30) = 5.48x
# the FIELD amplitude. design01's knee sat between amp 10 and 100 with a ceiling
# near 80 (amp 100 drove populations negative), so the predicted knee here is
# 55-550 and the predicted ceiling ~440. The ladder brackets both, ascending, so
# the risky high-amplitude tasks run LAST and a ceiling blow-up costs the least
# GPU time. amp 3 anchors the small-signal gain, which should come out at
# exp(2.32) = 10.2x if the rescaling is right — that is the first number to check.
#
# WHY 10000 t.u. Transit is now 602*1.82 = 1096 t.u. (design01: 36) and the burn
# is 30x slower per unit intensity. The DFT window opens at 6000. dft_t_end is
# set EQUAL to run_until in the JSON on purpose: an explicit dft_t_end below
# run_until silently terminates the run there.
#
# COST WARNING, read before submitting the whole array. The cell is 606 x 32 um
# at resolution 20 = 7.8 M cells x 400k steps = 3.1e12 cell-steps, roughly 70x
# design01's pilot. If design01's pilot took T, budget ~70T per task. SUBMIT TASK
# 0 ALONE FIRST and read the wall time off it:
#     sbatch --array=0 scripts/slurm_sigmod03_lips.sh
# If that is over ~4 h, the two dials, in order of preference:
#   resolution 20 -> 15 in the JSON  (2.4x cheaper, 8.8 px per lam/n, still sane)
#   crystal 600 -> 300 um AND SGMA 6.2e-7 -> 1.24e-6  (2x cheaper, keeps gL 2.32,
#      but the doping is then 2x real, so say so in the writeup)
# Do NOT cut run_until to save time until snap_point proves the envelope has
# flattened before 6000.
#
# SUBMIT (the user submits; the assistant never runs srun/sbatch):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir
#   git pull                       # design03_realscale/simulation_data.json is tracked
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0 scripts/slurm_sigmod03_lips.sh          # timing probe first
#   sbatch --array=1-8%2 scripts/slurm_sigmod03_lips.sh      # then the rest
#
# READ OUT: gain = monitor_2/monitor_1 at 1.064 per amplitude (small-signal
# should be ~10.2x); settling time and any ringing from snap_point; N3 burn
# fraction from pop_monitor. The knee location sets the drive for design03b.
#
#SBATCH --job-name=signalmod03
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/sigmod03_%A_%a.log

set -euo pipefail

AMPS=(3 10 30 55 100 175 300 450 700)
IDX=${SLURM_ARRAY_TASK_ID:-0}
AMP=${AMPS[$IDX]}

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
DESIGN=data/signal_modulation/design03_realscale

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: opt env python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
# LCrelax FIRST on PYTHONPATH: SimpleSim's lc_block imports
# `src.class_meep_blocks` from it for ANY 'reservoir' object, isotropic or not.
# The path must be the repo ROOT (which contains src/), not src/.
export PYTHONPATH=/home/cerneziga/LCrelax${PYTHONPATH:+:$PYTHONPATH}
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/project/cerneziga/jax_compile_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export PYTHONUNBUFFERED=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Per-task scratch tag: concurrent tasks otherwise clobber each other's monitor
# npz in the shared simulation_gpumeep/ between run() and read-back.
export SIMPLESIM_SCRATCH_TAG="sigmod03_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

cd "$BASE_DIR"
echo "=== sigmod03 task $IDX amp=$AMP host $(hostname) gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) $(date) ==="
$PY -u run.py "$DESIGN" --backend gpumeep \
    --signal-amp "$AMP" --suffix "a${AMP}"
echo "=== task $IDX done $(date) ==="
