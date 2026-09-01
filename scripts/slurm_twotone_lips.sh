#!/bin/bash
# signal_modulation/design01b_twotone — NONLINEARITY BY THE FREQUENCY METHOD.
# Two co-located CW tones inside the Nd:YAG gain line; a linear medium returns
# only the two it was given, a saturating one puts power at 2f1-f2 and 2f2-f1.
# One drive per array task, gpumeep backend, lips F5-gpu.
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# WHY A LADDER AND NOT JUST THE TOP POINT: the sidebands are the observable, and
# their power relative to the fundamentals should scale as the CUBE of drive
# while the medium is weakly nonlinear, then bend over as it saturates. One
# point tells you the medium is nonlinear; the ladder tells you where the
# nonlinearity switches on, which is the operating point a reservoir wants.
# Beat peak = 2 x per-tone amplitude, so the top rung is the agreed ceiling 80
# (user, 2026-09-01: drive 100 drove the population solver to level3 = -0.40
# during the first ring — 80 keeps every population physical).
#
# WHY 18000 t.u. AND NOT THE 3000 THE SATURATION CURVE NEEDS: the tones must sit
# inside the 159 GHz gain line (gammaE 5.3e-4 FWHM) or the second one is not
# amplified and nothing mixes; that caps their separation at df = 2.65e-4. A DFT
# resolves df only if df*T >> 1, and T = 16000 (dft 2000-18000) gives df*T = 4.2
# — the sidebands land ~4 resolution elements off the fundamentals. At T = 1000
# the whole spectrum is one line. The narrow line is what buys the long run.
#
# SUBMIT (the user submits; never srun/sbatch from the assistant):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0-5%2 scripts/slurm_twotone_lips.sh
#
#   Top point alone (beat peak 80):  sbatch --array=5 scripts/slurm_twotone_lips.sh
#
# READ OUT after:
#   python plotting/plot_twotone_spectrum.py <design> figures/twotone_spectrum.png 40
#   — Hann-windowed FFT of snap_point (NOT the raw 1Ddft bins: a rectangular
#   DFT's -13 dB sidelobes would bury a percent-level sideband one resolution
#   element away from a fundamental).
#
#SBATCH --job-name=twotone01b
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/twotone01b_%A_%a.log

set -euo pipefail

# PER-TONE amplitudes; the crystal sees the beat, so peak field = 2x these:
#   2.5  5  10  20  40  80   <- beat peaks
AMPS=(1.25 2.5 5 10 20 40)
IDX=${SLURM_ARRAY_TASK_ID:-0}
AMP=${AMPS[$IDX]}
PEAK=$(awk -v a="$AMP" 'BEGIN{printf "%g", 2*a}')

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
DESIGN=data/signal_modulation/design01b_twotone

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: opt env python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
# LCrelax FIRST on PYTHONPATH, repo ROOT not src/ — SimpleSim's lc_block imports
# `src.class_meep_blocks` from it for ANY 'reservoir' object, isotropic or not.
# Omitting this killed the first sigmod arrays with ModuleNotFoundError before
# the geometry was even built (2026-09-01).
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
export SIMPLESIM_SCRATCH_TAG="twotone${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

cd "$BASE_DIR"
echo "=== twotone01b task $IDX per-tone=$AMP beat-peak=$PEAK host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) $(date) ==="
$PY -u run.py "$DESIGN" --backend gpumeep \
    --tone-amp "$AMP" --suffix "p${PEAK}"
echo "=== task $IDX done $(date) ==="
