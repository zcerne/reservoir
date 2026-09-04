#!/bin/bash
# signal_modulation/design03b_strips — SINGLE-STRIP SPREAD PROBE on lips F5-gpu.
# The design01d 'onestrip' run repeated at real doping and fabricable scale:
# drive strip A only, 2Ddft the whole cell, look at how far it spreads.
#
# PARTITION RULE (user, 2026-08-24): only F5 and F5-gpu are ever used on lips.
#
# Only source_1 is driven — source_2 sits at amplitude 0 in the JSON, so no
# override is needed to make this a single-strip run.
#
# THREE AMPLITUDES, because the spread is not guaranteed to be drive-independent.
# At low drive the map is pure diffraction plus guiding; at the working point the
# gain saturates where the strip is brightest, which can gain-guide or flatten the
# profile. design01d never tested this (it ran one amplitude), so it is worth the
# two extra tasks to know whether the spread we design around is the spread we get:
#   task 0  amp   3   small-signal reference, essentially linear optics
#   task 1  amp  55   near the low edge of the predicted saturation knee
#   task 2  amp 219   the working point (design01d's 40 x sqrt(30))
# If all three maps agree, the spread is a geometric fact and the 64-pair sweep
# can use any drive. If they differ, the overlap fraction becomes drive-dependent
# and the sweep has to be read at fixed drive.
#
# WHAT TO LOOK AT: plotting/plot_strip_spread.py prints I(other strip)/I(own
# strip) at five stations. design01d gave 0.002 / 0.150 / 0.575 / 0.478 / 0.175
# (entrance -> exit). If the sqrt(30) scaling is right, design03b should land
# CLOSE TO THOSE FIVE NUMBERS — that is the actual test, not the picture. Same
# L/L_mix = 1.49, same footprint/guide ratio 2.19, so the same overlap curve
# should follow. A mid-crystal value far from ~0.57 means the scaling argument
# is wrong somewhere.
#
# COST / SIZE. 606 x 32 um at resolution 20 = 7.8 M cells, 400k steps
# = 3.1e12 cell-steps per task — the same as one design03_realscale amp task,
# because this design deliberately shares its run_until 10000 / window
# 6000-10000 rather than guessing a shorter one (see the JSON comment: nothing
# has measured the settling time yet, and a field_map DFT over an unsettled
# window smears silently instead of failing).
# The field_map npz is ~370 MB PER TASK (53x design01d's grid, Ex/Ey/Ez complex),
# so three tasks write ~1.1 GB into simulation_gpumeep/. Check quota first.
#
# SUBMIT (the user submits; the assistant never runs srun/sbatch):
#   ssh -J cerneziga@f1login.ijs.si cerneziga@lips
#   cd /home/cerneziga/resevoir
#   git pull
#   mkdir -p /project/cerneziga/reservoir_runs/logs
#   sbatch --array=0 scripts/slurm_sigmod03b_spread_lips.sh       # one first
#   sbatch --array=1-2 scripts/slurm_sigmod03b_spread_lips.sh     # then the rest
#
#SBATCH --job-name=sigmod03b_spread
#SBATCH --partition=F5-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/sigmod03b_%A_%a.log

set -euo pipefail

AMPS=(3 55 219)
IDX=${SLURM_ARRAY_TASK_ID:-0}
AMP=${AMPS[$IDX]}
TAG="onestrip_a${AMP}"

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/micromamba/envs/opt/bin/python
DESIGN=data/signal_modulation/design03b_strips

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
export SIMPLESIM_SCRATCH_TAG="sigmod03b_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"

cd "$BASE_DIR"
echo "=== sigmod03b spread task $IDX amp=$AMP host $(hostname) gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) $(date) ==="
$PY -u run.py "$DESIGN" --backend gpumeep --signal-amp "$AMP" --suffix "$TAG"

# Figure + the five overlap numbers. Non-fatal: the npz is the deliverable and
# can always be replotted on the workbox from the sshfs mount.
echo "--- spread figure ---"
$PY -u plotting/plot_strip_spread.py "$DESIGN" \
    "$DESIGN/figures/strip_spread_a${AMP}.png" "$TAG" || \
    echo "WARNING: plotting failed, npz is on disk — replot on the workbox"

echo "=== task $IDX done $(date) ==="
