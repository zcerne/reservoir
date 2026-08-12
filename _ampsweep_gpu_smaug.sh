#!/usr/bin/env bash
# Amplitude sweep on a smaug 4090 (gpumeep). One process per host, whole block of
# levels in it: gpumeep compiles the FDTD once per process and reuses it for every
# level, so a single process beats several tasks each paying their own compile.
#
#   # split a 9-level ladder across both boxes
#   bash _ampsweep_gpu_smaug.sh <design> "<levels>" 0 5     # on smaug2 -> items 0-4
#   bash _ampsweep_gpu_smaug.sh <design> "<levels>" 1 5     # on smaug1 -> items 5-8
#   # then, once BOTH are done, from either host:
#   bash _ampsweep_gpu_smaug.sh <design> "<levels>" assemble
#
# --batch/--batch_size never assembles (only the serial default does), hence the
# explicit third mode. --skip_existing makes every launch resumable: parts already
# on disk are not recomputed, so a killed or re-split run costs nothing.
#
# The generator drives ONLY source_1 (first class=="source" that isn't
# "source_2"), so source_pump keeps its fixed amplitude and the inverting pulse
# fires on every sample. It also forces n_sources=1: the sweep varies drive LEVEL,
# and multi-strip probes would fold input-direction structure into what is meant
# to be a pure amplitude axis.
DESIGN=${1:?design rel path}
LEVELS=${2:?comma-separated levels}
BATCH=${3:-}
BSIZE=${4:-5}
cd "$HOME/resevoir" || exit 1
export RESERVOIR_SOLVER=gpumeep
export GPUMEEP_PATH=$HOME/GPUmeep/src
export SIMPLESIM_PATH=$HOME/SimpleSim
export PYTHONPATH=$HOME/LCrelax
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Host in the tag: /home is one NFS share, so two hosts running concurrently
# would otherwise write the same scratch monitor npz and clobber each other.
export SIMPLESIM_SCRATCH_TAG="asw_$(hostname)_$(basename "$DESIGN")"

# N2FLAM: which n2f wavelengths reach the readout. Unset keeps the historical
# single-wavelength default (0.55, the signal line) -- which contains NO pump line,
# so any pump-channel analysis then has to fall back on the monitor_2 extras.
# Set N2FLAM=all to put the whole comb in `output` itself.
COMMON=(--path "$DESIGN" --levels "$LEVELS" --n_probes 1
        --out_sensor n2f_map --components Ex,Ey,Ez
        ${N2FLAM:+--n2f_lam "$N2FLAM"}
        ${OUTNAME:+--out "$DESIGN/datasets/$OUTNAME"})

if [ "$BATCH" = "assemble" ]; then
    "$HOME"/micromamba/envs/pmp/bin/python -u \
        data_gen/generate_amplitude_sweep_data.py "${COMMON[@]}" --assemble
    exit $?
fi

LOG="ampsweep_gpu_$(basename "$DESIGN")_$(hostname).log"
nohup setsid "$HOME"/micromamba/envs/pmp/bin/python -u \
      data_gen/generate_amplitude_sweep_data.py "${COMMON[@]}" --skip_existing \
      ${BATCH:+--batch "$BATCH" --batch_size "$BSIZE"} \
      > "$LOG" 2>&1 < /dev/null &
sleep 2
echo "launched on $(hostname) -> $HOME/resevoir/$LOG"
