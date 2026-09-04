#!/usr/bin/env bash
# design03c_sweep64 — 64-pair CROSS sweep on the real-scale strips, smaug 4090s.
# ONE SAMPLE PER PROCESS (batch_size 1): the real-scale grid (12120x640 fp64)
# is marginal in a 4090's 24 GB, and running a whole block in one python
# process fragments/accumulates VRAM until a mid-run allocation OOMs (measured
# 2026-09-04: 8-per-process died after ~5 h with RESOURCE_EXHAUSTED). A fresh
# process per sample gets the whole clean 24 GB every time; the on-disk JAX
# compile cache makes process 2+ skip compilation. Fully resumable.
#
#   bash scripts/_sweep64_03c_smaug.sh 0 31      # smaug2: samples 0-31
#   bash scripts/_sweep64_03c_smaug.sh 32 63     # smaug1: samples 32-63
#   bash scripts/_sweep64_03c_smaug.sh assemble  # once both are done
FIRST=${1:?first sample index, or 'assemble'}
LAST=${2:-$FIRST}
cd "$HOME/resevoir" || exit 1
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_compile
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PLATFORMS=cuda,cpu RESERVOIR_SOLVER=gpumeep
export GPUMEEP_PATH=$HOME/GPUmeep/src
export PYTHONPATH=$HOME/LCrelax
export SIMPLESIM_PATH=$HOME/SimpleSim
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
D=data/signal_modulation/design03c_sweep64

if [ "$FIRST" = "assemble" ]; then
    "$HOME"/micromamba/envs/pmp/bin/python -u \
        data_gen/generate_harmonics_data.py --path "$D" \
        --tones 3,5 --channels 0,1 --n_t 64 --amps 20,20 --assemble
    exit $?
fi

nohup setsid bash -c '
  t0=$(date +%s); fails=0
  for (( j='"$FIRST"'; j<='"$LAST"'; j++ )); do
    ok=0
    for try in 1 2; do
      SIMPLESIM_SCRATCH_TAG="s64c_$(hostname)_$j" '"$HOME"'/micromamba/envs/pmp/bin/python -u \
        data_gen/generate_harmonics_data.py --path '"$D"' \
        --tones 3,5 --channels 0,1 --n_t 64 --amps 20,20 \
        --skip_existing --batch $j --batch_size 1 && { ok=1; break; }
      echo "=== sample $j attempt $try FAILED"; sleep 20
    done
    [ $ok -eq 1 ] || { echo "=== sample $j GAVE UP"; fails=$(( fails + 1 )); }
  done
  echo SWEEP64_03C_EXIT=$fails WALL=$(( $(date +%s) - t0 ))s
' > "sweep64_03c_$(hostname)_${FIRST}.log" 2>&1 < /dev/null &
sleep 2
echo "design03c sweep samples $FIRST-$LAST launched on $(hostname)"
