#!/usr/bin/env bash
# design03c_sweep64 — 64-pair CROSS sweep on the real-scale strips, smaug GPUs.
# Same protocol as design01c/d (tones 3,5 one per strip, DFT over the sweep
# index); amps 20,20 = the design03-calibrated knee. ~37 min/run without the
# snap/field_map sensors, so 8 blocks of 8 split across both boxes is ~20 h.
#
#   bash scripts/_sweep64_03c_smaug.sh 0 3      # on smaug2: blocks 0-3
#   bash scripts/_sweep64_03c_smaug.sh 4 7      # on smaug1: blocks 4-7
#   bash scripts/_sweep64_03c_smaug.sh assemble # once, after both are done
FIRST=${1:?first block, or 'assemble'}
LAST=${2:-$FIRST}
BLOCK=8
MAXTRY=2
cd "$HOME/resevoir" || exit 1
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_compile
export JAX_PLATFORMS=cuda,cpu RESERVOIR_SOLVER=gpumeep
export GPUMEEP_PATH=$HOME/GPUmeep/src
export PYTHONPATH=$HOME/LCrelax
export SIMPLESIM_PATH=$HOME/SimpleSim
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
D=data/signal_modulation/design03c_sweep64
COMMON=(--path "$D" --tones 3,5 --channels 0,1 --n_t 64 --amps 20,20)

if [ "$FIRST" = "assemble" ]; then
    "$HOME"/micromamba/envs/pmp/bin/python -u \
        data_gen/generate_harmonics_data.py "${COMMON[@]}" --assemble
    exit $?
fi

nohup setsid bash -c '
  t0=$(date +%s); fails=0
  for (( k='"$FIRST"'; k<='"$LAST"'; k++ )); do
    ok=0
    for try in $(seq 1 '"$MAXTRY"'); do
      SIMPLESIM_SCRATCH_TAG="s64c_$(hostname)_$k" '"$HOME"'/micromamba/envs/pmp/bin/python -u \
        data_gen/generate_harmonics_data.py --path '"$D"' \
        --tones 3,5 --channels 0,1 --n_t 64 --amps 20,20 \
        --skip_existing --batch $k --batch_size '"$BLOCK"' && { ok=1; break; }
      echo "=== block $k attempt $try FAILED"; sleep 30
    done
    [ $ok -eq 1 ] || { echo "=== block $k GAVE UP"; fails=$(( fails + 1 )); }
  done
  echo SWEEP64_03C_EXIT=$fails WALL=$(( $(date +%s) - t0 ))s
' > "sweep64_03c_$(hostname)_${FIRST}.log" 2>&1 < /dev/null &
sleep 2
echo "design03c sweep blocks $FIRST-$LAST launched on $(hostname)"
