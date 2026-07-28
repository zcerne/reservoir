#!/bin/bash
# smaug2 GPU queue: 03b superposition, then capacity batches.
cd "$HOME/resevoir" || exit 1
source "$HOME/resevoir/scripts/_overnight_lib.sh"
export SIMPLESIM_SCRATCH_TAG=ov2
export JAX_PLATFORMS=cuda,cpu

banner "03b superposition"
wait_slot 3
$PY -u data_gen/generate_superposition_data.py --path $D03 \
    --n_base 8 --n_trials 40 --scale 10 \
    --out_sensor n2f_map --components Ex,Ey,Ez --skip_existing

for B in 0 1 2 3; do
  banner "03b ipc batch $B"
  wait_slot 3
  $PY -u data_gen/generate_ipc_data.py --path $D03 --n 400 --scale 10 \
      --out_sensor n2f_map --components Ex,Ey,Ez \
      --batch $B --batch_size 50 --skip_existing
done
banner "smaug2 queue DONE"
