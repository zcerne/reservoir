#!/bin/bash
# Resume the design-04 jobs that were launched before the no-preallocate rule:
# the ipc reverse filler (smaug1) / ipc batches (smaug2), plus the amp_sweep
# that OOMed at 16/21 when those hogs left no headroom.
cd "$HOME/resevoir" || exit 1
source "$HOME/resevoir/scripts/_overnight_lib.sh"
export JAX_PLATFORMS=cuda,cpu
ROLE=$1
case "$ROLE" in
  smaug1)
    export SIMPLESIM_SCRATCH_TAG=r04a
    banner "04 amp_sweep (finish remaining parts)"
    wait_slot 4
    $PY -u data_gen/generate_amplitude_sweep_data.py --path $D04 \
        --levels 1,10,30,50,100,200,400 --n_probes 3 \
        --out_sensor n2f_map --components Ex,Ey,Ez --skip_existing
    banner "04 ipc reverse filler"
    wait_slot 4
    $PY -u data_gen/generate_ipc_data.py --path $D04 --n 400 --scale 10 \
        --out_sensor n2f_map --components Ex,Ey,Ez --reverse --skip_existing
    ;;
  smaug2)
    export SIMPLESIM_SCRATCH_TAG=r04b
    for B in 0 1 2 3 4; do
      banner "04 ipc batch $B"
      wait_slot 4
      $PY -u data_gen/generate_ipc_data.py --path $D04 --n 400 --scale 10 \
          --out_sensor n2f_map --components Ex,Ey,Ez \
          --batch $B --batch_size 50 --skip_existing
    done
    ;;
esac
banner "resume_04 $ROLE DONE"
