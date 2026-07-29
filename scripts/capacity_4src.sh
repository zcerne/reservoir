#!/bin/bash
# 4-source capacity (IPC) runs. ROLE picks which slice of work this host takes.
cd "$HOME/resevoir" || exit 1
source "$HOME/resevoir/scripts/_overnight_lib.sh"
export JAX_PLATFORMS=cuda,cpu
D4LC=data/lasing_testing/04_LC_4src
D4ISO=data/lasing_testing/03b_iso_4src
ROLE=$1
case "$ROLE" in
  a)  # forward half of the LC design
      export SIMPLESIM_SCRATCH_TAG=c4a
      for B in 0 1 2 3; do
        banner "04_LC_4src ipc batch $B"; wait_slot 4
        $PY -u data_gen/generate_ipc_data.py --path $D4LC --n 400 --scale 10 \
            --out_sensor n2f_map --components Ex,Ey,Ez \
            --batch $B --batch_size 50 --skip_existing
      done ;;
  b)  # backward half of the LC design, meets (a) in the middle
      export SIMPLESIM_SCRATCH_TAG=c4b
      banner "04_LC_4src ipc reverse"; wait_slot 4
      $PY -u data_gen/generate_ipc_data.py --path $D4LC --n 400 --scale 10 \
          --out_sensor n2f_map --components Ex,Ey,Ez --reverse --skip_existing ;;
  c)  # isotropic control, forward
      export SIMPLESIM_SCRATCH_TAG=c4c
      for B in 0 1 2 3; do
        banner "03b_iso_4src ipc batch $B"; wait_slot 4
        $PY -u data_gen/generate_ipc_data.py --path $D4ISO --n 400 --scale 10 \
            --out_sensor n2f_map --components Ex,Ey,Ez \
            --batch $B --batch_size 50 --skip_existing
      done ;;
  d)  # isotropic control, backward
      export SIMPLESIM_SCRATCH_TAG=c4d
      banner "03b_iso_4src ipc reverse"; wait_slot 4
      $PY -u data_gen/generate_ipc_data.py --path $D4ISO --n 400 --scale 10 \
          --out_sensor n2f_map --components Ex,Ey,Ez --reverse --skip_existing ;;
esac
banner "capacity_4src $ROLE DONE"
