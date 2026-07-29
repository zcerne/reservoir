#!/bin/bash
# Balance Scale task set on 04_LC_4src (625 samples, whole far-field map saved).
# ROLE a/b = forward batch worker / reverse filler; they meet in the middle and
# --skip_existing keeps the overlap free.
cd "$HOME/resevoir" || exit 1
source "$HOME/resevoir/scripts/_overnight_lib.sh"
export JAX_PLATFORMS=cuda,cpu
D=data/lasing_testing/${2:-04_LC_4src}   # $2 = design dir name
COMMON="--path $D --out_sensor n2f_map --components Ex,Ey,Ez --skip_existing"
case "$1" in
  a) export SIMPLESIM_SCRATCH_TAG=bsa_${2:-04}
     for B in $(seq 0 6); do
       banner "balance scale ${2:-04_LC_4src} batch $B"; wait_slot 4
       $PY -u data_gen/generate_balance_scale_data.py $COMMON --batch $B --batch_size 50
     done ;;
  b) export SIMPLESIM_SCRATCH_TAG=bsb_${2:-04}
     banner "balance scale ${2:-04_LC_4src} reverse filler"; wait_slot 4
     $PY -u data_gen/generate_balance_scale_data.py $COMMON --reverse ;;
esac
banner "balance_scale_run $1 DONE"
