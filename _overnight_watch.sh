#!/usr/bin/env bash
# Overnight watch: emits a line only on completion, stall, or failure.
# Silence means everything is advancing normally.
L=/home/ziga/Lips_project/reservoir_runs
O=/home/ziga/Orion/resevoir/data/lasing_testing
declare -A TGT=(
  [05b_harm]=64 [05b_bal]=625 [iso_a50]=1000 [iso_a100]=1000
  [05b_meep]=400 [smaug_a50]=400 [smaug_a100]=400)
count() {
  case $1 in
    05b_harm)  ls $L/05b_LC_patches/datasets/harmonics_a50.npz.parts 2>/dev/null | wc -l ;;
    05b_bal)   ls $L/05b_LC_patches/datasets/balance_scale_a10_50.npz.parts 2>/dev/null | wc -l ;;
    iso_a50)   ls $L/05_cav_4src/datasets/ipc_4src_a50.npz.parts 2>/dev/null | wc -l ;;
    iso_a100)  ls $L/05_cav_4src/datasets/ipc_4src_a100.npz.parts 2>/dev/null | wc -l ;;
    05b_meep)  ls $L/05b_LC_patches/datasets/ipc_4src_a100_meep.npz.parts 2>/dev/null | wc -l ;;
    smaug_a50) ls $O/05b_LC_patches/datasets/ipc_4src_a50.npz.parts 2>/dev/null | wc -l ;;
    smaug_a100)ls $O/05b_LC_patches/datasets/ipc_4src_a100.npz.parts 2>/dev/null | wc -l ;;
  esac
}
declare -A PREV STALL DONE
for k in "${!TGT[@]}"; do PREV[$k]=-1; STALL[$k]=0; DONE[$k]=0; done
while true; do
  for k in "${!TGT[@]}"; do
    [ "${DONE[$k]}" = "1" ] && continue
    n=$(count $k); t=${TGT[$k]}
    if [ "$n" -ge "$t" ]; then echo "COMPLETE $k $n/$t"; DONE[$k]=1; continue; fi
    if [ "$n" -eq "${PREV[$k]}" ]; then
      STALL[$k]=$(( ${STALL[$k]} + 1 ))
      # 6 checks x 5 min = 30 min with no new part
      [ "${STALL[$k]}" -eq 6 ] && echo "STALLED $k at $n/$t for 30min"
    else
      STALL[$k]=0
    fi
    PREV[$k]=$n
  done
  # failure signatures in the smaug worker logs (the ones I can restart)
  grep -l "GAVE UP" /home/ziga/Orion/resevoir/ipc05b_a*.log 2>/dev/null | while read f; do
    echo "FAILED $(basename $f) gave up on a block"; done
  all=1; for k in "${!TGT[@]}"; do [ "${DONE[$k]}" = "1" ] || all=0; done
  [ $all -eq 1 ] && { echo "ALL_JOBS_COMPLETE"; break; }
  sleep 300
done
