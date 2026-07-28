#!/bin/bash
# smaug1 GPU queue: nonlinearity datasets first, capacity as filler.
cd "$HOME/resevoir" || exit 1
source "$HOME/resevoir/scripts/_overnight_lib.sh"
export SIMPLESIM_SCRATCH_TAG=ov1
export JAX_PLATFORMS=cuda,cpu

banner "03b amp_sweep"
wait_slot 3
$PY -u data_gen/generate_amplitude_sweep_data.py --path $D03 \
    --levels 1,10,30,50,100,200,400 --n_probes 3 \
    --out_sensor n2f_map --components Ex,Ey,Ez --skip_existing

banner "waiting for chosen amplitudes"
wait_file "$AMPFILE"
read A1 A2 < <(read_amps)
banner "chosen amplitudes: $A1 $A2"

for A in "$A1" "$A2"; do
  banner "04 harmonics amp $A"
  wait_slot 3
  $PY -u data_gen/generate_harmonics_data.py --path $D04 \
      --out $D04/datasets/harmonics_amp${A}.npz \
      --tones 3,5 --channels 0,1 --n_t 64 --amps ${A},${A} \
      --out_sensor n2f_map --components Ex,Ey,Ez --skip_existing
done

banner "03b harmonics amp 10 (baseline, matches 04)"
wait_slot 3
$PY -u data_gen/generate_harmonics_data.py --path $D03 \
    --tones 3,5 --channels 0,1 --n_t 64 --amps 10,10 \
    --out_sensor n2f_map --components Ex,Ey,Ez --skip_existing

for A in "$A1" "$A2"; do
  banner "03b harmonics amp $A"
  wait_slot 3
  $PY -u data_gen/generate_harmonics_data.py --path $D03 \
      --out $D03/datasets/harmonics_amp${A}.npz \
      --tones 3,5 --channels 0,1 --n_t 64 --amps ${A},${A} \
      --out_sensor n2f_map --components Ex,Ey,Ez --skip_existing
done

banner "03b ipc — reverse filler (covers whatever the batch workers missed)"
wait_slot 3
$PY -u data_gen/generate_ipc_data.py --path $D03 --n 400 --scale 10 \
    --out_sensor n2f_map --components Ex,Ey,Ez --reverse --skip_existing
banner "smaug1 queue DONE"
