cd /home/cernez/resevoir
eval "$(~/.local/bin/micromamba shell hook -s bash)" >/dev/null 2>&1
micromamba activate pmp
export GPUMEEP_PATH=/home/cernez/GPUmeep/src JAX_PLATFORMS=cuda,cpu LADDER_SENSOR_POS=center LADDER_RUN_UNTIL=200 LADDER_SIG_LAM=0.55
for n3 in 0.0 1.0 3.0; do
  export LADDER_N3=$n3
  out=$(python ladder/ladder.py --config 3 --engine gpumeep 2>&1 | grep "gpumeep sensor")
  echo "GPU N3=$n3 :: $out"
done
