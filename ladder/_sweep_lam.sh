#!/bin/bash
# Wavelength sweep: all ladder configs x {0.45, 0.50, 0.55, 0.60}, both engines.
# Copies each run's sensors to _plots/data/ for the summary plot.
set -u
cd /home/cernez/resevoir/ladder
PY=/home/cernez/micromamba/envs/pmp/bin/python
export JAX_PLATFORMS=cuda,cpu
export GPUMEEP_PATH=/home/cernez/GPUmeep/src
mkdir -p _plots/data

for lam in 0.45 0.50 0.55 0.60; do
  for c in 1 2 3 4 5 6; do
    d=$(ls -d ../data/ladder/config_${c}_* 2>/dev/null | head -1)
    echo "=== cfg $c lam $lam ==="
    LADDER_SIG_LAM=$lam $PY ladder.py --config $c --engine both > /dev/null 2>&1
    d=$(ls -d ../data/ladder/config_${c}_* | head -1)
    cp "$d/simulation/monitor_2_meep.npz"    "_plots/data/cfg${c}_lam${lam}_meep.npz"
    cp "$d/simulation/monitor_2_gpumeep.npz" "_plots/data/cfg${c}_lam${lam}_gpumeep.npz"
  done
done
echo SWEEP-DONE
