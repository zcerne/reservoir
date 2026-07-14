#!/bin/bash
cd /home/cernez/resevoir
PY=/home/cernez/micromamba/envs/pmp/bin/python
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false GPUMEEP_PATH=/home/cernez/GPUmeep/src RESERVOIR_SOLVER=gpumeep
D=data/reservoir_clasifications/17_2D_thr_resonator
$PY data_gen/generate_superposition_data.py --path $D --n_base 8 --n_trials 40 --skip_existing
$PY data_gen/generate_superposition_data.py --path $D --n_base 8 --n_trials 40 --assemble
$PY data_gen/generate_harmonics_data.py --path $D --skip_existing
$PY data_gen/generate_harmonics_data.py --path $D --assemble
$PY data_gen/generate_amplitude_sweep_data.py --path $D --skip_existing
$PY data_gen/generate_amplitude_sweep_data.py --path $D --assemble
echo CLS17-ALL-DONE
