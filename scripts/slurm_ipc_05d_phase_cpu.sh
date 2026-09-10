#!/usr/bin/bash -l
#SBATCH --job-name=ipc05dphase
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --output=/project/cerneziga/reservoir_runs/logs/ipc05dphase_%A_%a.log
#
# IPC dataset #3 for 05d — PHASE encoding at constant max amplitude.
# u -> scale * e^{i pi u / 2}: every strip always at magnitude `scale`, the
# information rides on a continuous phase in [-pi/2, +pi/2] (half-turn, so the
# map is injective). Same default seed -> same 1000 u-vectors as the scale-70
# and scale-110 amplitude datasets: three encodings of identical probes.
#
# WHY CPU / MEEP: gpumeep CW sources are sin(wt)*amp with real amp — a complex
# amplitude's phase would be dropped ( _gen_common now refuses ). MEEP applies
# complex CW amplitudes exactly (verified 2026-09-10: relative-phase-pi sources
# cancel to 6e-17). PARTITION RULE: only F5 / F5-gpu, never grace.
#
# CONSTANT DRIVE = worst case ALWAYS: all four strips at full magnitude every
# probe — the same all-at-max validation rung gates this dataset:
#   sbatch --array=0-0 scripts/slurm_sat4src_gpu_lips.sh \
#       data/signal_modulation/05d_ampsweep 110
#
# Batched: 40 tasks x 25 samples (one 64-rank MEEP run per sample, serially
# inside the task). SUBMIT (the user submits; never sbatch from the assistant):
#   cd /home/cerneziga/resevoir
#   sbatch --array=0-39 scripts/slurm_ipc_05d_phase_cpu.sh
#
# ASSEMBLE when done (login node, x86 pmp env):
#   /project/cerneziga/mamba_x86/envs/pmp/bin/python \
#       data_gen/generate_ipc_data.py \
#       --path data/signal_modulation/05d_ampsweep --n 1000 --scale 110 \
#       --encode phase --out_sensor monitor_2 \
#       --components Ex,Ey,Ez \
#       --out data/signal_modulation/05d_ampsweep/datasets/ipc_phase110.npz \
#       --assemble
set -euo pipefail

BASE_DIR=${BASE_DIR:-/home/cerneziga/resevoir}
PY=/project/cerneziga/mamba_x86/envs/pmp/bin/python
MPI=$(dirname "$PY")/mpirun
DESIGN=data/signal_modulation/05d_ampsweep
OUT=$DESIGN/datasets/ipc_phase110.npz
BATCH_SIZE=25

[ -d "$BASE_DIR/$DESIGN" ] || { echo "ERROR: $BASE_DIR/$DESIGN missing — pull the repo via the workbox sshfs mount first"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: x86 pmp python missing at $PY"; exit 1; }

export SIMPLESIM_PATH=/home/cerneziga/SimpleSim
export GPUMEEP_PATH=/home/cerneziga/GPUmeep/src
export RESERVOIR_SOLVER=meep
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Per-task scratch tag: concurrent tasks share simulation_meep/; without a
# unique tag they clobber each other's monitor npz and silently swap results.
export SIMPLESIM_SCRATCH_TAG="ipcph_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

cd "$BASE_DIR"
echo "=== ipc 05d phase-encode batch ${SLURM_ARRAY_TASK_ID:-?} (x$BATCH_SIZE) host $(hostname) tag $SIMPLESIM_SCRATCH_TAG $(date) ==="
$MPI -np "${SLURM_NTASKS:-64}" $PY -u data_gen/generate_ipc_data.py \
    --path "$DESIGN" \
    --n 1000 --scale 110 --encode phase \
    --out_sensor monitor_2 --components Ex,Ey,Ez \
    --out "$OUT" \
    --skip_existing --batch "${SLURM_ARRAY_TASK_ID}" --batch_size "$BATCH_SIZE"
echo "=== batch ${SLURM_ARRAY_TASK_ID:-?} done $(date) ==="
