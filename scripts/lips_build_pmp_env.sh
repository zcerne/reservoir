#!/usr/bin/bash -l
#SBATCH --job-name=build_pmp
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --output=/project/cerneziga/reservoir_runs/build_pmp_env_%j.log
#
# Build the x86_64 MEEP environment the F5 partition needs.
#
#   sbatch scripts/lips_build_pmp_env.sh
#
# Submitted as a batch job on purpose: it has to run ON an F5 node so both the
# micromamba binary and every package resolve for the right architecture, and
# this avoids needing an interactive session at all.
#
# Why a second env exists: /project/cerneziga/micromamba/envs/opt is aarch64
# (verified from its ELF header, e_machine=0xb7) because it was built for the
# GH200 nodes on F5-gpu. f02..f07 are x86_64 AMD zen-4 and cannot run a single
# binary from it.
#
# Everything lands on /project so all six nodes see the same env.

set -e
cd /project/cerneziga

PREFIX=/project/cerneziga/mamba_x86
ENVDIR=$PREFIX/envs/pmp

echo "=== node $(hostname), arch $(uname -m), $(date)"
if [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: this is $(uname -m); the whole point is to build on x86_64."
    exit 1
fi

# --- x86_64 micromamba (the binary itself must match the node) ---
if [ ! -x ./micromamba-x86 ]; then
    echo "=== fetching x86_64 micromamba"
    URL=https://micro.mamba.pm/api/micromamba/linux-64/latest
    if command -v curl >/dev/null; then curl -Ls "$URL" | tar -xj bin/micromamba
    else wget -qO- "$URL" | tar -xj bin/micromamba; fi
    mv -f bin/micromamba ./micromamba-x86 && rmdir bin 2>/dev/null || true
    chmod +x ./micromamba-x86
fi
echo "=== micromamba: $(./micromamba-x86 --version)"

# --- the env ---
# pymeep MUST be the mpi_mpich_ build. The default is serial, and with a serial
# build `mpirun -np 64` launches 64 independent copies of the same simulation:
# it runs, writes plausible output, and is 64x slower than it appears.
export MAMBA_ROOT_PREFIX=$PREFIX
echo "=== creating $ENVDIR"
./micromamba-x86 create -y -p "$ENVDIR" -c conda-forge \
    python=3.11 "pymeep=*=mpi_mpich_*" mpich \
    numpy scipy h5py matplotlib

echo "=== jax (CPU only — no GPUs on this partition)"
"$ENVDIR/bin/pip" install --no-input "jax[cpu]"

# --- verify, because a silently-serial meep is the failure mode that hurts ---
echo "=== verification"
"$ENVDIR/bin/python" - <<'PY'
import platform
import numpy, jax, meep
print("  python   ", platform.python_version(), platform.machine())
print("  numpy    ", numpy.__version__)
print("  jax      ", jax.__version__, jax.devices())
print("  meep     ", meep.__version__)
print("  mpi ranks", meep.count_processors(), "(1 here is correct — not run under mpirun)")
PY
"$ENVDIR/bin/mpirun" --version 2>&1 | head -1

echo "=== parallel check under mpirun (should report 4 processors)"
"$ENVDIR/bin/mpirun" -np 4 "$ENVDIR/bin/python" -c \
    "import meep; print('  rank', meep.my_rank(), 'of', meep.count_processors())"

echo "=== done. Point the runner at it with:"
echo "    export RES_PY=$ENVDIR/bin/python"
echo "    export RES_MPI=$ENVDIR/bin/mpirun"
