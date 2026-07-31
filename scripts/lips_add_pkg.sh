#!/usr/bin/bash -l
#SBATCH --job-name=add_pkg
#SBATCH --partition=F5
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/project/cerneziga/reservoir_runs/add_pkg_%j.log
#
# Add packages to the F5 pmp env. Has to run as a job because the env is
# x86_64 and only the F5 nodes are — the login node cannot resolve or link
# packages for it.
#
#   sbatch scripts/lips_add_pkg.sh nlopt
#   sbatch scripts/lips_add_pkg.sh nlopt some-other-package
#
# Finishes by importing the whole stack, so a missing transitive dependency
# shows up here rather than four minutes into an array job.
set -e

PREFIX=/project/cerneziga/mamba_x86
ENVDIR=$PREFIX/envs/pmp
MM=/project/cerneziga/micromamba-x86
export MAMBA_ROOT_PREFIX=$PREFIX

[ $# -ge 1 ] || { echo "usage: sbatch scripts/lips_add_pkg.sh <package> [...]"; exit 1; }

echo "=== node $(hostname), arch $(uname -m), $(date)"
[ "$(uname -m)" = "x86_64" ] || { echo "ERROR: not x86_64"; exit 1; }
[ -x "$MM" ] || { echo "ERROR: no micromamba at $MM"; exit 1; }

echo "=== installing: $*"
"$MM" install -y -p "$ENVDIR" -c conda-forge "$@"

echo "=== import check (the real proof)"
"$ENVDIR/bin/python" - <<'PY'
mods = ["numpy", "scipy", "matplotlib", "h5py", "jax", "meep", "nlopt"]
bad = []
for m in mods:
    try:
        __import__(m)
        print(f"  ok      {m}")
    except Exception as e:
        bad.append(m)
        print(f"  MISSING {m}: {type(e).__name__}: {e}")
raise SystemExit(1 if bad else 0)
PY
echo "=== all imports fine — resubmit the run"
