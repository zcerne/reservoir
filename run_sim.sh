#!/bin/bash
set -e
PYTHON_LC=/home/ziga/miniforge3/envs/mp/bin/python
PYTHON_MEEP=/home/ziga/miniforge3/envs/pmp/bin/python
MPIRUN=/home/ziga/miniforge3/envs/pmp/bin/mpirun
PATH_ARG=${1:-data/test2D}
N=${2:-16}

cd "$(dirname "$0")"

echo "=== LC minimization: $PATH_ARG ==="
$PYTHON_LC class_reservoir.py --path "$PATH_ARG"

echo "=== MEEP LC run: $PATH_ARG ($N processes) ==="
$MPIRUN -np $N $PYTHON_MEEP class_simulation.py --path "$PATH_ARG" --lc-only

echo "=== MEEP empty run: $PATH_ARG ($N processes) ==="
$MPIRUN -np $N $PYTHON_MEEP class_simulation.py --path "$PATH_ARG" --empty-only
