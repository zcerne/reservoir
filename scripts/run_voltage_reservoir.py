"""Top-level driver: voltage input → director → FDTD → sensor.

Stitches `class_voltage_reservoir.VoltageReservoir` (input encoding) with
`class_simulation_gpu.SimulationGPU` (FDTD). Writes the director field to
`simulation/lc_fields.npz` (the format class_simulation_gpu reads), then
invokes the existing FDTD pipeline. No refactor of class_simulation_gpu
needed — it just consumes the saved director.

Usage:
    python run_voltage_reservoir.py --path data/test_voltage
    python run_voltage_reservoir.py --path data/test_voltage --voltages 1,-1,2,-2
"""
from __future__ import annotations
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # find root core modules
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

from class_voltage_reservoir import VoltageReservoir


def write_lc_fields(folder: str | Path, vr: VoltageReservoir) -> str:
    """Save director in the lc_fields.npz format class_simulation_gpu reads.

    Required keys: phi (nx, ny, nz), x (nx,), y (ny,). theta optional.
    """
    if vr.phi is None or vr.theta is None:
        raise RuntimeError("VoltageReservoir.compute() must be called first.")
    out_dir = Path(folder) / "simulation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lc_fields.npz"
    nx, ny, nz = vr.gshape
    sx, sy, sz = vr.sizes
    x = np.linspace(-sx / 2.0, sx / 2.0, nx)
    y = np.linspace(-sy / 2.0, sy / 2.0, ny)
    z = np.linspace(-sz / 2.0, sz / 2.0, nz)
    face_dump = {f"voltages_{fn}": vr.electrodes.voltages[fn]
                 for fn in vr.electrodes.voltages}
    np.savez(out_path, phi=vr.phi, theta=vr.theta, x=x, y=y, z=z,
             voltages_flat=vr.electrodes.all_voltages_flat,
             lc_mode=vr.lc.mode,
             **face_dump)
    return str(out_path)  # pyright: ignore[reportReturnType]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="design folder")
    ap.add_argument("--voltages", type=str, default=None,
                    help="optional JSON dict override, e.g. '{\"y_max\":[1,2,3],\"x_min\":[0]}'. "
                         "Faces not listed keep the JSON-loaded values.")
    ap.add_argument("--skip-fdtd", action="store_true",
                    help="only compute director, don't run FDTD")
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp32",
                    help="FDTD precision (passed to class_simulation_gpu)")
    args = ap.parse_args()

    # Stage 1: voltage → director
    print(f"=== Stage 1: voltage_reservoir compute ===")
    vr = VoltageReservoir(args.path)
    print(vr.summary())
    v = None
    if args.voltages:
        v = json.loads(args.voltages)
        print(f"Override voltages: {v}")
    t0 = time.time()
    phi, theta = vr.compute(voltages=v)
    assert vr.E is not None
    print(f"compute: {time.time()-t0:.2f}s  phi∈[{phi.min():+.3f},{phi.max():+.3f}]  "
          f"|E|max={float(np.linalg.norm(vr.E, axis=0).max()):.3f} V/µm")

    # Stage 2: write lc_fields.npz so class_simulation_gpu can read it
    lc_path = write_lc_fields(args.path, vr)
    print(f"[run_voltage_reservoir] wrote {lc_path}")
    vr.save()  # also save full voltage_reservoir.npz for inspection

    if args.skip_fdtd:
        print("[run_voltage_reservoir] --skip-fdtd: stopping after director.")
        return 0

    # Stage 3: FDTD via existing class_simulation_gpu
    print(f"\n=== Stage 3: FDTD (class_simulation_gpu --precision {args.precision}) ===")
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "class_simulation_gpu.py"),
           "--path", args.path, "--precision", args.precision]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
