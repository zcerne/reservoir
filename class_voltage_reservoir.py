"""Top-level composer: electrodes → Poisson → LC director → optical material.

This is the new "voltage_reservoir" reservoir type used in
`class_simulation_gpu.py` when JSON has `reservoir.class = "voltage_reservoir"`.
It encapsulates the data-encoding pipeline:

    input voltages → Dirichlet → V → E → n(E) → ε(n) → FDTD material

Returns the optical material (anisotropic ε_inv tensor at Yee faces) that
`class_simulation_gpu` slots into the same place the old reservoir's
`get_dielectric_3d` provided one.

Short class. Stitches `VoltageElectrodes` + `Poisson2D` + `LCFromField`.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from class_voltage_electrodes import VoltageElectrodes
from class_voltage_electrodes_3d import VoltageElectrodes3D
from class_poisson_2d import Poisson2D
from class_lc_from_field import LCFromField


class VoltageReservoir:
    """Voltage-driven LC reservoir.

    Usage from class_simulation_gpu:
        vr = VoltageReservoir(folder)
        vr.compute(voltages)         # one call per input data point
        phi_lc_2d = vr.phi_mid_z()   # 2D φ slice at mid-z for the FDTD material
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        with open(self.folder / "simulation_data.json") as f:
            d = json.load(f)
        self.cfg = d["reservoir"]

        # Optical (FDTD) dielectrics
        self.n_o = float(self.cfg.get("n_o", 1.521))   # E7 default
        self.n_e = float(self.cfg.get("n_e", 1.746))

        # Dispatch 2D vs 3D based on sizes length.
        sizes = self.cfg["sizes"]
        self.is_3d = len(sizes) == 3 and "n_patches_per_face" in self.cfg
        # Component classes
        if self.is_3d:
            self.electrodes = VoltageElectrodes3D(folder)
        else:
            self.electrodes = VoltageElectrodes(folder)
        # Poisson2D's name is legacy — it solves on (nx, ny, nz) regardless,
        # using whichever VoltageElectrodes(_3D) you hand it.
        self.poisson = Poisson2D(self.electrodes)
        self.lc = LCFromField(folder)

        # Last-compute cache
        self.V: np.ndarray | None = None
        self.E: np.ndarray | None = None
        self.phi: np.ndarray | None = None
        self.theta: np.ndarray | None = None

    # ---------------- One-shot: voltages → director ----------------

    def compute(self, voltages: dict | None = None,
                phi_init: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Run the full pipeline. Returns (phi, theta).

        Parameters
        ----------
        voltages : optional dict {face_name: voltage_array} (face_name ∈
                   {"x_min","x_max","y_min","y_max"}). Faces not in the dict
                   keep their JSON-loaded values. None → use JSON voltages.
        phi_init : optional warm-start director.
        """
        if voltages is not None:
            self.electrodes.set_voltages(**voltages)
        phi_warm = phi_init if phi_init is not None else np.zeros(self.electrodes.gshape)
        # Poisson uses phi_warm to build ε(n̂) — for the first call this is
        # phi=0 (isotropic-ish). For subsequent inputs the user may pass the
        # previous director as warm-start.
        V, E = self.poisson.solve(phi_warm)
        phi, theta = self.lc.compute(E, self.electrodes.gshape,
                                     self.electrodes.spacings,
                                     phi_init=phi_warm,
                                     full_3d=self.is_3d)
        self.V, self.E, self.phi, self.theta = V, E, phi, theta
        return phi, theta

    # ---------------- Convenience views ----------------

    @property
    def gshape(self) -> tuple[int, int, int]:
        return self.electrodes.gshape

    @property
    def sizes(self) -> tuple[float, float, float]:
        if self.is_3d:
            return (self.electrodes.sx, self.electrodes.sy, self.electrodes.sz)
        return (self.electrodes.sx, self.electrodes.sy,
                self.electrodes.dz * (self.electrodes.nz - 1))

    def phi_mid_z(self) -> np.ndarray:
        if self.phi is None:
            raise RuntimeError("Call compute() first.")
        return self.phi[:, :, self.electrodes.nz // 2]

    # ---------------- Persistence ----------------

    def save(self, fname: str = "voltage_reservoir.npz",
             subdir: str = "simulation") -> str:
        """Save V + E + phi + theta + voltages for later inspection / plotting."""
        if self.V is None or self.E is None or self.phi is None or self.theta is None:
            raise RuntimeError("Call compute() first.")
        out_dir = self.folder / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname
        # Collect every face that this electrode set knows about (4 in 2D, 6 in 3D).
        face_dump = {f"voltages_{fn}": self.electrodes.voltages[fn]
                     for fn in self.electrodes.voltages}
        np.savez(out_path,
                 V=self.V, E=self.E, phi=self.phi, theta=self.theta,
                 voltages_flat=self.electrodes.all_voltages_flat,
                 sizes=np.asarray(self.sizes),
                 spacings=np.asarray(self.electrodes.spacings),
                 gshape=np.asarray(self.gshape),
                 n_o=self.n_o, n_e=self.n_e,
                 lc_mode=self.lc.mode,
                 **face_dump)
        return str(out_path)

    def summary(self) -> str:
        fc = self.electrodes.face_counts
        face_str = ", ".join(f"{k}={v}" for k, v in fc.items() if v > 0) or "(no electrodes)"
        return (f"VoltageReservoir(faces: {face_str}, "
                f"size {self.electrodes.sx}×{self.electrodes.sy} µm, "
                f"lc_mode={self.lc.mode!r}, n_o={self.n_o}, n_e={self.n_e})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Full voltage→director pipeline for one input.")
    ap.add_argument("--path", required=True, help="design folder")
    ap.add_argument("--voltages", type=str, default=None,
                    help="optional override as JSON dict, "
                         "e.g. '{\"y_max\":[1,2,3],\"x_min\":[0]}'. Faces not "
                         "listed keep the JSON-loaded values.")
    args = ap.parse_args()

    vr = VoltageReservoir(args.path)
    print(vr.summary())
    v = None
    if args.voltages:
        v = json.loads(args.voltages)
        print(f"override voltages: {v}")
    import time
    t0 = time.time()
    phi, theta = vr.compute(voltages=v)
    assert vr.E is not None
    print(f"compute done in {time.time()-t0:.2f}s   phi range "
          f"[{phi.min():+.3f}, {phi.max():+.3f}]   "
          f"|E|max={float(np.linalg.norm(vr.E, axis=0).max()):.3f} V/µm")
    out = vr.save()
    print(f"saved {out}")
