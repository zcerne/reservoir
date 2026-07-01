"""Voltage-electrode boundary condition for the LC reservoir.

Geometry: 4 in-plane faces (x_min, x_max, y_min, y_max). Each face holds an
independent number of equally-spaced electrodes with user-set voltages.
Faces with no electrodes specified are Neumann (zero-flux) walls in the
Poisson solve — NO automatic ground. If the user wants a reference 0 V,
they should set some electrode to 0 V explicitly.

JSON config (reservoir block):
  "class": "voltage_reservoir",
  "sizes": [sx, sy] (+ optional sz for 3D),
  "voltages_x_min": [v1, ..., vN_left],
  "voltages_x_max": [v1, ..., vN_right],
  "voltages_y_min": [v1, ..., vN_front],
  "voltages_y_max": [v1, ..., vN_back],
  "electrode_width_um": w           # optional, default = min(pitch_x, pitch_y) / 2
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
import numpy as np


FaceName = Literal["x_min", "x_max", "y_min", "y_max"]
_FACE_NAMES: tuple[FaceName, ...] = ("x_min", "x_max", "y_min", "y_max")


class VoltageElectrodes:
    """Build per-face electrode masks + voltages from JSON `reservoir` block.

    Used by `class_poisson_2d.Poisson2D` as the Dirichlet boundary condition.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        with open(self.folder / "simulation_data.json") as f:
            d = json.load(f)
        cfg = d["reservoir"]

        if cfg.get("class") != "voltage_reservoir":
            raise ValueError(
                f"VoltageElectrodes expects reservoir.class='voltage_reservoir', "
                f"got {cfg.get('class')!r}")

        self.sizes = tuple(float(s) for s in cfg["sizes"])
        if len(self.sizes) == 2:
            self.sx, self.sy = self.sizes
            self.sz: float | None = None
        elif len(self.sizes) == 3:
            self.sx, self.sy, self.sz = self.sizes
        else:
            raise ValueError(f"reservoir.sizes must have 2 or 3 entries, got {self.sizes}")

        self.resolution = int(cfg["resolution"])

        # Per-face voltage arrays. Empty/missing → no electrodes on that face.
        self.voltages: dict[FaceName, np.ndarray] = {}
        for fn in _FACE_NAMES:
            key = f"voltages_{fn}"
            raw = cfg.get(key, [])
            self.voltages[fn] = np.asarray(raw, dtype=np.float64).flatten()

        # Backwards-compat alias: old `voltages_y_max` was called `electrode_voltages`
        # with implicit grounded `y_min`. If the new keys are all empty but the
        # legacy key is present, map it onto y_max for convenience (no implicit
        # ground — user must add `"voltages_y_min": [0,0,...]` if they want one).
        if all(v.size == 0 for v in self.voltages.values()) and "electrode_voltages" in cfg:
            self.voltages["y_max"] = np.asarray(cfg["electrode_voltages"],
                                                dtype=np.float64).flatten()

        # Default electrode width: half the smallest in-face pitch (so electrodes
        # don't touch). Faces with 0 electrodes contribute no pitch constraint.
        pitches: list[float] = []
        for fn in _FACE_NAMES:
            n = self.voltages[fn].size
            if n > 0:
                face_len = self.sx if fn.startswith("y") else self.sy
                pitches.append(face_len / n)
        default_width = (min(pitches) / 2.0) if pitches else 1.0
        self.electrode_width_um = float(cfg.get("electrode_width_um", default_width))

        # Grid (LC grid — same resolution as reservoir, matches Poisson grid).
        self.nx = int(round(self.sx * self.resolution)) + 1
        self.ny = int(round(self.sy * self.resolution)) + 1
        if self.sz is None:
            self.nz = 5
            self.dz = 4.0 / self.resolution / (self.nz - 1)
        else:
            self.nz = int(round(self.sz * self.resolution)) + 1
            self.dz = self.sz / (self.nz - 1)
        self.dx = self.sx / (self.nx - 1)
        self.dy = self.sy / (self.ny - 1)
        self.spacings = (self.dx, self.dy, self.dz)
        self.gshape = (self.nx, self.ny, self.nz)
        self.n_total = self.nx * self.ny * self.nz

    # ---------------- Dirichlet mask + values ----------------

    def build_dirichlet(self, voltages: dict[FaceName, np.ndarray] | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mask, V_dirichlet), both shape `gshape`. mask[i,j,k]=True at
        any electrode pixel. Faces with no electrodes contribute nothing
        (Neumann zero-flux via the Poisson solver's default).

        Parameters
        ----------
        voltages : dict mapping face name → voltage array. None → use the
                   instance's stored voltages.
        """
        if voltages is None:
            voltages = self.voltages
        mask = np.zeros(self.gshape, dtype=bool)
        Vdir = np.zeros(self.gshape, dtype=np.float64)
        for fn in _FACE_NAMES:
            self._paint_face(mask, Vdir, fn, voltages[fn])
        return mask, Vdir

    def _paint_face(self, mask: np.ndarray, Vdir: np.ndarray,
                    face: FaceName, vs: np.ndarray) -> None:
        """Stamp `vs.size` equally-spaced electrodes onto `face` of mask/Vdir."""
        if vs.size == 0:
            return
        n_e = vs.size
        if face.startswith("y"):
            # Electrodes span x direction
            pitch = self.sx / n_e
            j_idx = 0 if face == "y_min" else self.ny - 1
            for k in range(n_e):
                xc = -self.sx / 2.0 + (k + 0.5) * pitch
                i_lo = int(round((xc - self.electrode_width_um / 2.0 + self.sx / 2.0) / self.dx))
                i_hi = int(round((xc + self.electrode_width_um / 2.0 + self.sx / 2.0) / self.dx)) + 1
                i_lo = max(0, i_lo); i_hi = min(self.nx, i_hi)
                mask[i_lo:i_hi, j_idx, :] = True
                Vdir[i_lo:i_hi, j_idx, :] = vs[k]
        else:
            # Electrodes span y direction
            pitch = self.sy / n_e
            i_idx = 0 if face == "x_min" else self.nx - 1
            for k in range(n_e):
                yc = -self.sy / 2.0 + (k + 0.5) * pitch
                j_lo = int(round((yc - self.electrode_width_um / 2.0 + self.sy / 2.0) / self.dy))
                j_hi = int(round((yc + self.electrode_width_um / 2.0 + self.sy / 2.0) / self.dy)) + 1
                j_lo = max(0, j_lo); j_hi = min(self.ny, j_hi)
                mask[i_idx, j_lo:j_hi, :] = True
                Vdir[i_idx, j_lo:j_hi, :] = vs[k]

    # ---------------- Mutators ----------------

    def set_voltages(self, **per_face: np.ndarray) -> None:
        """Set voltages for one or more faces by name. Unspecified faces unchanged.
        Example: ve.set_voltages(y_max=[1,2,3,4], y_min=[0,0,0,0])
        """
        for fn, vs in per_face.items():
            if fn not in self.voltages:
                raise KeyError(f"unknown face {fn!r}; expected one of {_FACE_NAMES}")
            self.voltages[fn] = np.asarray(vs, dtype=np.float64).flatten()

    # ---------------- Flattened view (for convenience) ----------------

    @property
    def all_voltages_flat(self) -> np.ndarray:
        """All face voltages concatenated in x_min, x_max, y_min, y_max order.
        Useful as an "input vector" for the reservoir computer."""
        return np.concatenate([self.voltages[fn] for fn in _FACE_NAMES])

    @property
    def face_counts(self) -> dict[FaceName, int]:
        return {fn: int(self.voltages[fn].size) for fn in _FACE_NAMES}

    # ---------------- Diagnostic ----------------

    def summary(self) -> str:
        parts = []
        for fn in _FACE_NAMES:
            vs = self.voltages[fn]
            if vs.size:
                parts.append(f"{fn}=({vs.size}: {vs.tolist()})")
        face_str = "  ".join(parts) if parts else "(no electrodes)"
        return (f"VoltageElectrodes: {face_str}\n"
                f"  electrode_width = {self.electrode_width_um:.3f} µm, "
                f"gshape = {self.gshape}, "
                f"spacings = ({self.dx:.4f}, {self.dy:.4f}, {self.dz:.4f})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build voltage-electrode mask from JSON.")
    ap.add_argument("--path", required=True, help="design folder containing simulation_data.json")
    args = ap.parse_args()
    ve = VoltageElectrodes(args.path)
    print(ve.summary())
    mask, Vdir = ve.build_dirichlet()
    print(f"Dirichlet: {int(mask.sum())} pixels fixed")
    if mask.any():
        print(f"V range over fixed pixels: [{Vdir[mask].min():+.3f}, {Vdir[mask].max():+.3f}]")
