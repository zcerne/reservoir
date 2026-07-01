"""3D voltage-electrode boundary condition for the LC reservoir.

Six cube faces, each holding a 2D patch grid of electrodes. Used for the 3D
cube reservoir geometry (see `data/3_3D_cube_mnist/`).

JSON config (reservoir block):
  "class": "voltage_reservoir",
  "sizes": [sx, sy, sz],
  "n_patches_per_face": [m, n],          # 2D grid per face; e.g. [7, 7]
  "voltages_x_min": [v0, ..., v_{m*n-1}], # row-major scan in face coords
  "voltages_x_max": [...],
  "voltages_y_min": [...],
  "voltages_y_max": [...],
  "voltages_z_min": [...],
  "voltages_z_max": [...],
  "electrode_width_um": w                 # patch-pad inset (default 0.5 um)

Face → in-face axis convention:
  x_min / x_max : face spans (y, z); m → y-direction, n → z-direction
  y_min / y_max : face spans (x, z); m → x-direction, n → z-direction
  z_min / z_max : face spans (x, y); m → x-direction, n → y-direction

Voltage k = m_i * n_per_face[1] + n_j  (row-major).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
import numpy as np


FaceName3D = Literal["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
_FACE_NAMES_3D: tuple[FaceName3D, ...] = (
    "x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


class VoltageElectrodes3D:
    """6-face cube voltage electrode boundary condition. Each face holds an
    m × n grid of patches; patches tile the full face (no gaps).
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        with open(self.folder / "simulation_data.json") as f:
            d = json.load(f)
        cfg = d["reservoir"]
        if cfg.get("class") != "voltage_reservoir":
            raise ValueError(
                f"VoltageElectrodes3D expects reservoir.class='voltage_reservoir', "
                f"got {cfg.get('class')!r}")

        sizes = cfg["sizes"]
        if len(sizes) != 3:
            raise ValueError(
                f"VoltageElectrodes3D requires 3D sizes [sx,sy,sz], got {sizes}")
        self.sx, self.sy, self.sz = (float(s) for s in sizes)
        self.resolution = int(cfg["resolution"])

        # Patch grid per face. n_patches_per_face = [m, n]
        npf = cfg.get("n_patches_per_face")
        if npf is None or len(npf) != 2:
            raise ValueError(
                f"VoltageElectrodes3D needs n_patches_per_face = [m, n]; got {npf}")
        self.m_patch = int(npf[0])
        self.n_patch = int(npf[1])
        self.n_per_face = self.m_patch * self.n_patch

        # Per-face voltages.
        self.voltages: dict[FaceName3D, np.ndarray] = {}
        for fn in _FACE_NAMES_3D:
            raw = cfg.get(f"voltages_{fn}", [])
            arr = np.asarray(raw, dtype=np.float64).flatten()
            if arr.size not in (0, self.n_per_face):
                raise ValueError(
                    f"voltages_{fn} has {arr.size} entries; "
                    f"expected 0 or {self.n_per_face} (={self.m_patch}*{self.n_patch})")
            self.voltages[fn] = arr

        # Electrode pad inside each patch (default = small fraction of patch).
        # Patches are rectangles whose dims depend on face; use min half-dim by default.
        default_w = 0.5 * min(self.sx / max(self.m_patch, 1),
                              self.sy / max(self.n_patch, 1),
                              self.sz / max(self.n_patch, 1)) / 2.0
        self.electrode_width_um = float(cfg.get("electrode_width_um", default_w))

        # LC grid (same as 2D class's convention).
        self.nx = int(round(self.sx * self.resolution)) + 1
        self.ny = int(round(self.sy * self.resolution)) + 1
        self.nz = int(round(self.sz * self.resolution)) + 1
        self.dx = self.sx / (self.nx - 1)
        self.dy = self.sy / (self.ny - 1)
        self.dz = self.sz / (self.nz - 1)
        self.spacings = (self.dx, self.dy, self.dz)
        self.gshape = (self.nx, self.ny, self.nz)
        self.n_total = self.nx * self.ny * self.nz

    # ---------------- Dirichlet construction ----------------

    def build_dirichlet(self, voltages: dict | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
        if voltages is None:
            voltages = self.voltages
        mask = np.zeros(self.gshape, dtype=bool)
        Vdir = np.zeros(self.gshape, dtype=np.float64)
        for fn in _FACE_NAMES_3D:
            vs = voltages.get(fn)
            if vs is None or len(vs) == 0:
                continue
            vs_arr = np.asarray(vs, dtype=np.float64).flatten()
            if vs_arr.size != self.n_per_face:
                raise ValueError(
                    f"voltages_{fn} has {vs_arr.size} entries; need {self.n_per_face}")
            self._paint_face(mask, Vdir, fn, vs_arr)
        return mask, Vdir

    def _paint_face(self, mask: np.ndarray, Vdir: np.ndarray,
                    face: FaceName3D, vs: np.ndarray) -> None:
        """Stamp m×n patches on `face`. vs[k] for k = i*n_patch + j (row-major)."""
        m, n = self.m_patch, self.n_patch
        pad = self.electrode_width_um  # pad inset, applied around patch center

        if face.startswith("x"):
            # Face spans (y, z). i is along y, j is along z.
            i_idx = 0 if face == "x_min" else self.nx - 1
            patch_a = self.sy / m
            patch_b = self.sz / n
            for i in range(m):
                yc = -self.sy / 2.0 + (i + 0.5) * patch_a
                ja_lo = max(0, int(round((yc - pad + self.sy / 2.0) / self.dy)))
                ja_hi = min(self.ny, int(round((yc + pad + self.sy / 2.0) / self.dy)) + 1)
                for j in range(n):
                    zc = -self.sz / 2.0 + (j + 0.5) * patch_b
                    jb_lo = max(0, int(round((zc - pad + self.sz / 2.0) / self.dz)))
                    jb_hi = min(self.nz, int(round((zc + pad + self.sz / 2.0) / self.dz)) + 1)
                    v = vs[i * n + j]
                    mask[i_idx, ja_lo:ja_hi, jb_lo:jb_hi] = True
                    Vdir[i_idx, ja_lo:ja_hi, jb_lo:jb_hi] = v
        elif face.startswith("y"):
            # Face spans (x, z). i is along x, j is along z.
            j_idx = 0 if face == "y_min" else self.ny - 1
            patch_a = self.sx / m
            patch_b = self.sz / n
            for i in range(m):
                xc = -self.sx / 2.0 + (i + 0.5) * patch_a
                ia_lo = max(0, int(round((xc - pad + self.sx / 2.0) / self.dx)))
                ia_hi = min(self.nx, int(round((xc + pad + self.sx / 2.0) / self.dx)) + 1)
                for j in range(n):
                    zc = -self.sz / 2.0 + (j + 0.5) * patch_b
                    jb_lo = max(0, int(round((zc - pad + self.sz / 2.0) / self.dz)))
                    jb_hi = min(self.nz, int(round((zc + pad + self.sz / 2.0) / self.dz)) + 1)
                    v = vs[i * n + j]
                    mask[ia_lo:ia_hi, j_idx, jb_lo:jb_hi] = True
                    Vdir[ia_lo:ia_hi, j_idx, jb_lo:jb_hi] = v
        else:
            # z faces. Face spans (x, y). i is along x, j is along y.
            k_idx = 0 if face == "z_min" else self.nz - 1
            patch_a = self.sx / m
            patch_b = self.sy / n
            for i in range(m):
                xc = -self.sx / 2.0 + (i + 0.5) * patch_a
                ia_lo = max(0, int(round((xc - pad + self.sx / 2.0) / self.dx)))
                ia_hi = min(self.nx, int(round((xc + pad + self.sx / 2.0) / self.dx)) + 1)
                for j in range(n):
                    yc = -self.sy / 2.0 + (j + 0.5) * patch_b
                    jb_lo = max(0, int(round((yc - pad + self.sy / 2.0) / self.dy)))
                    jb_hi = min(self.ny, int(round((yc + pad + self.sy / 2.0) / self.dy)) + 1)
                    v = vs[i * n + j]
                    mask[ia_lo:ia_hi, jb_lo:jb_hi, k_idx] = True
                    Vdir[ia_lo:ia_hi, jb_lo:jb_hi, k_idx] = v

    # ---------------- Mutators / views ----------------

    def set_voltages(self, **per_face: np.ndarray) -> None:
        for fn, vs in per_face.items():
            if fn not in self.voltages:
                raise KeyError(f"unknown face {fn!r}; expected one of {_FACE_NAMES_3D}")
            arr = np.asarray(vs, dtype=np.float64).flatten()
            if arr.size not in (0, self.n_per_face):
                raise ValueError(
                    f"voltages_{fn} has {arr.size} entries; need 0 or {self.n_per_face}")
            self.voltages[fn] = arr

    @property
    def all_voltages_flat(self) -> np.ndarray:
        return np.concatenate([self.voltages[fn] for fn in _FACE_NAMES_3D])

    @property
    def face_counts(self) -> dict:
        return {fn: int(self.voltages[fn].size) for fn in _FACE_NAMES_3D}

    def summary(self) -> str:
        parts = []
        for fn in _FACE_NAMES_3D:
            vs = self.voltages[fn]
            if vs.size:
                parts.append(f"{fn}=({vs.size})")
        face_str = "  ".join(parts) if parts else "(no electrodes)"
        return (f"VoltageElectrodes3D: {face_str}\n"
                f"  patches per face: {self.m_patch}x{self.n_patch} = {self.n_per_face}\n"
                f"  electrode_width = {self.electrode_width_um:.3f} um, "
                f"gshape = {self.gshape}, "
                f"spacings = ({self.dx:.4f}, {self.dy:.4f}, {self.dz:.4f}) um")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build 3D voltage-electrode mask.")
    ap.add_argument("--path", required=True)
    args = ap.parse_args()
    ve = VoltageElectrodes3D(args.path)
    print(ve.summary())
    mask, Vdir = ve.build_dirichlet()
    print(f"Dirichlet: {int(mask.sum())} pixels fixed (of {ve.n_total} total)")
    if mask.any():
        print(f"V range over fixed pixels: [{Vdir[mask].min():+.3f}, "
              f"{Vdir[mask].max():+.3f}] V")
