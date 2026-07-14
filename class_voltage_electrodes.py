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
        # Optional per-face width override: "electrode_widths": {"x_max": 3.0}
        # (e.g. finger comb on one face + full-coverage ground plane opposite).
        self.electrode_widths = {str(k): float(v) for k, v in
                                 (cfg.get("electrode_widths", {}) or {}).items()}

        # ---- JSON: graded far-field padding (variable outside resolution) ----
        #   "domain_padding": {"enabled": true, "faces": ["y_min","y_max"],
        #                      "n_pad": 20, "growth": 1.2, "w_max_um": 2.0,
        #                      "eps_outside": 1.0}
        # Adds geometrically-growing cells beyond the listed faces so the
        # Neumann walls sit effectively at infinity; the Poisson solve runs on
        # the padded grid and Poisson2D crops V/E back to the LC core grid.
        dp = cfg.get("domain_padding", {}) or {}
        self.pad_enabled = bool(dp.get("enabled", False))
        self.pad_faces = list(dp.get("faces", ["y_min", "y_max"]))
        self.pad_n = int(dp.get("n_pad", 20))
        self.pad_growth = float(dp.get("growth", 1.2))
        self.pad_wmax = dp.get("w_max_um", 2.0)
        self.pad_wmax = None if self.pad_wmax in (None, 0) else float(self.pad_wmax)
        self.eps_outside = float(dp.get("eps_outside", 1.0))

        # ---- JSON: spline boundary-voltage electrode (2-electrode scheme) ----
        #   "spline_electrode": {"enabled": true, "face": "x_min",
        #                        "coeffs": [c1..cN], "span_um": [y_lo,y_hi]|null,
        #                        "degree": 3, "ground_face": "x_max"}
        # V(face coord) = Σ c_k B_k (clamped B-splines; partition of unity →
        # coefficients ARE voltages). Overrides per-pad voltages on `face` and
        # paints a full ground plane on `ground_face`.
        se = cfg.get("spline_electrode", {}) or {}
        self.spline_enabled = bool(se.get("enabled", False))
        self.spline_face = str(se.get("face", "x_min"))
        self.spline_coeffs = np.asarray(se.get("coeffs", []), dtype=np.float64)
        self.spline_span = se.get("span_um", None)
        self.spline_degree = int(se.get("degree", 3))
        self.spline_ground = se.get("ground_face", "x_max")

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

        # Padded-grid geometry (identity when padding is off).
        self._build_padding()

    # ---------------- Graded padding geometry ----------------

    def _build_padding(self):
        """Per-axis width arrays + core offsets for the padded Poisson grid.
        pad_lo/pad_hi[axis] = number of extra nodes below/above the core;
        spacings_padded[axis] = scalar (uniform) or 1D width array (graded)."""
        self.pad_lo = [0, 0, 0]
        self.pad_hi = [0, 0, 0]
        widths = [None, None, None]
        if self.pad_enabled:
            base = {0: self.dx, 1: self.dy}
            for ax, (fmin, fmax, n_core) in enumerate(
                    [("x_min", "x_max", self.nx), ("y_min", "y_max", self.ny)]):
                lo = fmin in self.pad_faces
                hi = fmax in self.pad_faces
                if not (lo or hi):
                    continue
                w = base[ax]
                ramp = w * self.pad_growth ** np.arange(1, self.pad_n + 1)
                if self.pad_wmax is not None:
                    ramp = np.minimum(ramp, self.pad_wmax)
                parts = []
                if lo:
                    parts.append(ramp[::-1])
                    self.pad_lo[ax] = self.pad_n
                parts.append(np.full(n_core, w))
                if hi:
                    parts.append(ramp)
                    self.pad_hi[ax] = self.pad_n
                widths[ax] = np.concatenate(parts)
        self.spacings_padded = tuple(
            widths[ax] if widths[ax] is not None else self.spacings[ax]
            for ax in range(3))
        self.gshape_padded = tuple(
            self.gshape[ax] + self.pad_lo[ax] + self.pad_hi[ax]
            for ax in range(3))
        self.core_slices = tuple(
            slice(self.pad_lo[ax], self.pad_lo[ax] + self.gshape[ax])
            for ax in range(3))

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
        mask_c = np.zeros(self.gshape, dtype=bool)
        Vdir_c = np.zeros(self.gshape, dtype=np.float64)
        for fn in _FACE_NAMES:
            if self.spline_enabled and fn in (self.spline_face, self.spline_ground):
                continue                    # spline scheme owns these faces
            self._paint_face(mask_c, Vdir_c, fn, voltages[fn])
        if self.spline_enabled:
            self._paint_spline(mask_c, Vdir_c, voltages)
        if not self.pad_enabled:
            return mask_c, Vdir_c
        # embed the core Dirichlet into the padded grid (electrodes stay on
        # the physical cell faces; the padding carries no Dirichlet at all)
        mask = np.zeros(self.gshape_padded, dtype=bool)
        Vdir = np.zeros(self.gshape_padded, dtype=np.float64)
        mask[self.core_slices] = mask_c
        Vdir[self.core_slices] = Vdir_c
        return mask, Vdir

    def _paint_spline(self, mask: np.ndarray, Vdir: np.ndarray,
                      voltages: dict | None = None) -> None:
        """Paint the spline-voltage electrode on `spline_face` and a full
        ground plane on `spline_ground`. Coefficients may be overridden at
        call time via voltages={"spline": array} (reservoir-computer input)."""
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(
            _os.path.abspath(__file__)), "..", "BlockOptimization", "E_field_stuff"))
        try:
            from spline_voltage import bspline_basis
        except ImportError:
            from E_field_stuff.spline_voltage import bspline_basis  # type: ignore
        coeffs = self.spline_coeffs
        if voltages and "spline" in voltages:
            coeffs = np.asarray(voltages["spline"], dtype=np.float64)
        if coeffs.size == 0:
            raise ValueError("spline_electrode.enabled but no coeffs given")
        face = self.spline_face
        if face.startswith("x"):
            fc = (np.arange(self.ny) * self.dy) - self.sy / 2.0
            span = self.spline_span or [-self.sy / 2.0, self.sy / 2.0]
        else:
            fc = (np.arange(self.nx) * self.dx) - self.sx / 2.0
            span = self.spline_span or [-self.sx / 2.0, self.sx / 2.0]
        B = np.asarray(bspline_basis(fc, float(span[0]), float(span[1]),
                                     n_ctrl=coeffs.size, degree=self.spline_degree))
        prof = B @ coeffs
        sup = B.sum(axis=1) > 1e-12
        if face == "x_min":
            mask[0, sup, :] = True;  Vdir[0, sup, :] = prof[sup, None]
        elif face == "x_max":
            mask[-1, sup, :] = True; Vdir[-1, sup, :] = prof[sup, None]
        elif face == "y_min":
            mask[sup, 0, :] = True;  Vdir[sup, 0, :] = prof[sup, None]
        else:
            mask[sup, -1, :] = True; Vdir[sup, -1, :] = prof[sup, None]
        g = self.spline_ground
        if g:
            if g == "x_min":
                mask[0, :, :] = True;  Vdir[0, :, :] = 0.0
            elif g == "x_max":
                mask[-1, :, :] = True; Vdir[-1, :, :] = 0.0
            elif g == "y_min":
                mask[:, 0, :] = True;  Vdir[:, 0, :] = 0.0
            else:
                mask[:, -1, :] = True; Vdir[:, -1, :] = 0.0

    def _paint_face(self, mask: np.ndarray, Vdir: np.ndarray,
                    face: FaceName, vs: np.ndarray) -> None:
        """Stamp `vs.size` equally-spaced electrodes onto `face` of mask/Vdir."""
        if vs.size == 0:
            return
        n_e = vs.size
        width = self.electrode_widths.get(face, self.electrode_width_um)
        # NaN entry = NO electrode at that slot (floating wall, no Dirichlet) —
        # used by the discrete electrode-position optimization.
        if face.startswith("y"):
            # Electrodes span x direction
            pitch = self.sx / n_e
            j_idx = 0 if face == "y_min" else self.ny - 1
            for k in range(n_e):
                if not np.isfinite(vs[k]):
                    continue
                xc = -self.sx / 2.0 + (k + 0.5) * pitch
                i_lo = int(round((xc - width / 2.0 + self.sx / 2.0) / self.dx))
                i_hi = int(round((xc + width / 2.0 + self.sx / 2.0) / self.dx)) + 1
                i_lo = max(0, i_lo); i_hi = min(self.nx, i_hi)
                mask[i_lo:i_hi, j_idx, :] = True
                Vdir[i_lo:i_hi, j_idx, :] = vs[k]
        else:
            # Electrodes span y direction
            pitch = self.sy / n_e
            i_idx = 0 if face == "x_min" else self.nx - 1
            for k in range(n_e):
                if not np.isfinite(vs[k]):
                    continue
                yc = -self.sy / 2.0 + (k + 0.5) * pitch
                j_lo = int(round((yc - width / 2.0 + self.sy / 2.0) / self.dy))
                j_hi = int(round((yc + width / 2.0 + self.sy / 2.0) / self.dy)) + 1
                j_lo = max(0, j_lo); j_hi = min(self.ny, j_hi)
                mask[i_idx, j_lo:j_hi, :] = True
                Vdir[i_idx, j_lo:j_hi, :] = vs[k]

    # ---------------- Mutators ----------------

    def set_voltages(self, **per_face: np.ndarray) -> None:
        """Set voltages for one or more faces by name. Unspecified faces unchanged.
        Example: ve.set_voltages(y_max=[1,2,3,4], y_min=[0,0,0,0])
        """
        for fn, vs in per_face.items():
            if fn == "spline":
                # spline-electrode control points (the 2-electrode scheme's
                # input vector) — consumed by _paint_spline at build time
                self.spline_coeffs = np.asarray(vs, dtype=np.float64).flatten()
                continue
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
