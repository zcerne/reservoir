"""GPUmeep replica of `class_simulation.py` from the reservoir project.

Reads a `simulation_data.json` describing the same geometry conventions as the
MEEP class_simulation, but runs the simulation with the GPUmeep JAX FDTD
engine instead of MEEP. Output files are written in the same npz format as the
MEEP version so existing post-processing (T-matrix assembly, etc.) just works.

Scope:
  * 2D simulations (dimention == 2): native TE solver (fdtd_2d).
  * 3D simulations (dimention == 3): full anisotropic Yee solver (fdtd_core + pml).
  * Object types: guide, reservoir, source, monitor (type "flux"/"1Ddft"/"2Ddft").
  * Single CW source; amplitude scalar, 1D strip list (2D), or grid_shape list (3D).
  * Boundary: PML in all non-propagation axes.

Usage:
    python class_simulation_gpu.py --path data/test2D [--empty] [--precision fp32]
    python class_simulation_gpu.py --path data/source_mnist --precision fp64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

# Decide precision before importing JAX. Default fp64 (matches MEEP); pass
# --precision fp32 on the CLI for 3-10x speed on consumer GPUs.
# fp32 is fine for res ≤ 40 / shorter runs but can develop instability at
# higher resolutions due to round-off accumulation over many steps.
# NOTE: the GPUmeep core hardcodes jnp.float64 in array allocations; when x64
# is OFF, JAX silently downcasts those to float32, so the same code runs at
# fp32. When x64 is ON, they stay fp64.
if "--precision" in sys.argv:
    _i = sys.argv.index("--precision")
    _prec = sys.argv[_i + 1]
else:
    _prec = "fp64"
import jax
if _prec == "fp64":
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
_JDTYPE = jnp.float64 if _prec == "fp64" else jnp.float32
import numpy as np

# Locate the GPUmeep `src/` directory containing fdtd_2d, materials, etc.
# Tries (in order): $GPUMEEP_PATH, then common checkout locations.
# On the cluster/RunPod, set GPUMEEP_PATH to wherever GPUmeep is checked out.
_gpumeep_candidates = [
    os.environ.get("GPUMEEP_PATH"),
    os.path.expanduser("~/Nextcloud/Doktorski/Projects/GPUmeep/gitcode/src"),
    os.path.expanduser("~/Nextcloud/Doktorski/Projects/GPUmeep/src"),
    os.path.expanduser("~/GPUmeep/gitcode/src"),
    os.path.expanduser("~/GPUmeep/src"),
    os.path.expanduser("~/Projects/GPUmeep/src"),
    os.path.dirname(os.path.abspath(__file__)),
]
for _p in _gpumeep_candidates:
    if _p and os.path.exists(os.path.join(_p, "fdtd_2d.py")):
        sys.path.insert(0, _p)
        break
else:
    raise ImportError(
        "Could not find GPUmeep src/ directory. Set GPUMEEP_PATH env var "
        "to point to it, or place this script in the same folder as fdtd_2d.py."
    )
import fdtd_2d as f2  # noqa: E402
import fdtd_core as fc  # noqa: E402  (3D engine)
import pml as pml3d  # noqa: E402  (3D CPML + anisotropic step)
import materials as mats  # noqa: E402
import sources as src3d  # noqa: E402
import monitors as mon3d  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _pos_to_center_size_2d(pos, on_edge_x, on_size_x, cell_x, cell_y):
    """Translate a MEEP-style position dict to (center_x, size_y) in 2D.

    Position can be 'left', 'right', 'center' relative to the on_object.
    Returns x in MEEP coords (cell centered at origin) and the y span of the
    feature (full cell_y by default for monitors/sources without size).

    Size convention disambiguation (matches BlockOptimization fix 2026-06-02):
      * scalar             → 1D: size_y = value
      * [sy]               → 1D: size_y = sy
      * [sy, 0]            → 1D / plane source / 1D monitor (MEEP plane convention)
      * [sx, sy] (sy > 0)  → 2D box monitor: size_y = sy (the SECOND entry)
    Disambiguator: size[1] > 0 → 2D, so size[1] is sy; else size[0] is sy.
    """
    if isinstance(pos, dict):
        label = pos.get("position", "center")
        raw = pos.get("size", [0.0, 0.0])
    else:
        label = str(pos) if pos else "center"
        raw = [0.0, 0.0]
    if isinstance(raw, (int, float)):
        raw = [float(raw)]
    if len(raw) == 1:
        sy = float(raw[0]) if raw[0] else cell_y
    elif float(raw[1]) > 0:                  # [sx, sy] 2D box
        sy = float(raw[1])
    else:                                    # [sy, 0] plane / 1D
        sy = float(raw[0]) if raw[0] else cell_y
    x_meep = {"left": on_edge_x,
              "right": on_edge_x + on_size_x}.get(label, on_edge_x + on_size_x / 2)
    return x_meep, sy


def _mirror_n_layers(T_target: float, indices) -> int:
    """Number of quarter-wave layers for T ≤ T_target (MEEP class_mirror formula).
    R = [(ρ−1)/(ρ+1)]², ρ = (n_H/n_L)^(2N) ⇒ N = ceil(log ρ / (2 log(n_H/n_L)))."""
    if T_target <= 0:
        raise ValueError(f"Mirror transmission must be > 0, got {T_target}")
    if T_target >= 1.0:
        return 2
    n_H = float(max(indices)); n_L = float(min(indices))
    if n_H <= n_L:
        raise ValueError("Mirror: n_H must be > n_L")
    sqrtR = np.sqrt(1.0 - T_target)
    rho = (1.0 + sqrtR) / (1.0 - sqrtR)
    N = np.log(rho) / (2.0 * np.log(n_H / n_L))
    return 2 * max(1, int(np.ceil(N)))


@dataclass(frozen=True)
class _STEDSource:
    """Current (J) source over a 2D amplitude map, MEEP semantics: E += -dt·ε⁻¹·J.
    `width>0` → MEEP Gaussian pulse (peak at start_time+cutoff·width); else CW with
    tanh turn-on. Works on Fields2D or FieldsFull2D (reconstructs the same type)."""
    component: str
    amp_map: object
    eps_inv_map: object
    freq: float
    width: float
    start_time: float
    cutoff: float
    dt: float
    src_scale: float = 1.0    # MEEP current-source absolute-scale factor (≈0.39·res)

    def _dipole(self, t):
        # MEEP gaussian_src_time::dipole (sources.cpp:104), copied verbatim:
        #   exp(-tt²/2w²) · polar(1,-ωtt) · 1/(-2πi·f),   tt = t - peak_time,
        # zeroed for |tt| > cutoff·width. peak_time = start_time + cutoff·width.
        tt = t - (self.start_time + self.cutoff * self.width)
        omega = 2.0 * jnp.pi * self.freq
        g = jnp.exp(-(tt ** 2) / (2.0 * self.width ** 2))
        amp = 1.0 / (-2.0j * jnp.pi * self.freq)          # = i/(2πf)
        dip = g * jnp.exp(-1j * omega * tt) * amp
        return jnp.where(jnp.abs(tt) > self.cutoff * self.width, 0.0 + 0.0j, dip)

    def _J(self, t):
        if self.width > 0.0:
            # MEEP-native D-source current (is_integrated=False, our Ey pulse):
            # computed by calc_sources(time()+0.5·dt) (step.cpp:96) with the
            # FORWARD-difference current (meep.hpp: current(a,dt) =
            # (dipole(a+dt) − dipole(a))/dt), real part injected. gpu's loop
            # time() = t (pre-step), so a = t + 0.5·dt. Paired with the DFT
            # referencing E at (n+1)·dt, this reproduces MEEP's phase exactly.
            a = t + 0.5 * self.dt
            return jnp.real((self._dipole(a + self.dt) - self._dipole(a)) / self.dt)
        period = 1.0 / self.freq
        ramp = 0.5 * (1.0 + jnp.tanh((t - 5.0 * period) / (0.5 * period)))
        return ramp * jnp.sin(2.0 * jnp.pi * self.freq * t)

    def apply(self, fields, t):
        val = -self.dt * self.src_scale * self.eps_inv_map * self._J(t) * self.amp_map
        d = dict(fields._asdict())
        d[self.component] = d[self.component] + val
        return type(fields)(**d)

    def apply_D(self, D, t):
        """D-form injection: a current source adds to the displacement field D
        directly (NO ε⁻¹ — the ε⁻¹ is applied later in E = ε⁻¹·(D−P)).
        D = (Dx@Ex-face, Dy@Ey-face, Dz@node); component maps Ex→Dx, Ey→Dy, Ez→Dz."""
        val = -self.dt * self.src_scale * self._J(t) * self.amp_map
        Dx, Dy, Dz = D
        if self.component == "Ex":
            Dx = Dx + val
        elif self.component == "Ey":
            Dy = Dy + val
        else:  # "Ez"
            Dz = Dz + val
        return (Dx, Dy, Dz)


def _src_overlap_weights(lo, hi, n, dx, offset, cen):
    """MEEP source weighting for a FINITE-size direction (sources.cpp:273
    IVEC_LOOP_WEIGHT). Each Yee sample k sits at p_k=(k+offset)·dx−cen and
    'owns' [p_k−dx/2, p_k+dx/2]; its weight is the fraction of that cell inside
    the source span [lo,hi] (interior → 1, boundary → fractional). This makes the
    integrated source current independent of resolution. `cen` = MEEP grid-centre
    offset = floor(N/2)·dx (NOT N·dx/2; differ by dx/2 for odd N)."""
    k = np.arange(n)
    pk = (k + offset) * dx - cen
    ov = np.clip(np.minimum(pk + dx / 2.0, hi) - np.maximum(pk - dx / 2.0, lo), 0.0, dx)
    return ov / dx


def _src_delta_weights(c, n, dx, offset, cen):
    """MEEP source weighting for a ZERO-size (delta) direction: linear
    interpolation onto the two nearest Yee samples, weights summing to 1. The
    absolute delta-function density (×gv.a=res) is carried separately in
    src_scale (sources.cpp:483). `cen` = MEEP grid-centre offset = floor(N/2)·dx."""
    w = np.zeros(n)
    f = (c + cen) / dx - offset                    # fractional sample index
    i0 = int(np.floor(f)); frac = f - i0
    if 0 <= i0 < n:
        w[i0] += 1.0 - frac
    if 0 <= i0 + 1 < n:
        w[i0 + 1] += frac
    return w


def _meep_to_grid_x(x_meep, cen, dx):
    """Convert MEEP x coord to GPUmeep grid index. `cen` = MEEP grid-centre offset
    = floor(Nx/2)·dx (icenter=round_down_to_even(Nx), vec.cpp:1089); this is
    gx/2 only for EVEN Nx, and gx/2−dx/2 for odd Nx."""
    return int(round((x_meep + cen) / dx))


def _meep_to_grid_y_range(center_y, sy, cen, dx):
    """Convert MEEP y center+size to GPUmeep grid index range. `cen` = floor(Ny/2)·dx."""
    y_meep_lo = center_y - sy / 2
    y_meep_hi = center_y + sy / 2
    j_lo = int(round((y_meep_lo + cen) / dx))
    j_hi = int(round((y_meep_hi + cen) / dx))
    return j_lo, j_hi


# ----------------------------------------------------------------------
# Main Simulation class
# ----------------------------------------------------------------------


@dataclass
class SimulationGPU:
    folder_path: str
    empty: bool = False     # if True, skip building reservoir/SLM materials
    # Per-run amplitude override for the SIGNAL source (basis/forward runs).
    # None → use the JSON amplitude. Set via {source_key: [amps]} to sweep inputs
    # without rewriting the JSON (mirrors SimulationT._run_basis).
    amp_override: dict | None = None

    args: dict = field(default_factory=dict)
    objects_args: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)

    resolution: int = 40
    cell_x: float = 0.0
    cell_y: float = 0.0
    cell_z: float = 0.0
    dim: int = 2
    dx: float = 0.0
    Nx: int = 0
    Ny: int = 0
    Nz: int = 0
    grid: object = None             # f2.Grid2D (2D) or fc.Grid (3D)
    material: object = None         # f2.Aniso2DYee (2D) or mats.Anisotropic (3D)
    eps_inv_zz: object = None       # scalar ε_zz⁻¹ at nodes (full-vector/STED path)
    pml: object = None              # f2.CPML2D (2D) or pml3d.CPML (3D)
    sources: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    gain: object = None             # STED gain: {"coeffs", "state", "mask"} or None
    dt: float = 0.0
    _n_max: float = 1.0
    run_until_override: object = None   # if set, overrides JSON run_until (testing)
    force_fullvector: bool = False      # route non-dye 2D runs through the full-vector path

    # ---------------- Setup ----------------

    def _set_data(self):
        with open(os.path.join(self.folder_path, "simulation_data.json")) as f:
            self.args.update(json.load(f))
        sim_dir = "simulation_empty" if self.empty else "simulation"
        self.paths = {
            "simulation": os.path.join(self.folder_path, sim_dir),
            "snapshots":  os.path.join(self.folder_path, sim_dir, "snapshots"),
            "figures":    os.path.join(self.folder_path, "figures"),
        }
        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)

    def _update_all_args(self):
        """Compute cell dimensions + each object's absolute position, mirroring
        the MEEP class_simulation logic."""
        self.objects_args = []      # reset (idempotent: safe to call more than once)
        pml = float(self.args.get("pml_size", 2.0))
        self.dim = self.args.get("dimention", 1)
        self.resolution = int(self.args["resolution"])
        self.dx = 1.0 / self.resolution
        cell_y_arg = (float(self.args.get("cell_size_y", 0.0)) if self.dim > 1
                      else 4.0 / self.resolution)
        cell_z_arg = float(self.args.get("cell_size_z", 0.0)) if self.dim > 2 else 0.0

        # Walk objects in declared order, accumulate x positions
        current_x = 0.0
        for key in self.args["object_order"]:
            obj = dict(self.args[key])
            obj["_key"] = key
            obj["edge_x_local"] = current_x
            if isinstance(obj.get("sizes"), list):
                obj["size_x"] = float(obj["sizes"][0])
            # DBR mirror: x-thickness = Σ quarter-wave layers (MEEP class_simulation)
            if obj.get("class") == "mirror" and "size_x" not in obj:
                lam = float(obj["lam"])
                indices = obj.get("n_indexes", obj.get("indexes", [1.0, 1.0]))
                n_lays = (int(obj["n_layers"]) if "n_layers" in obj
                          else _mirror_n_layers(float(obj["transmission"]), indices))
                obj["n_layers_resolved"] = n_lays
                obj["size_x"] = sum(lam / 4.0 / float(indices[i % 2]) for i in range(n_lays))
            self.objects_args.append(obj)
            current_x += float(obj.get("size_x", 0.0))

        self.cell_x = current_x + 2 * pml
        self.cell_y = cell_y_arg
        self.cell_z = cell_z_arg
        x0 = -self.cell_x / 2 + pml   # left edge of first object in MEEP coords

        for obj in self.objects_args:
            edge_x = obj["edge_x_local"] + x0
            size_x = float(obj.get("size_x", 0.0))
            cls = obj.get("class", "")

            if cls == "guide" or cls == "reservoir" or cls == "voltage_reservoir":
                obj["center_x_meep"] = edge_x + size_x / 2
                obj["edge_x_meep"] = edge_x

            elif cls == "mirror":
                obj["x_start_meep"] = edge_x         # left edge (MEEP coords)
                obj["edge_x_meep"] = edge_x
                obj["center_x_meep"] = edge_x + size_x / 2

            elif cls in ("source", "monitor"):
                on_object = obj.get("on_object", -1)
                if on_object == -1 and isinstance(obj.get("position"), dict):
                    on_object = obj["position"].get("on_object", -1)
                ref = next((r for r in self.objects_args
                            if (isinstance(on_object, str) and r.get("_key") == on_object)
                            or (isinstance(on_object, int) and on_object >= 0
                                and r is self.objects_args[on_object])),
                           None) if on_object != -1 else None
                on_edge = ref["edge_x_meep"] if ref else -self.cell_x / 2
                on_size = float(ref.get("size_x", 0.0)) if ref else self.cell_x
                x_meep, sy = _pos_to_center_size_2d(
                    obj.get("position", {}), on_edge, on_size, self.cell_x, self.cell_y
                )
                obj["center_x_meep"] = x_meep
                obj["size_y_meep"] = sy

        self.Nx = int(round(self.cell_x / self.dx))
        self.Ny = int(round(self.cell_y / self.dx))
        # MEEP rounds the cell to an integer number of pixels and CENTERS that rounded
        # grid at 0 (grid_volume = [-N*dx/2, +N*dx/2]); the GEOMETRY stays in the
        # continuous unrounded cell frame (x0 = -cell_x/2 + pml above). Replicate this:
        # use the rounded extents gx/gy(/gz) for every grid<->coordinate mapping so the
        # Yee nodes coincide with MEEP's. For integer-pixel cells gx==cell_x (LC/air
        # configs unchanged); only mirror configs — whose λ/4 layer thicknesses make
        # cell_x non-integer — shift, correcting a ~0.25 px grid mis-registration that
        # sampled the thin DBR layers at the wrong sub-pixel phase and detuned the cavity.
        self.gx = self.Nx * self.dx
        self.gy = self.Ny * self.dx
        # MEEP grid-centre offset: icenter = round_down_to_even(N) half-pixels
        # (vec.cpp:1089) → physical origin at floor(N/2)·dx, NOT N·dx/2. Equal for
        # EVEN N; for ODD N they differ by dx/2. Config 4 is 588 px (even) at res40
        # but 1177 px (odd) at res80 — using gx/2 shifted the whole grid dx/2 vs
        # MEEP, detuning the cavity. Use cx/cy(/cz) for every coordinate↔grid map.
        self.cx = (self.Nx // 2) * self.dx
        self.cy = (self.Ny // 2) * self.dx
        if self.dim >= 3:
            self.Nz = int(round(self.cell_z / self.dx))
            self.gz = self.Nz * self.dx
            self.cz = (self.Nz // 2) * self.dx
            self.grid = fc.Grid(Nx=self.Nx, Ny=self.Ny, Nz=self.Nz,
                                dx=self.dx, dy=self.dx, dz=self.dx)
            print(f"Cell = {self.cell_x} x {self.cell_y} x {self.cell_z}, "
                  f"grid = {self.grid.shape}, dx = {self.dx}, "
                  f"{self.Nx*self.Ny*self.Nz/1e6:.1f}M cells")
        else:
            self.grid = f2.Grid2D(Nx=self.Nx, Ny=self.Ny, dx=self.dx, dy=self.dx)
            print(f"Cell = {self.cell_x} x {self.cell_y}, grid = {self.grid.shape}, dx = {self.dx}")

    # ---------------- Material ----------------

    def _build_material(self):
        """Build the full 3×3 anisotropic ε⁻¹ tensor (AnisoFull2D) over the cell,
        reproducing MEEP's subpixel scheme EXACTLY (meepgeom.cpp eff_chi1inv_matrix).

        Two regimes, exactly as MEEP:
          * The LC reservoir is a material-FUNCTION block → in MEEP a "variable"
            material.  get_front_object() bails on variable materials (meepgeom.cpp
            :1012) and, with do_averaging off for a plain callback, the code hits
            `goto trivial` → ε is POINT-SAMPLED at the pixel centre.  So the whole LC
            region (interior AND its outer boundary) is point-sampled — never averaged.
          * The isotropic constant-mp.Medium blocks (DBR mirror layers, index≠1 guides)
            ARE genuine 2-material object boundaries.  MEEP computes the exact analytic
            normal (normal_to_fixed_object) and the exact fill fraction
            (box_overlap_with_object) and applies the Kottke–Farjadpour–Johnson
            τ-transform: harmonic mean of ε along the interface normal, arithmetic mean
            in the tangential plane, with the tensor rotated to the normal frame and
            back.  For AXIS-ALIGNED isotropic blocks the analytic normal is a coordinate
            axis and the τ-transform collapses to a per-axis harmonic/arithmetic average
            that is bit-for-bit the same as the full tensor routine — and, because the
            harmonic and arithmetic means are symmetric under front↔behind swap, it is
            independent of the geometry object id ordering.

        Set GPUMEEP_NOAVG=1 to reproduce a MEEP run with eps_averaging=False (pure
        point-sampling everywhere).
        """
        if self.empty:
            self._build_vacuum_material()
            return
        self._setup_lc_interp()
        i = np.arange(self.Nx); j = np.arange(self.Ny)

        def build_at(x_off, y_off):
            X = ((i + x_off) * self.dx - self.cx)[:, None] * np.ones((1, self.Ny))
            Y = ((j + y_off) * self.dx - self.cy)[None, :] * np.ones((self.Nx, 1))
            e6 = list(self._eps_sharp_at(X, Y))          # MEEP `goto trivial`: point-sample
            if not os.environ.get("GPUMEEP_NOAVG"):
                e6 = self._overlay_iso_kottke(e6, X, Y)  # exact Kottke at isotropic boundaries
            return e6

        eEx = build_at(0.5, 0.0); eEy = build_at(0.0, 0.5); end = build_at(0.0, 0.0)

        def inv3(e6):
            exx, eyy, ezz, exy, exz, eyz = e6
            det = (exx * (eyy * ezz - eyz * eyz) - exy * (exy * ezz - eyz * exz)
                   + exz * (exy * eyz - eyy * exz))
            return ((eyy * ezz - eyz * eyz) / det, (exx * ezz - exz * exz) / det,
                    (exx * eyy - exy * exy) / det, (exz * eyz - exy * ezz) / det,
                    (exy * eyz - exz * eyy) / det, (exz * exy - exx * eyz) / det)
        ixx_Ex, _, _, ixy_Ex, ixz_Ex, _ = inv3(eEx)
        _, iyy_Ey, _, ixy_Ey, _, iyz_Ey = inv3(eEy)
        _, _, izz_nd, _, ixz_nd, iyz_nd = inv3(end)
        J = lambda a: jnp.asarray(a, _JDTYPE)
        self.material = f2.AnisoFull2D(
            ixx_Ex=J(ixx_Ex), ixy_Ex=J(ixy_Ex), ixz_Ex=J(ixz_Ex),
            ixy_Ey=J(ixy_Ey), iyy_Ey=J(iyy_Ey), iyz_Ey=J(iyz_Ey),
            ixz_nd=J(ixz_nd), iyz_nd=J(iyz_nd), izz_nd=J(izz_nd))
        self.eps_inv_zz = J(izz_nd)
        self._n_max = float(np.sqrt(max(eEx[0].max(), eEy[1].max(), end[2].max())))

    def _iso_rects(self):
        """Isotropic constant-ε rectangles that MEEP subpixel-averages: DBR mirror
        layers and index≠1 guides, as (x0, x1, y0, y1, n2) in MEEP coordinates. The
        LC reservoir is deliberately EXCLUDED — it is a material-function block that
        MEEP point-samples (get_front_object bails on variable materials)."""
        rects = []
        for obj in self.objects_args:
            cls = obj.get("class")
            if cls == "guide":
                idx = float(obj.get("index", 1.0))
                if abs(idx - 1.0) < 1e-12:
                    continue
                x0 = obj["edge_x_meep"]; x1 = x0 + float(obj["size_x"])
                sizes = obj.get("sizes")
                sy = (float(sizes[1]) if isinstance(sizes, list) and len(sizes) > 1
                      else self.cell_y)
                rects.append((x0, x1, -sy / 2, sy / 2, idx ** 2))
            elif cls == "mirror":
                sy = float(obj.get("size_y", self.cell_y))
                for (x0, x1, n) in self._mirror_layers(obj):
                    rects.append((x0, x1, -sy / 2, sy / 2, n ** 2))
        return rects

    def _overlay_iso_kottke(self, e6, X, Y):
        """MEEP's EXACT subpixel average at isotropic-block object boundaries.

        Reproduces geom_epsilon::eff_chi1inv_matrix (meepgeom.cpp:1066) specialised to
        axis-aligned constant-ε blocks:
          * fill  = box_overlap_with_object: the analytic fraction of the pixel
                    [X±dx/2]×[Y±dx/2] lying inside each rectangle (product of the
                    per-axis 1-D overlaps).
          * normal = normal_to_fixed_object: for an axis-aligned block this is the
                     coordinate axis of the nearest face, i.e. the axis along which the
                     pixel is partially covered (x-face takes priority at a corner, as
                     the nearest-face rule does for the thin DBR layers).
          * τ-transform: harmonic mean of ε along the normal, arithmetic mean in the
                     two tangential directions, off-diagonals zero (isotropic media).
        Pixels wholly inside one material, wholly in vacuum, or inside the LC region
        keep their trivial point-sampled ε (`goto trivial`)."""
        rects = self._iso_rects()
        if not rects:
            return e6
        exx, eyy, ezz, exy, exz, eyz = (np.array(c, dtype=np.float64) for c in e6)
        h = 0.5 * self.dx
        xlo = X - h; xhi = X + h; ylo = Y - h; yhi = Y + h
        tol = 1e-9

        cov_sum = np.zeros(X.shape)      # Σ pixel-area fraction inside iso rects
        arith = np.zeros(X.shape)        # Σ cov_k · n²_k (vacuum contribution added below)
        harm_inv = np.zeros(X.shape)     # Σ cov_k / n²_k
        maxcov = np.zeros(X.shape)       # largest single-rect coverage → interior test
        part_x = np.zeros(X.shape, bool) # pixel straddles an x-normal face
        part_y = np.zeros(X.shape, bool) # pixel straddles a y-normal face
        for (x0, x1, y0, y1, n2) in rects:
            fx = np.clip(np.minimum(xhi, x1) - np.maximum(xlo, x0), 0.0, self.dx) / self.dx
            fy = np.clip(np.minimum(yhi, y1) - np.maximum(ylo, y0), 0.0, self.dx) / self.dx
            cov = fx * fy
            arith = arith + cov * n2
            harm_inv = harm_inv + cov / n2
            cov_sum = cov_sum + cov
            maxcov = np.maximum(maxcov, cov)
            part_x = part_x | ((fx > tol) & (fx < 1.0 - tol) & (fy > tol))
            part_y = part_y | ((fy > tol) & (fy < 1.0 - tol) & (fx > tol))

        fvac = np.clip(1.0 - cov_sum, 0.0, 1.0)          # remaining pixel fraction = vacuum
        arith_full = arith + fvac * 1.0
        harm_full = 1.0 / (harm_inv + fvac / 1.0)

        # A boundary pixel = not entirely inside one material AND not entirely vacuum.
        boundary = (cov_sum > tol) & (maxcov < 1.0 - tol)
        use_x = boundary & (part_x | ~part_y)            # normal ∥ x (corner → x priority)
        use_y = boundary & part_y & ~part_x              # normal ∥ y

        exx = np.where(use_x, harm_full, np.where(use_y, arith_full, exx))
        eyy = np.where(use_y, harm_full, np.where(use_x, arith_full, eyy))
        ezz = np.where(boundary, arith_full, ezz)        # z always tangential → arithmetic
        exy = np.where(boundary, 0.0, exy)
        exz = np.where(boundary, 0.0, exz)
        eyz = np.where(boundary, 0.0, eyz)
        return (exx, eyy, ezz, exy, exz, eyz)

    # ---------------- point-sampled (trivial) ε ----------------

    def _eps_sharp_at(self, X, Y):
        """FULL 3×3 ε (6 comp) at MEEP-coord points (X,Y) with SHARP boundaries — no
        blending (Kottke does the averaging). LC director inside the reservoir, n²·I
        for mirror layers / index≠1 guides, vacuum elsewhere. X,Y are (Nx,Ny) arrays."""
        sh = X.shape
        exx = np.ones(sh); eyy = np.ones(sh); ezz = np.ones(sh)
        exy = np.zeros(sh); exz = np.zeros(sh); eyz = np.zeros(sh)
        # --- LC reservoir ---
        res = next((o for o in self.objects_args
                    if o.get("class") in ("reservoir", "voltage_reservoir")), None)
        if res is not None and getattr(self, "_lc_interp", None) is not None:
            n_o = float(res.get("n_o", 1.5)); n_e = float(res.get("n_e", 1.7))
            eps_perp = n_o ** 2; delta = n_e ** 2 - n_o ** 2
            rx0 = res["edge_x_meep"]; rx1 = rx0 + float(res["size_x"])
            sizes = res.get("sizes")
            ry = (float(sizes[1]) if isinstance(sizes, list) and len(sizes) > 1 else self.cell_y)
            m = (X >= rx0) & (X < rx1) & (np.abs(Y) <= ry / 2)
            ip, it, x0, x1, y0, y1 = self._lc_interp
            xe = np.clip(X, x0, x1); ye = np.clip(Y, y0, y1)
            phi = ip.ev(xe, ye); theta = it.ev(xe, ye)
            nx = np.sin(theta) * np.cos(phi); ny = np.sin(theta) * np.sin(phi); nz = np.cos(theta)
            exx = np.where(m, eps_perp + delta * nx * nx, exx)
            eyy = np.where(m, eps_perp + delta * ny * ny, eyy)
            ezz = np.where(m, eps_perp + delta * nz * nz, ezz)
            exy = np.where(m, delta * nx * ny, exy)
            exz = np.where(m, delta * nx * nz, exz)
            eyz = np.where(m, delta * ny * nz, eyz)
        # --- isotropic guides + DBR mirror layers (override LC/vacuum) ---
        for obj in self.objects_args:
            cls = obj.get("class")
            if cls == "guide":
                idx = float(obj.get("index", 1.0))
                if abs(idx - 1.0) < 1e-12:
                    continue
                x0 = obj["edge_x_meep"]; x1 = x0 + float(obj["size_x"])
                sizes = obj.get("sizes")
                sy = (float(sizes[1]) if isinstance(sizes, list) and len(sizes) > 1 else self.cell_y)
                m = (X >= x0) & (X < x1) & (np.abs(Y) <= sy / 2)
                for arr in (exy, exz, eyz):
                    arr[m] = 0.0
                exx = np.where(m, idx ** 2, exx); eyy = np.where(m, idx ** 2, eyy)
                ezz = np.where(m, idx ** 2, ezz)
            elif cls == "mirror":
                sy = float(obj.get("size_y", self.cell_y))
                for (x0, x1, n) in self._mirror_layers(obj):
                    m = (X >= x0) & (X < x1) & (np.abs(Y) <= sy / 2)
                    exy = np.where(m, 0.0, exy); exz = np.where(m, 0.0, exz); eyz = np.where(m, 0.0, eyz)
                    exx = np.where(m, n ** 2, exx); eyy = np.where(m, n ** 2, eyy)
                    ezz = np.where(m, n ** 2, ezz)
        return exx, eyy, ezz, exy, exz, eyz

    def _setup_lc_interp(self):
        """Build the LC director interpolators once (used by _eps_sharp_at / Kottke)."""
        self._lc_interp = None
        res = next((o for o in self.objects_args
                    if o.get("class") in ("reservoir", "voltage_reservoir")), None)
        if res is not None:
            lc_path = os.path.join(self.folder_path, "simulation", "lc_fields.npz")
            if not os.path.exists(lc_path):
                raise FileNotFoundError(f"Need relaxed LC field at {lc_path}.")
            lc = np.load(lc_path)
            mid = lc["phi"].shape[2] // 2
            phi_lc = np.asarray(lc["phi"])[:, :, mid]; theta_lc = np.asarray(lc["theta"])[:, :, mid]
            lc_x = np.asarray(lc["x"]); lc_y = np.asarray(lc["y"])
            sizes = res.get("sizes")
            ry = (float(sizes[1]) if isinstance(sizes, list) and len(sizes) > 1 else self.cell_y)
            # LC local coords → MEEP coords (reservoir centered in y, starts at edge in x)
            xg = (lc_x - lc_x.min()) + (res["edge_x_meep"])
            yg = (lc_y - lc_y.min()) + (-ry / 2)
            from scipy.interpolate import RectBivariateSpline
            ip = RectBivariateSpline(xg, yg, phi_lc, kx=3, ky=3)
            it = RectBivariateSpline(xg, yg, theta_lc, kx=3, ky=3)
            self._lc_interp = (ip, it, xg[0], xg[-1], yg[0], yg[-1])

    def _build_vacuum_material(self):
        one = jnp.ones((self.Nx, self.Ny), dtype=_JDTYPE)
        zero = jnp.zeros((self.Nx, self.Ny), dtype=_JDTYPE)
        self.material = f2.Aniso2DYee(
            eps_inv_xx_Ex=one, eps_inv_xy_Ex=zero,
            eps_inv_yy_Ey=one, eps_inv_xy_Ey=zero,
        )
        self.eps_inv_zz = one
        self._n_max = 1.0

    def _mirror_layers(self, obj):
        """(x_lo, x_hi, n) per quarter-wave DBR layer in MEEP coords (matches
        class_mirror: alternating n_indexes, each λ/4/n thick from x_start)."""
        lam = float(obj["lam"])
        indices = obj.get("n_indexes", obj.get("indexes", [1.0, 1.0]))
        n_lays = obj.get("n_layers_resolved") or (
            int(obj["n_layers"]) if "n_layers" in obj
            else _mirror_n_layers(float(obj["transmission"]), indices))
        x = obj["x_start_meep"]; out = []
        for k in range(n_lays):
            n = float(indices[k % 2]); lw = lam / 4.0 / n
            out.append((x, x + lw, n)); x += lw
        return out

    # ---------------- Sources ----------------

    def _build_sources(self):
        """Build CW plane sources from each `class: source` object."""
        for obj in self.objects_args:
            if obj.get("class") != "source":
                continue
            comp = obj.get("component", "Ey")
            lam = float(obj["lam"])
            f0 = 1.0 / lam
            amp_raw = obj.get("amplitude", 1.0)
            # per-run override (basis/forward sweeps) keyed by source object key
            if self.amp_override and obj.get("_key") in self.amp_override:
                amp_raw = self.amp_override[obj["_key"]]
            x_meep = obj["center_x_meep"]
            sy = obj["size_y_meep"]
            i_src = _meep_to_grid_x(x_meep, self.cx, self.dx)
            j_lo, j_hi = _meep_to_grid_y_range(0.0, sy, self.cy, self.dx)

            amp_1d = np.zeros(self.Ny, dtype=np.float64)
            if isinstance(amp_raw, list):
                n = len(amp_raw)
                edges = np.linspace(j_lo, j_hi, n + 1).astype(int)
                for p, a in enumerate(amp_raw):
                    amp_1d[edges[p]:edges[p + 1]] = float(a)
            else:
                amp_1d[j_lo:j_hi] = float(amp_raw)

            print(f"Source {obj['_key']}: component={comp}, x=i_src={i_src}, "
                  f"y∈[{j_lo},{j_hi}], f0={f0}")
            self.sources.append(
                f2.PlaneSource2D(
                    axis=0, index=i_src, component=comp,
                    amplitude_1d=jnp.asarray(amp_1d, dtype=_JDTYPE),
                    frequency=f0,
                )
            )

    # ---------------- Monitors ----------------

    def _build_monitors(self):
        """Build single-frequency DFT monitors from each `class: monitor` obj.
        Supports 'flux', '1Ddft' (x-normal plane), and '2Ddft' (full-grid DFT
        over a [i_lo:i_hi, j_lo:j_hi] box, accumulated in the time loop).
        2Ddft monitors set `is_2d=True` and `state2d` to a (4, Nx, Ny) tuple of
        (rE, iE, rH, iH) accumulators — the loop updates them with cos/sin
        weighting per step; the cropped slice is extracted in _save_monitor.
        """
        for obj in self.objects_args:
            if obj.get("class") != "monitor":
                continue
            x_meep = obj["center_x_meep"]
            sy = obj["size_y_meep"]
            mtype = obj.get("type", "1Ddft")
            i_mon = _meep_to_grid_x(x_meep, self.cx, self.dx)
            j_lo, j_hi = _meep_to_grid_y_range(0.0, sy, self.cy, self.dx)
            # 2Ddft: optional x-span from position.size[0] (default = full reservoir along x)
            sx = 0.0
            if mtype == "2Ddft":
                pos = obj.get("position", {})
                size_raw = pos.get("size", [])
                if isinstance(size_raw, list) and len(size_raw) >= 2 and float(size_raw[1]) > 0:
                    sx = float(size_raw[0])
                if sx > 0:
                    i_lo = _meep_to_grid_x(x_meep - sx / 2.0, self.cx, self.dx)
                    i_hi = _meep_to_grid_x(x_meep + sx / 2.0, self.cx, self.dx)
                else:
                    i_lo = 0; i_hi = self.Nx
                i_lo = max(0, i_lo); i_hi = min(self.Nx, i_hi)
            else:
                i_lo = i_mon; i_hi = i_mon + 1

            lam_range = obj.get("lam_range", [0.5, 0.5])
            n_lam = int(obj.get("n_lam", 1))
            if n_lam == 1:
                freqs = np.array([1.0 / lam_range[0]])
            else:
                # MEEP (class_sensor) orders DFT freqs as linspace(1/lam_hi, 1/lam_lo);
                # _run_basis reads index 0 → f = 1/lam_range[1]. Match that ordering so
                # the GPU monitor extracts the SAME frequency MEEP saves.
                freqs = np.linspace(1.0 / lam_range[1], 1.0 / lam_range[0], n_lam)
            f0 = float(freqs[0])

            mon: dict = {
                "key": obj["_key"], "type": mtype,
                "i_mon": i_mon, "i_lo": i_lo, "i_hi": i_hi,
                "j_lo": j_lo, "j_hi": j_hi, "freqs": freqs,
            }
            if mtype == "2Ddft":
                # Full-grid DFT accumulator: (rEx, iEx, rEy, iEy, rHz, iHz) all (Nx, Ny).
                z = jnp.zeros((self.Nx, self.Ny), dtype=_JDTYPE)
                mon["is_2d"] = True
                mon["omega"] = 2.0 * float(np.pi) * f0
                mon["state"] = (z, z, z, z, z, z)
                # No updater fn — the run_loop has a special branch for is_2d entries.
                mon["updater"] = None
                print(f"Monitor {obj['_key']} (2Ddft): i∈[{i_lo},{i_hi}] j∈[{j_lo},{j_hi}], f0={f0}")
            else:
                # collocate=True → reproduce MEEP's yee_grid=False readout: collocate
                # to the cell centre AND interpolate to the EXACT continuous monitor x
                # (MEEP reports the field at the requested center, not a snapped grid
                # point). frac = fractional cell-centre position of the monitor; =0.5
                # when it lands on a grid face (integer-pixel cell), ≠0.5 otherwise
                # (non-integer cell → carries the sub-pixel position, fixes the phase).
                fc = (x_meep + self.cx) / self.dx - 0.5
                frac = float(fc - np.floor(fc))
                updater = f2.make_dft_updater_2d(axis=0, index=i_mon, frequency=f0,
                                                 dt=self.dt, collocate=True, frac=frac)
                mon["is_2d"] = False
                mon["updater"] = updater
                mon["state"] = f2.make_dft_state_2d(self.grid, axis=0)
                print(f"Monitor {obj['_key']}: x=i_mon={i_mon}, y∈[{j_lo},{j_hi}], "
                      f"type={mtype}, f0={f0}")
            self.monitors.append(mon)

    # ---------------- STED gain (full-vector 2D) ----------------

    def _sted_args(self):
        """Return the reservoir.sted dict if gain is enabled, else None."""
        res = next((o for o in self.objects_args
                    if o.get("class") in ("reservoir", "voltage_reservoir")), None)
        s = res.get("sted") if res else None
        return (res, s) if (s and s.get("enabled", False)) else (res, None)

    def _build_gain(self):
        """Build the 4-level MultilevelAtom gain state over the reservoir region —
        the SAME atom MEEP builds in class_reservoir._build_sted_susceptibilities."""
        import multilevel as ml
        res, s = self._sted_args()
        if s is None:
            self.gain = None
            return
        lbdA = float(s["lbdA"]); gammaA = float(s["gammaA"])
        lbdE = float(s["lbdE"]); gammaE = float(s["gammaE"])
        SGMA = float(s["SGMA"]); N1_0 = float(s["N1_0"]); N3_0 = float(s.get("N3_0", 0.0))
        r43 = float(s.get("rate_43", 10.0)); r21 = float(s.get("rate_21", 100.0))
        transitions = [
            ml.Transition(1, 4, frequency=1.0 / lbdA, gamma=gammaA, sigma=1.0),
            ml.Transition(4, 3, transition_rate=r43),
            ml.Transition(2, 3, frequency=1.0 / lbdE, gamma=gammaE, sigma=1.0),
            ml.Transition(2, 1, transition_rate=r21),
        ]
        atom = ml.MultilevelAtom(4, transitions, [N1_0, 0.0, N3_0, 0.0],
                                 sigma=SGMA, sigma_diag=(1.0, 1.0, 1.0))
        coeffs = ml.build_coeffs(atom, self.dt)
        # gain mask = reservoir box at nodes (i,j)
        res_x_lo = res["edge_x_meep"]; res_x_hi = res_x_lo + float(res["size_x"])
        sizes = res.get("sizes")
        res_y = (float(sizes[1]) if isinstance(sizes, list) and len(sizes) > 1
                 else self.cell_y)
        i = np.arange(self.Nx); j = np.arange(self.Ny)
        X = (i * self.dx - self.cx)[:, None]
        Y = (j * self.dx - self.cy)[None, :]
        mask = ((X >= res_x_lo) & (X < res_x_hi) & (np.abs(Y) <= res_y / 2)).astype(np.float64)
        state = ml.init_state_full(atom, coeffs, (self.Nx, self.Ny), jnp.asarray(mask))
        self.gain = {"coeffs": coeffs, "state": state, "atom": atom}
        print(f"STED gain: {int(mask.sum())} nodes, N1_0={N1_0} N3_0={N3_0}, "
              f"pump {1/lbdA:.3f} emit {1/lbdE:.3f}, dt={self.dt:.5f}")

    def _build_sources_sted(self):
        """Build pulsed current sources (signal Ey plane + pump Ez area) matching
        MEEP's `pulsed` GaussianSource (fs → MEEP units) with current semantics."""
        _FS = 3.335640952
        self.sources = []
        if isinstance(self.material, f2.AnisoFull2D):
            inv_yy = np.array(self.material.iyy_Ey)
            inv_xx = np.array(self.material.ixx_Ex)
        else:
            inv_yy = np.array(self.material.eps_inv_yy_Ey)
            inv_xx = np.array(self.material.eps_inv_xx_Ex)
        inv_zz = np.array(self.eps_inv_zz)
        for obj in self.objects_args:
            if obj.get("class") != "source":
                continue
            comp = obj.get("component", "Ey")
            f0 = 1.0 / float(obj["lam"])
            amp_raw = obj.get("amplitude", 1.0)
            if self.amp_override and obj.get("_key") in self.amp_override:
                amp_raw = self.amp_override[obj["_key"]]
            stype = obj.get("source_type", "continuous")
            if stype == "pulsed":
                fwhm = float(obj.get("pulse_fwhm_fs", 1309.0))
                delay = float(obj.get("pulse_delay_fs", 0.0))
                width = (fwhm / _FS) / 2.35482
                start = delay / _FS if delay > 0 else 0.0
            elif stype == "gaussian":
                dlam = float(obj.get("dlam", 0.0))
                fwidth = ((1.0 / (float(obj["lam"]) - dlam) - 1.0 / (float(obj["lam"]) + dlam))
                          if dlam > 0 else 0.2 * f0)
                width = 1.0 / (2.0 * np.pi * fwidth) if fwidth > 0 else 0.0
                start = 0.0
            else:
                width = 0.0; start = 0.0
            x_meep = obj["center_x_meep"]; sy = obj["size_y_meep"]
            pos = obj.get("position", {})
            size_raw = pos.get("size", []) if isinstance(pos, dict) else []
            sx = (float(size_raw[0]) if isinstance(size_raw, list) and len(size_raw) >= 1
                  else 0.0)
            # Per-component Yee sample offsets (vec.hpp:1132 iyee_shift):
            #   Ey@(i, j+½), Ex@(i+½, j), Ez node@(i, j).
            if comp == "Ex":
                xoff, yoff = 0.5, 0.0
            elif comp == "Ez":
                xoff, yoff = 0.0, 0.0
            else:                                        # Ey
                xoff, yoff = 0.0, 0.5
            yoff = float(os.environ.get("GPUMEEP_SRC_YOFF", yoff))
            # y-profile: amplitude value per sample × MEEP fractional cell weight
            # (source spans [-sy/2, +sy/2], centered at y=0).
            ycen = (np.arange(self.Ny) + yoff) * self.dx - self.cy
            aprof = np.zeros(self.Ny)
            if isinstance(amp_raw, (list, tuple)) and len(amp_raw) > 1:
                seg = sy / len(amp_raw)
                for p, a in enumerate(amp_raw):
                    m = (ycen >= -sy / 2.0 + p * seg) & (ycen < -sy / 2.0 + (p + 1) * seg)
                    aprof[m] = float(np.real(a))
            else:
                a = amp_raw[0] if isinstance(amp_raw, (list, tuple)) else amp_raw
                aprof[:] = float(np.real(a))
            wy = _src_overlap_weights(-sy / 2.0, sy / 2.0, self.Ny, self.dx, yoff, self.cy)
            strip = aprof * wy                            # MEEP-exact y cell weights
            # x-profile: delta (plane) → linear interp summing to 1; area → overlap
            if sx > 0:
                wx = _src_overlap_weights(x_meep - sx / 2.0, x_meep + sx / 2.0,
                                          self.Nx, self.dx, xoff, self.cx)
            else:
                wx = _src_delta_weights(x_meep, self.Nx, self.dx, xoff, self.cx)
            amp_map = wx[:, None] * strip[None, :]
            eps_inv_map = (inv_zz if comp == "Ez" else inv_yy if comp == "Ey"
                           else inv_xx if comp == "Ex" else np.ones((self.Nx, self.Ny)))
            # MEEP delta-function source normalization (sources.cpp:483
            # `data.amp *= gv.a` = one factor of resolution per ZERO-size source
            # direction). No empirical constant: the absolute scale is carried by
            # MEEP's dipole amp 1/(-2πif) in _J and the DFT dt/√(2π) at readout.
            #   PLANE/line src (sx=0): 1 delta dir (x)  → gv.a = res
            #   AREA/volume src (sx>0): 0 delta dirs     → 1  (res·dx cancels)
            # GPUMEEP_SRC_C / GPUMEEP_SRC_POW kept only as diagnostic overrides.
            C = float(os.environ.get("GPUMEEP_SRC_C", "1.0"))
            pw = float(os.environ.get("GPUMEEP_SRC_POW", "1"))
            src_scale = C * (self.resolution ** pw)
            if sx > 0:
                src_scale *= self.dx        # 0 delta dirs → gv.a^0 = 1 (res·dx)
            self.sources.append(_STEDSource(
                component=comp, amp_map=jnp.asarray(amp_map, _JDTYPE),
                eps_inv_map=jnp.asarray(eps_inv_map, _JDTYPE),
                freq=f0, width=width, start_time=start, cutoff=5.0, dt=self.dt,
                src_scale=src_scale))
            print(f"STED src {obj['_key']}: comp={comp} sx={sx} "
                  f"x={x_meep:.3f} sy={sy:.3f} yoff={yoff} Σwx={float(wx.sum()):.4f} "
                  f"Σwy={float(wy.sum()):.3f} width={width:.2f} start={start:.2f} f0={f0:.3f}")

    def _build_pml_full(self):
        n = int(round(float(self.args.get("pml_size", 2.0)) / self.dx))
        self.pml = f2.make_cpml_full_2d(self.grid, self.dt, n_pml=(n, n))

    def _run_2d_sted(self):
        """Full-vector 2D run with STED gain: Ez pump inverts the medium that
        amplifies the Ey signal, both coupled through the atomic populations.
        Reproduces the MEEP STED-resonator forward run on GPU (differentiable)."""
        self._build_material()          # sets material, eps_inv_zz, _n_max (incl mirrors)
        # MEEP convention: dt = Courant·Δx (Δx = 1/resolution), with NO n_max
        # tightening. High ε only slows waves (v=c/n) — it does not tighten the
        # CFL bound, which is a property of the vacuum wave operator. Matching
        # MEEP's dt exactly is essential for reproducing the cavity round-trip
        # phase (numerical dispersion) in resonant DBR geometries.
        courant = float(self.args.get("courant", 0.5))
        self.dt = courant * self.dx
        print(f"dt = {self.dt} (MEEP-matched, Courant={courant}; n_max={self._n_max:.3f} unused for dt)")
        self._build_pml_full()
        self._build_gain()              # coeffs depend on dt
        self._build_sources_sted()
        self._build_monitors()

        run_until = float(self.run_until_override if self.run_until_override
                          else self.args.get("run_until", 500.0))
        # MEEP's driver runs `until=run_until` then, sources off, `until=50` more,
        # with the DFT sensors accumulating throughout (class_simulation.py:342-345)
        # → it integrates [0, run_until+decay]. Match that window exactly; otherwise
        # for a high-Q cavity (config 4) whose field is still ringing at run_until,
        # gpu's [0, run_until] slice differs from MEEP's [0, run_until+50].
        # (The pulsed source is long dead by run_until, so extending steps ≡ MEEP's
        # source-off decay phase.) NOTE: for high-Q cavities run_until itself must be
        # large enough for the ring-down to decay, or BOTH engines stay unconverged.
        decay = float(self.args.get("source_off_decay", 50.0))
        n_total = int((run_until + decay) / self.dt)
        print(f"run_until = {run_until} + decay {decay}, n_total = {n_total} steps")

        grid = self.grid; dt = self.dt; material = self.material; ezz = self.eps_inv_zz
        sources = self.sources
        has_gain = self.gain is not None
        coeffs = self.gain["coeffs"] if has_gain else None
        _nogain = bool(os.environ.get("GPUMEEP_NOGAIN")) or not has_gain
        if _nogain:
            print("Full-vector run: gain OFF (passive)" if not has_gain
                  else "DIAG: gain DISABLED (passive full-vector run)")
        is_2d_flags = [m["is_2d"] for m in self.monitors]
        omegas_2d = [m["omega"] if m["is_2d"] else 0.0 for m in self.monitors]
        updaters_1d = [m["updater"] for m in self.monitors]

        def apply_sources_D(D, t):
            # MEEP D-form: a current source is added to the displacement field D
            # (no ε⁻¹ — ε⁻¹ is applied later in E = ε⁻¹·(D−ΣP)).
            for s in sources:
                D = s.apply_D(D, t)
            return D

        def _update_one_mon(idx, mon_state, fields, t):
            if is_2d_flags[idx]:
                rEx, iEx, rEy, iEy, rHz, iHz = mon_state
                c = jnp.cos(omegas_2d[idx] * t); s = jnp.sin(omegas_2d[idx] * t)
                # MEEP e^{+iωt} convention (Im = +Σ f sin); H staggered by −dt/2
                # (Yee E/H time offset, dft.cpp:253) for correct E×H relative phase.
                cH = jnp.cos(omegas_2d[idx] * (t - 0.5 * dt))
                sH = jnp.sin(omegas_2d[idx] * (t - 0.5 * dt))
                return (rEx + c * fields.Ex, iEx + s * fields.Ex,
                        rEy + c * fields.Ey, iEy + s * fields.Ey,
                        rHz + cH * fields.Hz, iHz + sH * fields.Hz)
            return updaters_1d[idx](mon_state, fields, t)

        @jax.jit
        def run_loop(D, fields, pml_state, ml_state, mon_states, n_steps):
            # MEEP D-form leapfrog per step:
            #   (1) P^{n+1} from Eⁿ (gain), (2) D^{n+1}=Dⁿ+dt·∇×H − dt·J (source
            #   current already folded into D), (3) E^{n+1}=ε⁻¹(D^{n+1}−ΣP^{n+1}),
            #   (4) H^{n+3/2}. D is the primary integrated field; E is derived.
            def body(i, state):
                D, f, p, ml, ms = state
                t = i * dt
                D = apply_sources_D(D, t)
                if _nogain:
                    D, f, p = f2.step_2d_full_dform(D, f, grid, dt, p, material)
                else:
                    D, f, p, ml = f2.step_2d_full_gain_dform(
                        D, f, grid, dt, p, material, ml, coeffs)
                # DFT time-labeling: after the step, f holds E^{n+1}, which MEEP
                # references at (n+1)·dt (step.cpp does `t += 1` THEN update_dfts()).
                # Label it t+dt to match MEEP; the H stagger inside picks up
                # (t+dt)−0.5·dt = t+0.5·dt (H^{n+1/2}). Paired with the D-source
                # current at time()+0.5·dt, this reproduces MEEP's phase.
                ms = [_update_one_mon(k, m, f, t + dt) for k, m in enumerate(ms)]
                return (D, f, p, ml, ms)
            return jax.lax.fori_loop(0, n_steps, body, (D, fields, pml_state, ml_state, mon_states))

        fields = f2.zero_fields_full(self.grid)
        D_state = f2.zero_D_full(self.grid)
        pml_state = self.pml
        ml_state = self.gain["state"] if has_gain else jnp.zeros(())
        mon_states = [m["state"] for m in self.monitors]

        t0 = time.time()
        D_state, fields, pml_state, ml_state, mon_states = run_loop(
            D_state, fields, pml_state, ml_state, mon_states, n_total)
        fields.Ey.block_until_ready()
        print(f"STED run finished in {time.time()-t0:.1f} s ({n_total} steps)")

        if os.environ.get("GPUMEEP_DIAG") and has_gain:
            Nf = np.asarray(ml_state.N)                 # (n_levels, Nx, Ny)
            gm = np.asarray(self.gain["state"].mask) > 1e-6
            for l in range(Nf.shape[0]):
                v = Nf[l][gm]
                print(f"DIAG N[{l}] (gain region): mean={v.mean():.4f} max={v.max():.4f}")
            inv = Nf[2][gm] - Nf[1][gm]                 # emission inversion N3-N2 (2→3)
            print(f"DIAG emission inversion N3-N2: mean={inv.mean():.4f} max={inv.max():.4f}")
            print(f"DIAG final max|Ey|={float(jnp.max(jnp.abs(fields.Ey))):.4g} "
                  f"max|Ez|={float(jnp.max(jnp.abs(fields.Ez))):.4g} "
                  f"max|Hz|={float(jnp.max(jnp.abs(fields.Hz))):.4g}")

        # MEEP DFT convention (dft.cpp:225): dft += f·e^{iωt}·dt_factor,
        # dt_factor = dt/√(2π)·decimation (decimation=1 here). Copy it EXACTLY —
        # the missing 1/√(2π) was the √(2π) half of the old empirical src constant.
        dft_scale = dt / np.sqrt(2.0 * np.pi)
        for m, st in zip(self.monitors, mon_states):
            if m["is_2d"]:
                rEx, iEx, rEy, iEy, rHz, iHz = st
                amps = {"Ex": (np.asarray(rEx) + 1j * np.asarray(iEx)) * dft_scale,
                        "Ey": (np.asarray(rEy) + 1j * np.asarray(iEy)) * dft_scale,
                        "Hz": (np.asarray(rHz) + 1j * np.asarray(iHz)) * dft_scale}
            else:
                re_Ex, im_Ex, re_Ey, im_Ey, re_Hz, im_Hz = st
                amps = {"Ex": (np.asarray(re_Ex) + 1j * np.asarray(im_Ex)) * dft_scale,
                        "Ey": (np.asarray(re_Ey) + 1j * np.asarray(im_Ey)) * dft_scale,
                        "Hz": (np.asarray(re_Hz) + 1j * np.asarray(im_Hz)) * dft_scale}
            self._save_monitor(m, amps)

    # ---------------- Run ----------------

    def _build_pml(self):
        n_pml_cells = int(round(float(self.args.get("pml_size", 2.0)) / self.dx))
        self.pml = f2.make_cpml_2d(self.grid, self.dt,
                                    n_pml=(n_pml_cells, n_pml_cells))

    def run_basis(self, amplitude_list, source_key=None):
        """One forward run with a given SIGNAL amplitude → complex (Ey, Ex, Ez) at
        monitor_2. GPUmeep analogue of SimulationT._run_basis so open_reservoir()
        can dispatch to either engine. Works for 2D and 3D (run() saves monitor_2.npz
        in the same MEEP-compatible schema; we read it back)."""
        if source_key is None:
            # first 'source' object whose component is NOT the STED pump (Ez area)
            self._set_data(); self._update_all_args()
            source_key = next(o["_key"] for o in self.objects_args
                              if o.get("class") == "source"
                              and o.get("_key") != "source_2")
        self.amp_override = {source_key: list(amplitude_list)}
        self.run()
        m2 = np.load(os.path.join(self.paths["simulation"], "monitor_2.npz"))
        Ey = m2["Ey"]; Ex = m2["Ex"]; Ez = m2["Ez"]
        return np.asarray(Ey).ravel(), np.asarray(Ex).ravel(), np.asarray(Ez).ravel()

    def run(self):
        self._set_data()
        self._update_all_args()
        if self.dim >= 3:
            return self._run_3d()
        # STED gain resonator (or forced) → full-vector path (Ez pump + gain + mirrors).
        # force_fullvector routes non-dye configs through the same fixed source/DFT path.
        if not self.empty and (self._sted_args()[1] is not None
                               or getattr(self, "force_fullvector", False)):
            return self._run_2d_sted()
        self._build_material()
        self._build_sources()
        self._build_monitors()

        # MEEP convention: dt = Courant·Δx, no n_max tightening (see _run_2d_sted).
        courant = float(self.args.get("courant", 0.5))
        self.dt = courant * self.dx
        print(f"dt = {self.dt} (MEEP-matched, Courant={courant})")
        self._build_pml()

        run_until = float(self.args.get("run_until", 500.0))
        n_total = int(run_until / self.dt)
        print(f"run_until = {run_until}, n_total = {n_total} steps")

        # Build the time-loop body
        sources = self.sources
        grid = self.grid; dt = self.dt; material = self.material
        # Static lists captured by closure (one per monitor, by index).
        is_2d_flags = [m["is_2d"] for m in self.monitors]
        omegas_2d = [m["omega"] if m["is_2d"] else 0.0 for m in self.monitors]
        updaters_1d = [m["updater"] for m in self.monitors]

        def apply_sources(fields, t):
            for s in sources:
                fields = s.apply(fields, t)
            return fields

        def _update_one_mon(idx, mon_state, fields, t):
            if is_2d_flags[idx]:
                rEx, iEx, rEy, iEy, rHz, iHz = mon_state
                c = jnp.cos(omegas_2d[idx] * t); s = jnp.sin(omegas_2d[idx] * t)
                # H staggered by −dt/2 (Yee E/H time offset, MEEP dft.cpp:253)
                cH = jnp.cos(omegas_2d[idx] * (t - 0.5 * self.dt))
                sH = jnp.sin(omegas_2d[idx] * (t - 0.5 * self.dt))
                rEx = rEx + c * fields.Ex; iEx = iEx + s * fields.Ex
                rEy = rEy + c * fields.Ey; iEy = iEy + s * fields.Ey
                rHz = rHz + cH * fields.Hz; iHz = iHz + sH * fields.Hz
                return (rEx, iEx, rEy, iEy, rHz, iHz)
            else:
                u = updaters_1d[idx]
                return u(mon_state, fields, t)

        @jax.jit
        def run_loop(fields, pml_state, mon_states, n_steps):
            def body(i, state):
                f, p, ms = state
                t = i * dt
                f = apply_sources(f, t)
                f, p = f2.step_2d(f, grid, dt, p, material)
                ms = [_update_one_mon(k, m, f, t) for k, m in enumerate(ms)]
                return (f, p, ms)
            return jax.lax.fori_loop(0, n_steps, body, (fields, pml_state, mon_states))

        fields = self.grid.zero_fields()
        pml_state = self.pml
        mon_states = [m["state"] for m in self.monitors]

        t0 = time.time()
        fields, pml_state, mon_states = run_loop(fields, pml_state, mon_states, n_total)
        fields.Ey.block_until_ready()
        print(f"Run finished in {time.time()-t0:.1f} s ({n_total} steps)")

        # Save monitor outputs
        for m, st in zip(self.monitors, mon_states):
            if m["is_2d"]:
                # Convert accumulators to complex + apply standard 2/N scale (same as 1D).
                rEx, iEx, rEy, iEy, rHz, iHz = st
                scale = 2.0 / n_total
                amps = {
                    "Ex": (np.asarray(rEx) + 1j * np.asarray(iEx)) * scale,
                    "Ey": (np.asarray(rEy) + 1j * np.asarray(iEy)) * scale,
                    "Hz": (np.asarray(rHz) + 1j * np.asarray(iHz)) * scale,
                }
            else:
                amps = f2.extract_complex_2d(st, n_total)
            self._save_monitor(m, amps)

    def _save_monitor(self, mon, amps):
        out_path = os.path.join(self.paths["simulation"], f"{mon['key']}.npz")
        if mon["type"] == "flux":
            # Time-averaged S_x from the accumulated TE fields: ½·Re(Ey·conj(Hz)).
            # (mon3d.poynting_density_x isn't available in the 2D `monitors` module.)
            Ey = np.asarray(amps["Ey"]); Hz = np.asarray(amps["Hz"])
            Sx = 0.5 * np.real(Ey * np.conj(Hz))
            flux = float(np.sum(Sx[mon["j_lo"]:mon["j_hi"]]) * self.dx)
            np.savez(out_path,
                     freqs=mon["freqs"], fluxes=np.array([flux]))
            print(f"Saved {out_path}: flux={flux:.4g}")
        else:
            # DFT monitor: save Ex, Ey, Ez complex arrays cropped to monitor span.
            # 2Ddft: save the (i_lo:i_hi, j_lo:j_hi) box. 1Ddft: just the j-strip.
            if mon["type"] == "2Ddft":
                Ex = np.asarray(amps["Ex"])[mon["i_lo"]:mon["i_hi"],
                                            mon["j_lo"]:mon["j_hi"]]
                Ey = np.asarray(amps["Ey"])[mon["i_lo"]:mon["i_hi"],
                                            mon["j_lo"]:mon["j_hi"]]
                Ez = np.zeros_like(Ex)
                np.savez(out_path,
                         Ex=Ex[None, :, :], Ey=Ey[None, :, :], Ez=Ez[None, :, :],
                         freqs=mon["freqs"],
                         i_lo=mon["i_lo"], i_hi=mon["i_hi"],
                         j_lo=mon["j_lo"], j_hi=mon["j_hi"])
                print(f"Saved {out_path}: 2D shape {Ex.shape}")
                return
            Ex = np.array(amps["Ex"])[mon["j_lo"]:mon["j_hi"]]
            Ey = np.array(amps["Ey"])[mon["j_lo"]:mon["j_hi"]]
            Ez = np.zeros_like(Ex)
            # MEEP saves shape (n_freq, n_y_samples). For single freq:
            np.savez(out_path,
                     Ex=Ex[None, :], Ey=Ey[None, :], Ez=Ez[None, :],
                     freqs=mon["freqs"])
            print(f"Saved {out_path}: shape {Ex[None, :].shape}")


    # ================================================================
    # 3D path (dimention == 3): fdtd_core + pml + materials/sources/monitors
    # ================================================================

    def _build_material_3d(self):
        """Build a full 3D Anisotropic eps tensor. Vacuum everywhere except the
        reservoir block, which uses the relaxed 3D LC director (phi, theta)."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        if self.empty:
            ones = jnp.ones((Nx, Ny, Nz), dtype=_JDTYPE)
            zeros = jnp.zeros((Nx, Ny, Nz), dtype=_JDTYPE)
            self.material = mats.anisotropic_from_tensor(
                ones, ones, ones, zeros, zeros, zeros)
            return

        res_args = next((o for o in self.objects_args
                         if o.get("class") in ("reservoir", "voltage_reservoir")), None)
        if res_args is None:
            ones = jnp.ones((Nx, Ny, Nz), dtype=_JDTYPE)
            zeros = jnp.zeros((Nx, Ny, Nz), dtype=_JDTYPE)
            self.material = mats.anisotropic_from_tensor(
                ones, ones, ones, zeros, zeros, zeros)
            return

        n_o = float(res_args.get("n_o", 1.52))
        n_e = float(res_args.get("n_e", 1.71))
        sizes = res_args["sizes"]   # [size_x, size_y, size_z]
        res_y = float(sizes[1]); res_z = float(sizes[2])
        # Reservoir extents in grid coords (origin at 0)
        res_x_lo = res_args["edge_x_meep"] + self.cell_x / 2
        res_x_hi = res_x_lo + float(res_args["size_x"])
        res_y_lo = (self.cell_y - res_y) / 2
        res_y_hi = res_y_lo + res_y
        res_z_lo = (self.cell_z - res_z) / 2
        res_z_hi = res_z_lo + res_z

        lc_path = os.path.join(self.folder_path, "simulation", "lc_fields.npz")
        if not os.path.exists(lc_path):
            raise FileNotFoundError(
                f"Need relaxed LC field at {lc_path}. "
                f"Run `python class_reservoir.py --path {self.folder_path}` first.")
        lc = np.load(lc_path)
        phi_lc = np.asarray(lc["phi"])      # (nx, ny, nz)
        theta_lc = np.asarray(lc["theta"])
        lc_x = np.asarray(lc["x"]); lc_y = np.asarray(lc["y"]); lc_z = np.asarray(lc["z"])

        # 3D interpolation onto the reservoir sub-grid (integer voxel centers)
        from scipy.interpolate import RegularGridInterpolator
        lc_x_g = (lc_x - lc_x.min()) + res_x_lo
        lc_y_g = (lc_y - lc_y.min()) + res_y_lo
        lc_z_g = (lc_z - lc_z.min()) + res_z_lo
        interp_phi = RegularGridInterpolator((lc_x_g, lc_y_g, lc_z_g), phi_lc,
                                             bounds_error=False, fill_value=None)
        interp_theta = RegularGridInterpolator((lc_x_g, lc_y_g, lc_z_g), theta_lc,
                                               bounds_error=False, fill_value=None)

        # Project-specific samplers for the LC director and reservoir block
        # mask. The half-cell Yee shift lives in src (`mats.sample_yee`),
        # so the driver only describes WHAT to sample, not WHERE.

        def _interp(interp, xs, ys, zs):
            Xc = np.clip(xs, lc_x_g[0], lc_x_g[-1])
            Yc = np.clip(ys, lc_y_g[0], lc_y_g[-1])
            Zc = np.clip(zs, lc_z_g[0], lc_z_g[-1])
            I, J, K = np.meshgrid(Xc, Yc, Zc, indexing="ij")
            pts = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1)
            return interp(pts).reshape(len(xs), len(ys), len(zs))

        def director_phi(xs, ys, zs):   return _interp(interp_phi,   xs, ys, zs)
        def director_theta(xs, ys, zs): return _interp(interp_theta, xs, ys, zs)

        def reservoir_mask(xs, ys, zs):
            return ((xs[:, None, None] >= res_x_lo) & (xs[:, None, None] < res_x_hi)
                    & (ys[None, :, None] >= res_y_lo) & (ys[None, :, None] < res_y_hi)
                    & (zs[None, None, :] >= res_z_lo) & (zs[None, None, :] < res_z_hi))

        phi_Ex,   phi_Ey,   phi_Ez   = mats.sample_yee(self.grid, director_phi)
        theta_Ex, theta_Ey, theta_Ez = mats.sample_yee(self.grid, director_theta)
        mask_Ex,  mask_Ey,  mask_Ez  = mats.sample_yee(self.grid, reservoir_mask)

        self.material = mats.anisotropic3dyee_from_director_at_yee(
            n_o_sq=float(n_o) ** 2, n_e_sq=float(n_e) ** 2,
            theta_Ex=jnp.asarray(theta_Ex, dtype=_JDTYPE),
            phi_Ex  =jnp.asarray(phi_Ex,   dtype=_JDTYPE),
            mask_Ex =jnp.asarray(mask_Ex),
            theta_Ey=jnp.asarray(theta_Ey, dtype=_JDTYPE),
            phi_Ey  =jnp.asarray(phi_Ey,   dtype=_JDTYPE),
            mask_Ey =jnp.asarray(mask_Ey),
            theta_Ez=jnp.asarray(theta_Ez, dtype=_JDTYPE),
            phi_Ez  =jnp.asarray(phi_Ez,   dtype=_JDTYPE),
            mask_Ez =jnp.asarray(mask_Ez),
            S=1.0,
        )

    def _build_sources_3d(self):
        for obj in self.objects_args:
            if obj.get("class") != "source":
                continue
            comp = obj.get("component", "Ey")
            lam = float(obj["lam"])
            f0 = 1.0 / lam
            x_meep = obj["center_x_meep"]
            i_src = _meep_to_grid_x(x_meep, self.cx, self.dx)
            pos = obj.get("position", {})
            raw = pos.get("size", [self.cell_y, self.cell_z]) if isinstance(pos, dict) else [self.cell_y, self.cell_z]
            src_y = float(raw[0]) if raw and raw[0] else self.cell_y
            src_z = float(raw[1]) if len(raw) > 1 and raw[1] else self.cell_z
            j_lo, j_hi = _meep_to_grid_y_range(0.0, src_y, self.cy, self.dx)
            k_lo, k_hi = _meep_to_grid_y_range(0.0, src_z, self.cz, self.dx)

            amp_raw = obj.get("amplitude", 1.0)
            grid_shape = obj.get("grid_shape")
            amp_2d = np.zeros((self.Ny, self.Nz), dtype=np.float64)
            if isinstance(amp_raw, list) and grid_shape:
                gh, gw = grid_shape
                pattern = np.array(amp_raw, dtype=np.float64).reshape(gh, gw)
                # upscale the gh×gw pattern across the source yz window
                yi = ((np.arange(j_hi - j_lo)) * gh // max(j_hi - j_lo, 1)).astype(int)
                zi = ((np.arange(k_hi - k_lo)) * gw // max(k_hi - k_lo, 1)).astype(int)
                yi = np.clip(yi, 0, gh - 1); zi = np.clip(zi, 0, gw - 1)
                amp_2d[j_lo:j_hi, k_lo:k_hi] = pattern[yi[:, None], zi[None, :]]
            elif isinstance(amp_raw, list):
                # 1D strip pattern along y, uniform in z
                n = len(amp_raw)
                edges = np.linspace(j_lo, j_hi, n + 1).astype(int)
                for p, a in enumerate(amp_raw):
                    amp_2d[edges[p]:edges[p + 1], k_lo:k_hi] = float(a)
            else:
                amp_2d[j_lo:j_hi, k_lo:k_hi] = float(amp_raw)

            print(f"Source {obj['_key']}: comp={comp}, x={i_src}, "
                  f"y∈[{j_lo},{j_hi}], z∈[{k_lo},{k_hi}], f0={f0}")
            self.sources.append(src3d.PlaneSource(
                axis=0, index=i_src, component=comp,
                amplitude_2d=jnp.asarray(amp_2d, dtype=_JDTYPE), frequency=f0))

    def _build_monitors_3d(self):
        for obj in self.objects_args:
            if obj.get("class") != "monitor":
                continue
            x_meep = obj["center_x_meep"]
            i_mon = _meep_to_grid_x(x_meep, self.cx, self.dx)
            pos = obj.get("position", {})
            raw = pos.get("size", self.cell_y) if isinstance(pos, dict) else self.cell_y
            # MEEP convention (class_sensor): the monitor `size` sets the
            # y-extent; the z-extent ALWAYS spans the full cell_z.
            if isinstance(raw, (int, float)):
                size_y = float(raw) if raw else self.cell_y
            else:
                size_y = float(raw[0]) if raw and raw[0] else self.cell_y
            j_lo, j_hi = _meep_to_grid_y_range(0.0, size_y, self.cy, self.dx)
            k_lo, k_hi = 0, self.Nz   # full z-extent
            lam_range = obj.get("lam_range", [0.5, 0.5])
            f0 = 1.0 / lam_range[0]
            updater = mon3d.make_dft_updater(axis=0, index=i_mon, frequency=f0)
            print(f"Monitor {obj['_key']}: x={i_mon}, y∈[{j_lo},{j_hi}], "
                  f"z∈[{k_lo},{k_hi}], type={obj.get('type')}, f0={f0}")
            self.monitors.append({
                "key": obj["_key"], "type": obj.get("type", "2Ddft"),
                "i_mon": i_mon, "j_lo": j_lo, "j_hi": j_hi,
                "k_lo": k_lo, "k_hi": k_hi,
                "freqs": np.array([f0]),
                "updater": updater,
                "state": mon3d.make_dft_state(self.grid, axis=0),
            })

    def _run_3d(self):
        self._build_material_3d()
        self._build_sources_3d()
        self._build_monitors_3d()

        # Material-aware Courant from peak eps eigenvalue (use n_e as upper bound)
        n_max = 0.0
        res_args = next((o for o in self.objects_args if o.get("class") in ("reservoir", "voltage_reservoir")), None)
        if res_args and not self.empty:
            n_max = float(res_args.get("n_e", 1.71))
        n_max = max(n_max, 1.0)
        self.dt = float(fc.courant_dt(self.grid, safety=0.5)) / n_max
        print(f"dt = {self.dt} (n_max={n_max})")

        n_pml = int(round(float(self.args.get("pml_size", 2.0)) / self.dx))
        self.pml = pml3d.make_cpml(self.grid, self.dt, n_pml=n_pml)

        run_until = float(self.args.get("run_until", 300.0))
        n_total = int(run_until / self.dt)
        print(f"run_until = {run_until}, n_total = {n_total} steps")

        sources = self.sources
        updaters = [m["updater"] for m in self.monitors]
        grid = self.grid; dt = self.dt; material = self.material

        def apply_sources(fields, t):
            for s in sources:
                fields = s.apply(fields, t)
            return fields

        @jax.jit
        def run_loop(fields, pml_state, mon_states, n_steps):
            def body(idx, state):
                f, p, ms = state
                t = idx * dt
                f = apply_sources(f, t)
                f, p = pml3d.step_cpml_aniso(f, grid, dt, p, material)
                ms = [u(m, f, t) for u, m in zip(updaters, ms)]
                return (f, p, ms)
            return jax.lax.fori_loop(0, n_steps, body, (fields, pml_state, mon_states))

        fields = self.grid.zero_fields(dtype=_JDTYPE)
        pml_state = self.pml
        mon_states = [m["state"] for m in self.monitors]

        t0 = time.time()
        fields, pml_state, mon_states = run_loop(fields, pml_state, mon_states, n_total)
        fields.Ey.block_until_ready()
        print(f"Run finished in {time.time()-t0:.1f} s ({n_total} steps)")

        for m, st in zip(self.monitors, mon_states):
            amps = mon3d.extract_complex(st, n_total)
            self._save_monitor_3d(m, amps)

    def _save_monitor_3d(self, mon, amps):
        out_path = os.path.join(self.paths["simulation"], f"{mon['key']}.npz")
        jl, jh, kl, kh = mon["j_lo"], mon["j_hi"], mon["k_lo"], mon["k_hi"]
        if mon["type"] == "flux":
            # S_x density via library helper; integrate over the yz patch.
            Sx = mon3d.poynting_density_x(amps["Ey"], amps["Ez"],
                                          amps["Hy"], amps["Hz"])
            flux = np.sum(Sx[jl:jh, kl:kh]) * self.dx * self.dx
            np.savez(out_path, freqs=mon["freqs"], fluxes=np.array([flux]))
            print(f"Saved {out_path}: flux={flux:.4g}")
        else:
            # 2Ddft: save Ex/Ey/Ez over the yz monitor patch, shape (1, ny, nz)
            Ex = np.array(amps["Ex"])[jl:jh, kl:kh]
            Ey = np.array(amps["Ey"])[jl:jh, kl:kh]
            Ez = np.array(amps["Ez"])[jl:jh, kl:kh]
            np.savez(out_path,
                     Ex=Ex[None, :, :], Ey=Ey[None, :, :], Ez=Ez[None, :, :],
                     freqs=mon["freqs"])
            print(f"Saved {out_path}: shape {Ex[None, :, :].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="Path to simulation folder (containing simulation_data.json)")
    parser.add_argument("--empty", action="store_true",
                        help="Run with vacuum (no reservoir/SLM material) for reference flux")
    parser.add_argument("--precision", choices=["fp32", "fp64"], default="fp64",
                        help="Field precision (default fp64 matches MEEP; fp32 is faster on consumer GPUs)")
    args = parser.parse_args()
    # NOTE: --precision is consumed during module import (above) to decide JAX
    # x64 config before JAX initialises.
    sim = SimulationGPU(folder_path=args.path, empty=args.empty)
    sim.run()


if __name__ == "__main__":
    main()
