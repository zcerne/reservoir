"""GPUmeep replica of `class_simulation.py` from the reservoir project.

Reads a `simulation_data.json` describing the same geometry conventions as the
MEEP class_simulation, but runs the simulation with the GPUmeep JAX FDTD
engine instead of MEEP. Output files are written in the same npz format as the
MEEP version so existing post-processing (T-matrix assembly, etc.) just works.

Scope (initial port — extend as needed):
  * 2D simulations only (dimention == 2). 1D and 3D dispatch can be added.
  * Object types: guide, reservoir, source, monitor (type "flux" or "1Ddft").
  * Single CW source; the source amplitude can be a scalar or a 1D pattern.
  * Boundary: PML in x and y (matches MEEP default for non-periodic test2D).

Usage:
    python class_simulation_gpu.py --path data/test2D [--empty]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

# Decide precision before importing JAX. Default fp64 (matches MEEP); pass
# --precision fp32 on the CLI for 3-10x speed on consumer GPUs (RTX A1000).
# fp32 is fine for res ≤ 40 / shorter runs but can develop instability at
# higher resolutions due to round-off accumulation over many steps.
if "--precision" in sys.argv:
    _i = sys.argv.index("--precision")
    _prec = sys.argv[_i + 1]
else:
    _prec = "fp64"
import jax
if _prec == "fp64":
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
_JDTYPE = _JDTYPE if _prec == "fp64" else jnp.float32
import numpy as np

# Locate the GPUmeep `src/` directory containing fdtd_2d, materials, etc.
# Tries (in order): $GPUMEEP_PATH, ~/Nextcloud/Doktorski/Projects/GPUmeep/src,
# the script's own folder (development). On the cluster, set GPUMEEP_PATH
# to wherever GPUmeep is checked out.
_gpumeep_candidates = [
    os.environ.get("GPUMEEP_PATH"),
    os.path.expanduser("~/Nextcloud/Doktorski/Projects/GPUmeep/src"),
    os.path.expanduser("~/Projects/GPUmeep/src"),
    os.path.expanduser("~/GPUmeep/src"),
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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _pos_to_center_size_2d(pos, on_edge_x, on_size_x, cell_x, cell_y):
    """Translate a MEEP-style position dict to (center_x, size_y) in 2D.

    Position can be 'left', 'right', 'center' relative to the on_object.
    Returns x in MEEP coords (cell centered at origin) and the y span of the
    feature (full cell_y by default for monitors/sources without size).
    """
    if isinstance(pos, dict):
        label = pos.get("position", "center")
        raw = pos.get("size", [0.0, 0.0])
    else:
        label = str(pos) if pos else "center"
        raw = [0.0, 0.0]
    if isinstance(raw, (int, float)):
        raw = [float(raw), 0.0]
    sy = float(raw[0]) if raw[0] else cell_y
    x_meep = {"left": on_edge_x,
              "right": on_edge_x + on_size_x}.get(label, on_edge_x + on_size_x / 2)
    return x_meep, sy


def _meep_to_grid_x(x_meep, cell_x, dx):
    """Convert MEEP x coord (centered at 0) to GPUmeep grid index (origin at 0)."""
    return int(round((x_meep + cell_x / 2) / dx))


def _meep_to_grid_y_range(center_y, sy, cell_y, dx):
    """Convert MEEP y center+size to GPUmeep grid index range (origin at 0)."""
    y_meep_lo = center_y - sy / 2
    y_meep_hi = center_y + sy / 2
    j_lo = int(round((y_meep_lo + cell_y / 2) / dx))
    j_hi = int(round((y_meep_hi + cell_y / 2) / dx))
    return j_lo, j_hi


# ----------------------------------------------------------------------
# Main Simulation class
# ----------------------------------------------------------------------


@dataclass
class SimulationGPU:
    folder_path: str
    empty: bool = False     # if True, skip building reservoir/SLM materials

    args: dict = field(default_factory=dict)
    objects_args: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)

    resolution: int = 40
    cell_x: float = 0.0
    cell_y: float = 0.0
    dim: int = 2
    dx: float = 0.0
    Nx: int = 0
    Ny: int = 0
    grid: f2.Grid2D | None = None
    material: f2.Aniso2DYee | None = None
    pml: f2.CPML2D | None = None
    sources: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    dt: float = 0.0

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
        pml = float(self.args.get("pml_size", 2.0))
        self.dim = self.args.get("dimention", 1)
        self.resolution = int(self.args["resolution"])
        self.dx = 1.0 / self.resolution
        cell_y_arg = (float(self.args.get("cell_size_y", 0.0)) if self.dim > 1
                      else 4.0 / self.resolution)

        # Walk objects in declared order, accumulate x positions
        current_x = 0.0
        for key in self.args["object_order"]:
            obj = dict(self.args[key])
            obj["_key"] = key
            obj["edge_x_local"] = current_x
            if isinstance(obj.get("sizes"), list):
                obj["size_x"] = float(obj["sizes"][0])
            self.objects_args.append(obj)
            current_x += float(obj.get("size_x", 0.0))

        self.cell_x = current_x + 2 * pml
        self.cell_y = cell_y_arg
        x0 = -self.cell_x / 2 + pml   # left edge of first object in MEEP coords

        for obj in self.objects_args:
            edge_x = obj["edge_x_local"] + x0
            size_x = float(obj.get("size_x", 0.0))
            cls = obj.get("class", "")

            if cls == "guide" or cls == "reservoir":
                obj["center_x_meep"] = edge_x + size_x / 2
                obj["edge_x_meep"] = edge_x

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
        self.grid = f2.Grid2D(Nx=self.Nx, Ny=self.Ny, dx=self.dx, dy=self.dx)
        print(f"Cell = {self.cell_x} x {self.cell_y}, grid = {self.grid.shape}, dx = {self.dx}")

    # ---------------- Material ----------------

    def _build_material(self):
        """Build a single Aniso2DYee tensor covering the whole cell. Vacuum
        everywhere except inside reservoir/guide blocks. Reservoir uses the
        relaxed LC director field; guide is currently treated as vacuum (index=1).

        When self.empty is True, the reservoir block is replaced by vacuum
        (matches MEEP's empty reference run).
        """
        if self.empty:
            self._build_vacuum_material()
            return
        # Find reservoir object (only one supported here)
        res_args = next((o for o in self.objects_args
                         if o.get("class") == "reservoir"), None)
        if res_args is None:
            self._build_vacuum_material()
            return

        n_o = float(res_args.get("n_o", 1.5))
        n_e = float(res_args.get("n_e", 1.7))
        # Sizes in MEEP coords
        res_x_lo_meep = res_args["edge_x_meep"]
        res_x_hi_meep = res_x_lo_meep + float(res_args["size_x"])
        res_y_meep = float(res_args["sizes"][1]) if isinstance(res_args.get("sizes"), list) and len(res_args["sizes"]) > 1 else self.cell_y
        # In grid coords (origin at 0)
        res_x_lo = res_x_lo_meep + self.cell_x / 2
        res_x_hi = res_x_hi_meep + self.cell_x / 2
        res_y_lo = (self.cell_y - res_y_meep) / 2
        res_y_hi = res_y_lo + res_y_meep

        # Load relaxed LC field
        lc_path = os.path.join(self.folder_path, "simulation", "lc_fields.npz")
        if not os.path.exists(lc_path):
            raise FileNotFoundError(
                f"Need relaxed LC field at {lc_path}. "
                f"Run `python class_reservoir.py --path {self.folder_path}` first."
            )
        lc = np.load(lc_path)
        phi_full_lc = np.asarray(lc["phi"])
        lc_x = np.asarray(lc["x"])
        lc_y = np.asarray(lc["y"])
        # Take central z slice for 2D simulation
        phi_lc = phi_full_lc[:, :, phi_full_lc.shape[2] // 2]

        # Build phi at Ex and Ey Yee grids with area-fraction masks
        from scipy.interpolate import RectBivariateSpline
        # LC local coords + shifted into reservoir bounds
        lc_x_in_grid = (lc_x - lc_x.min()) + res_x_lo
        lc_y_in_grid = (lc_y - lc_y.min()) + res_y_lo
        interp_phi = RectBivariateSpline(lc_x_in_grid, lc_y_in_grid, phi_lc, kx=3, ky=3)

        def sample_yee(x_offset, y_offset, cell_half_x, cell_half_y):
            i_grid = np.arange(self.Nx)
            j_grid = np.arange(self.Ny)
            x_pos = i_grid * self.dx + x_offset
            y_pos = j_grid * self.dx + y_offset
            cell_xlo = x_pos - cell_half_x * self.dx
            cell_xhi = x_pos + cell_half_x * self.dx
            cell_ylo = y_pos - cell_half_y * self.dx
            cell_yhi = y_pos + cell_half_y * self.dx
            ox = np.clip(np.minimum(cell_xhi, res_x_hi) - np.maximum(cell_xlo, res_x_lo),
                         0.0, 2 * cell_half_x * self.dx) / (2 * cell_half_x * self.dx)
            oy = np.clip(np.minimum(cell_yhi, res_y_hi) - np.maximum(cell_ylo, res_y_lo),
                         0.0, 2 * cell_half_y * self.dx) / (2 * cell_half_y * self.dx)
            frac = (ox[:, None] * oy[None, :]).astype(np.float64)
            x_eval = np.clip(x_pos, lc_x_in_grid[0], lc_x_in_grid[-1])
            y_eval = np.clip(y_pos, lc_y_in_grid[0], lc_y_in_grid[-1])
            phi = interp_phi(x_eval, y_eval).astype(np.float64)
            return phi, frac

        phi_Ex, frac_Ex = sample_yee(0.5 * self.dx, 0.0, 0.5, 0.5)
        phi_Ey, frac_Ey = sample_yee(0.0, 0.5 * self.dx, 0.5, 0.5)

        eps_perp = n_o ** 2
        delta = n_e ** 2 - n_o ** 2

        def tensor_at(phi, frac):
            c = np.cos(phi); s = np.sin(phi)
            exx_lc = eps_perp + delta * c * c
            eyy_lc = eps_perp + delta * s * s
            exy_lc = delta * c * s
            exx = frac * exx_lc + (1.0 - frac) * 1.0
            eyy = frac * eyy_lc + (1.0 - frac) * 1.0
            exy = frac * exy_lc
            det = exx * eyy - exy ** 2
            return eyy / det, exx / det, -exy / det

        inv_xx_Ex, _, inv_xy_Ex = tensor_at(phi_Ex, frac_Ex)
        _, inv_yy_Ey, inv_xy_Ey = tensor_at(phi_Ey, frac_Ey)
        self.material = f2.Aniso2DYee(
            eps_inv_xx_Ex=jnp.asarray(inv_xx_Ex, dtype=_JDTYPE),
            eps_inv_xy_Ex=jnp.asarray(inv_xy_Ex, dtype=_JDTYPE),
            eps_inv_yy_Ey=jnp.asarray(inv_yy_Ey, dtype=_JDTYPE),
            eps_inv_xy_Ey=jnp.asarray(inv_xy_Ey, dtype=_JDTYPE),
        )

    def _build_vacuum_material(self):
        one = jnp.ones((self.Nx, self.Ny), dtype=_JDTYPE)
        zero = jnp.zeros((self.Nx, self.Ny), dtype=_JDTYPE)
        self.material = f2.Aniso2DYee(
            eps_inv_xx_Ex=one, eps_inv_xy_Ex=zero,
            eps_inv_yy_Ey=one, eps_inv_xy_Ey=zero,
        )

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
            x_meep = obj["center_x_meep"]
            sy = obj["size_y_meep"]
            i_src = _meep_to_grid_x(x_meep, self.cell_x, self.dx)
            j_lo, j_hi = _meep_to_grid_y_range(0.0, sy, self.cell_y, self.dx)

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
        Currently supports type 'flux' and '1Ddft' on x-normal planes."""
        for obj in self.objects_args:
            if obj.get("class") != "monitor":
                continue
            x_meep = obj["center_x_meep"]
            sy = obj["size_y_meep"]
            i_mon = _meep_to_grid_x(x_meep, self.cell_x, self.dx)
            j_lo, j_hi = _meep_to_grid_y_range(0.0, sy, self.cell_y, self.dx)
            lam_range = obj.get("lam_range", [0.5, 0.5])
            n_lam = int(obj.get("n_lam", 1))
            if n_lam == 1:
                freqs = np.array([1.0 / lam_range[0]])
            else:
                lambdas = np.linspace(lam_range[0], lam_range[1], n_lam)
                freqs = 1.0 / lambdas
            f0 = float(freqs[0])  # single-freq for now

            updater = f2.make_dft_updater_2d(axis=0, index=i_mon, frequency=f0)
            print(f"Monitor {obj['_key']}: x=i_mon={i_mon}, y∈[{j_lo},{j_hi}], "
                  f"type={obj.get('type')}, f0={f0}")
            self.monitors.append({
                "key": obj["_key"],
                "type": obj.get("type", "1Ddft"),
                "i_mon": i_mon,
                "j_lo": j_lo,
                "j_hi": j_hi,
                "freqs": freqs,
                "updater": updater,
                "state": f2.make_dft_state_2d(self.grid, axis=0),
            })

    # ---------------- Run ----------------

    def _build_pml(self):
        n_pml_cells = int(round(float(self.args.get("pml_size", 2.0)) / self.dx))
        self.pml = f2.make_cpml_2d(self.grid, self.dt,
                                    n_pml=(n_pml_cells, n_pml_cells))

    def run(self):
        self._set_data()
        self._update_all_args()
        self._build_material()
        self._build_sources()
        self._build_monitors()

        # Courant-stable dt (material-aware)
        self.dt = float(f2.courant_dt_2d(self.grid, safety=0.5,
                                          material=self.material))
        print(f"dt = {self.dt}")
        self._build_pml()

        run_until = float(self.args.get("run_until", 500.0))
        n_total = int(run_until / self.dt)
        print(f"run_until = {run_until}, n_total = {n_total} steps")

        # Build the time-loop body
        sources = self.sources
        updaters = [m["updater"] for m in self.monitors]
        grid = self.grid; dt = self.dt; material = self.material

        def apply_sources(fields, t):
            for s in sources:
                fields = s.apply(fields, t)
            return fields

        @jax.jit
        def run_loop(fields, pml_state, mon_states, n_steps):
            def body(i, state):
                f, p, ms = state
                t = i * dt
                f = apply_sources(f, t)
                f, p = f2.step_2d(f, grid, dt, p, material)
                ms = [u(m, f, t) for u, m in zip(updaters, ms)]
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
            amps = f2.extract_complex_2d(st, n_total)
            self._save_monitor(m, amps)

    def _save_monitor(self, mon, amps):
        out_path = os.path.join(self.paths["simulation"], f"{mon['key']}.npz")
        if mon["type"] == "flux":
            # Compute time-averaged Poynting flux through x-plane (S_x)
            # S_x = 0.5*Re(Ey*conj(Hz) - Ez*conj(Hy)). In TE 2D: Ez = Hy = 0,
            # so S_x = 0.5*Re(Ey * conj(Hz)).
            Sx = 0.5 * np.real(amps["Ey"] * np.conj(amps["Hz"]))
            flux = np.sum(Sx[mon["j_lo"]:mon["j_hi"]]) * self.dx
            np.savez(out_path,
                     freqs=mon["freqs"], fluxes=np.array([flux]))
            print(f"Saved {out_path}: flux={flux:.4g}")
        else:
            # DFT monitor: save Ex, Ey, Ez complex arrays cropped to monitor span
            # Ez doesn't exist in 2D TE — save zeros to match MEEP format
            Ex = np.array(amps["Ex"])[mon["j_lo"]:mon["j_hi"]]
            Ey = np.array(amps["Ey"])[mon["j_lo"]:mon["j_hi"]]
            Ez = np.zeros_like(Ex)
            # MEEP saves shape (n_freq, n_y_samples). For single freq:
            np.savez(out_path,
                     Ex=Ex[None, :], Ey=Ey[None, :], Ez=Ez[None, :],
                     freqs=mon["freqs"])
            print(f"Saved {out_path}: shape {Ex[None, :].shape}")


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
