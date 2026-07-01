"""Poisson solver wrapper for the 2D LC reservoir (with anisotropic ε).

Given a director field n̂ (encoded as φ, θ on the LC grid), build the
anisotropic dielectric tensor ε(n̂), solve −∇·(ε∇V) = 0 with Dirichlet BCs
from `VoltageElectrodes`, and return V + E = −∇V on the same grid.

Wraps `electrostatics_jax` (ported from BlockOptimization). Pure JAX, GPU-ready.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import jax.numpy as jnp

import electrostatics_jax as esj
from class_voltage_electrodes import VoltageElectrodes


class Poisson2D:
    """Solve Poisson on the LC grid given director + electrode voltages.

    Pipeline:
      director (phi, theta on grid) → build_eps_diag_jax → solve_poisson_jax
        → V (full grid) → gradient_V_jax → E = -∇V (3, nx, ny, nz).

    The dielectric tensor uses DC values (`eps_perp_dc`, `eps_a_dc`) — these
    are the static reorientational dielectrics, different from optical n_o/n_e.
    """

    def __init__(self, electrodes: VoltageElectrodes):
        self.electrodes = electrodes
        # Read DC dielectric params from the same JSON
        import json
        with open(electrodes.folder / "simulation_data.json") as f:
            d = json.load(f)
        cfg = d["reservoir"]
        ec = cfg.get("elastic_constants", {})
        self.eps_perp = float(cfg.get("eps_perp_dc",
                              cfg.get("eps_perp",
                              2.0)))
        self.eps_a    = float(cfg.get("eps_a_dc",
                              ec.get("epsilon_a",
                              10.0)))
        self.rtol = float(cfg.get("poisson_rtol", 1e-6))
        self.maxiter = int(cfg.get("poisson_maxiter", 2000))

        # Cache last solve
        self.V: np.ndarray | None = None
        self.E: np.ndarray | None = None

    # ---------------- Solve ----------------

    def solve(self, phi: np.ndarray, theta: np.ndarray | None = None,
              voltages: dict | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """Solve Poisson given director (phi, theta) + electrode voltages.

        Parameters
        ----------
        phi    : director azimuth (rad), shape `gshape`.
        theta  : director polar (rad), shape `gshape`. Default = π/2 (planar).
        voltages : optional dict {face_name: voltage_array} (face_name ∈
                   {"x_min","x_max","y_min","y_max"}). None → uses the
                   instance's stored per-face voltages.

        Returns
        -------
        V : (nx, ny, nz) potential
        E : (3, nx, ny, nz) electric field, E = −∇V
        """
        gshape = self.electrodes.gshape
        if theta is None:
            theta = np.full(gshape, np.pi / 2.0)
        assert phi.shape == gshape, f"phi shape {phi.shape} != gshape {gshape}"
        assert theta.shape == gshape, f"theta shape {theta.shape} != gshape {gshape}"

        mask, Vdir = self.electrodes.build_dirichlet(voltages)
        eps_diag = esj.build_eps_diag_jax(
            jnp.asarray(phi, dtype=jnp.float64),
            jnp.asarray(theta, dtype=jnp.float64),
            self.eps_perp, self.eps_a)
        V_3d = esj.solve_poisson_jax(
            eps_diag,
            self.electrodes.spacings,
            jnp.asarray(mask),
            jnp.asarray(Vdir, dtype=jnp.float64),
            rtol=self.rtol, maxiter=self.maxiter)
        E_3d = esj.gradient_V_jax(V_3d, self.electrodes.spacings)
        self.V = np.asarray(V_3d)
        self.E = np.asarray(E_3d)
        return self.V, self.E

    # ---------------- Persistence ----------------

    def save(self, fname: str = "poisson.npz",
             subdir: str = "simulation") -> str:
        """Write V + E + voltages + grid metadata to <folder>/<subdir>/<fname>."""
        if self.V is None or self.E is None:
            raise ValueError("Nothing to save — call solve() first.")
        out_dir = self.electrodes.folder / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname
        np.savez(out_path,
                 V=self.V, E=self.E,
                 voltages_flat=self.electrodes.all_voltages_flat,
                 voltages_x_min=self.electrodes.voltages["x_min"],
                 voltages_x_max=self.electrodes.voltages["x_max"],
                 voltages_y_min=self.electrodes.voltages["y_min"],
                 voltages_y_max=self.electrodes.voltages["y_max"],
                 sizes=np.asarray([self.electrodes.sx, self.electrodes.sy,
                                   self.electrodes.dz * (self.electrodes.nz - 1)]),
                 spacings=np.asarray(self.electrodes.spacings),
                 gshape=np.asarray(self.electrodes.gshape),
                 eps_perp=self.eps_perp, eps_a=self.eps_a)
        return str(out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Poisson 2D for voltage-electrode reservoir.")
    ap.add_argument("--path", required=True, help="design folder")
    args = ap.parse_args()

    ve = VoltageElectrodes(args.path)
    print(ve.summary())
    ps = Poisson2D(ve)
    print(f"[poisson_2d] eps_perp_dc={ps.eps_perp}, eps_a_dc={ps.eps_a}, "
          f"rtol={ps.rtol}, maxiter={ps.maxiter}")
    phi0 = np.zeros(ve.gshape)
    V, E = ps.solve(phi0)
    print(f"[poisson_2d] V range [{V.min():+.4f}, {V.max():+.4f}] V")
    print(f"[poisson_2d] |E|max = {float(np.linalg.norm(E, axis=0).max()):.4f} V/µm")
    out = ps.save()
    print(f"[poisson_2d] saved {out}")
