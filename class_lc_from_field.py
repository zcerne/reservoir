"""LC director from an external E field — two modes:

  * "shortcut": n = atan2(Ey, Ex) pointwise (F_E-dominant limit, fast, no relax).
  * "fwdbwd_relax": minimize Frank-Oseen + F_E with the forward+backward
                     gradient discretization (catches Nyquist mode, physics-correct).

Reads `reservoir.lc_mode` from JSON. Shortcut is fast (~1s); relax mode uses
NLopt LD_MMA with JAX-jit'd energy + gradient (~10-60 s depending on grid).
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp

from alcs_jax import (fe_core_director, fe_core_director_fwdbwd, n_sph,
                      fe_core_qtensor, qtensor_to_director, ldg_constants_from_frank)


class LCFromField:
    """Convert (E, BCs) → director field (phi, theta) on the LC grid."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        with open(self.folder / "simulation_data.json") as f:
            d = json.load(f)
        cfg = d["reservoir"]
        ec = cfg["elastic_constants"]
        # 5-tuple: (K1, K2, K3, q0, eps_a) for F_E coupling
        self.K = (float(ec["K1"]), float(ec["K2"]), float(ec["K3"]),
                  float(ec.get("q0", 0.0)),
                  float(ec.get("epsilon_a", 10.0)))
        self.mode = str(cfg.get("lc_mode", "shortcut"))
        if self.mode not in ("shortcut", "fwdbwd_relax", "mma3d", "qtensor_mma3d"):
            raise ValueError(
                f"reservoir.lc_mode must be 'shortcut', 'fwdbwd_relax', 'mma3d' "
                f"or 'qtensor_mma3d', got {self.mode!r}")
        self.relax_maxeval = int(cfg.get("lc_relax_maxeval", 2000))
        self.relax_xtol    = float(cfg.get("lc_relax_xtol", 1e-6))
        # Face anchoring: "face_phi": [x_min, x_max, y_min, y_max, z_min, z_max]
        # entries in RADIANS or null (free). Honored by fwdbwd_relax via
        # bound-equality pinning of the boundary-plane phi dofs (strong
        # anchoring). shortcut mode cannot honor anchoring — error out there.
        fp = cfg.get("face_phi", [None] * 6)
        self.face_phi = list(fp) + [None] * (6 - len(fp))
        # SOFT anchoring: "face_anchor_W" (same 6-slot layout) = Rapini-Papoular
        # surface strength W [pN/µm] per face; adds (W/2)·sin²(φ−φ₀)·dA on that
        # boundary plane instead of pinning it (extrapolation length L = K1/W).
        # null/0 with face_phi set keeps the hard pin.
        fw = cfg.get("face_anchor_W", [None] * 6)
        self.face_anchor_W = list(fw) + [None] * (6 - len(fw))

        # Q-tensor (Landau–de Gennes) constants — derived from the same Frank
        # constants + S_eq, so no new config keys are required. Reuses eps_a for
        # the electric coupling. Calibrated bulk min at S_eq (default 5CB ≈ 0.80).
        self.S_eq = float(ec.get("S_eq", 0.80))
        # S-cap for the Q-tensor electric term (kills the DC-field S→runaway); a
        # touch above S_eq so the equilibrium is reachable. Config-overridable.
        self.S_cap = float(cfg.get("lc_S_cap", 1.05 * self.S_eq))
        self.Kq = ldg_constants_from_frank(
            self.K[0], self.K[1], self.K[2], self.S_eq, eps_a=self.K[4])

        # Last results
        self.phi: np.ndarray | None = None
        self.theta: np.ndarray | None = None
        self.Q: np.ndarray | None = None     # q5 field (5,)+gshape, qtensor mode only
        self.S: np.ndarray | None = None     # scalar order, qtensor mode only

    # ---------------- Public API ----------------

    def compute(self, E: np.ndarray, gshape: tuple[int, int, int],
                spacings: tuple[float, float, float],
                phi_init: np.ndarray | None = None,
                full_3d: bool = False
                ) -> tuple[np.ndarray, np.ndarray]:
        """Return (phi, theta) on `gshape` grid given external E field.

        E shape: (3, nx, ny, nz).

        If `full_3d` is False (default, 2D-mode): theta is always π/2 and
        phi follows the in-plane E direction.

        If `full_3d` is True (3D-mode, shortcut only for now): the full 3D
        director is aligned with E, so both phi AND theta vary with E:
            phi   = atan2(Ey, Ex)
            theta = acos(Ez / |E|)        (clipped where |E| ≈ 0)
        Where |E| < eps, theta defaults to π/2 (planar in xy) and
        phi defaults to 0 — same as no-field case.

        phi_init: warm-start for relax mode (ignored in shortcut). Default zeros.
        """
        if full_3d:
            if self.mode == "shortcut":
                return self._compute_3d_shortcut(E, gshape)
            elif self.mode == "mma3d":
                return self._relax_mma3d(E, gshape, spacings, phi_init)
            elif self.mode == "qtensor_mma3d":
                return self._relax_qtensor_mma3d(E, gshape, spacings, phi_init)
            else:
                raise ValueError(
                    f"3D mode requires lc_mode in ('shortcut', 'mma3d', "
                    f"'qtensor_mma3d'); got {self.mode!r}")
        # 2D (planar) path
        if self.mode == "qtensor_mma3d":
            # planar Q relaxation: relax q5 then recover (phi, theta=π/2-ish).
            return self._relax_qtensor_mma3d(E, gshape, spacings, phi_init)
        theta = np.full(gshape, np.pi / 2.0)
        if self.mode == "shortcut":
            phi = np.asarray(jnp.arctan2(E[1], E[0]))
        else:
            phi = self._relax_fwdbwd(E, gshape, spacings, phi_init)
        self.phi = phi
        self.theta = theta
        return phi, theta

    def _relax_mma3d(self, E: np.ndarray, gshape: tuple[int, int, int],
                     spacings: tuple[float, float, float],
                     phi_init: np.ndarray | None
                     ) -> tuple[np.ndarray, np.ndarray]:
        """3D joint phi+theta relaxation via GPU_MMA (pure-JAX MMA on device).

        DOF vector: x = concat(phi.flatten(), theta.flatten()) of length 2*n_total.
        Energy: Frank-Oseen (fwdbwd discretization) + F_E coupling, both with
        full 3D director (phi, theta varying independently per cell).

        Bounds: ±8π for both phi and theta (angles, generous).
        """
        import sys
        import os as _os
        for _p in ("/home/ziga/Orion", "/home/cernez"):      # dir CONTAINING LCrelax pkg
            if _os.path.isdir(_os.path.join(_p, "LCrelax")) and _p not in sys.path:
                sys.path.append(_p)                          # append: never shadow own modules
        from LCrelax.mma_engine import minimize as gpumma_minimize, MMAParams

        nx, ny, nz = gshape
        n_total = nx * ny * nz
        constants_5 = self.K
        E_j = jnp.asarray(E, dtype=jnp.float32)

        def energy(x_flat):
            phi_flat = x_flat[:n_total]
            theta_flat = x_flat[n_total:]
            nv = n_sph(phi_flat, theta_flat).reshape((3,) + gshape)
            return fe_core_director_fwdbwd(nv, constants_5, spacings, fields_=E_j)

        # value_and_grad pair for gpumma's API
        vg = jax.jit(jax.value_and_grad(energy))
        def value_and_grad_fn(x):
            return vg(x)

        # Initial state: phi = phi_init or 0, theta = pi/2
        if phi_init is None:
            phi0 = jnp.zeros(n_total, dtype=jnp.float32)
        else:
            phi0 = jnp.asarray(phi_init, dtype=jnp.float32).flatten()
            assert phi0.size == n_total, f"phi_init {phi0.size} != n_total {n_total}"
        theta0 = jnp.full(n_total, jnp.pi / 2.0, dtype=jnp.float32)
        x0 = jnp.concatenate([phi0, theta0])
        bound = 8.0 * jnp.pi
        xmin = jnp.full(2 * n_total, -bound, dtype=jnp.float32)
        xmax = jnp.full(2 * n_total,  bound, dtype=jnp.float32)

        import time
        t0 = time.time()
        x_star, info = gpumma_minimize(
            value_and_grad_fn, x0, xmin, xmax,
            maxiter=self.relax_maxeval,
            xtol=self.relax_xtol,
            gtol=1e-7,
            params=MMAParams(),
        )
        elapsed = time.time() - t0
        niter = int(info["niter"])
        kkt = float(info["kkt_norm"])
        f_star = float(info["f_star"])
        print(f"[lc_from_field/mma3d] {niter} iters, f*={f_star:.3e}, "
              f"kkt={kkt:.3e}, wall={elapsed:.1f}s")

        phi = np.asarray(x_star[:n_total]).reshape(gshape)
        theta = np.asarray(x_star[n_total:]).reshape(gshape)
        self.phi = phi
        self.theta = theta
        return phi, theta

    def _relax_qtensor_mma3d(self, E: np.ndarray, gshape: tuple[int, int, int],
                             spacings: tuple[float, float, float],
                             phi_init: np.ndarray | None
                             ) -> tuple[np.ndarray, np.ndarray]:
        """Q-tensor (Landau–de Gennes) relaxation via GPU_MMA, parallel to
        `_relax_mma3d` but minimizing fe_core_qtensor over the 5 components q5.

        DOF vector: x = q5.flatten(), length 5*n_total.
        Energy: ½L|∇Q|² + LdG bulk(A,B,C) + electric −½eps_a·EᵀQE.
        Returns (phi, theta) recovered from Q by eigendecomposition; also stores
        self.Q (q5 field) and self.S (scalar order) for downstream use.

        Init: uniaxial Q from phi_init (or planar phi=0) at S=S_eq — i.e. the same
        starting director as the Frank path, lifted into Q-space.
        """
        import sys, time
        import os as _os
        for _p in ("/home/ziga/Orion", "/home/cernez"):      # dir CONTAINING LCrelax pkg
            if _os.path.isdir(_os.path.join(_p, "LCrelax")) and _p not in sys.path:
                sys.path.append(_p)                          # append: never shadow own modules
        from LCrelax.mma_engine import minimize as gpumma_minimize, MMAParams

        nx, ny, nz = gshape
        n_total = nx * ny * nz
        constants_q = self.Kq
        E_j = jnp.asarray(E, dtype=jnp.float32)

        def energy(x_flat):
            q5 = x_flat.reshape((5,) + gshape)
            return fe_core_qtensor(q5, constants_q, spacings, fields_=E_j)

        vg = jax.jit(jax.value_and_grad(energy))

        # initial uniaxial Q = S_eq (n⊗n − I/3) from phi_init, theta = π/2 (planar)
        if phi_init is None:
            phi0 = np.zeros(gshape, dtype=np.float32)
        else:
            phi0 = np.asarray(phi_init, dtype=np.float32).reshape(gshape)
        theta0 = np.full(gshape, np.pi / 2.0, dtype=np.float32)
        nv = np.asarray(n_sph(jnp.asarray(phi0), jnp.asarray(theta0)))   # (3,)+gshape
        S = self.S_eq
        Qxx = S * (nv[0] * nv[0] - 1.0 / 3.0)
        Qyy = S * (nv[1] * nv[1] - 1.0 / 3.0)
        Qxy = S * (nv[0] * nv[1])
        Qxz = S * (nv[0] * nv[2])
        Qyz = S * (nv[1] * nv[2])
        q5_0 = jnp.asarray(np.stack([Qxx, Qxy, Qxz, Qyy, Qyz]).reshape(-1),
                           dtype=jnp.float32)
        # S-cap (BlockOpt-proven): the electric term −½ε_a·EᵀQE is unbounded and
        # under a strong DC field drives S past its physical value. Cap S via the
        # box: a uniaxial Q has |Q_diag| = 2S/3, so bound = 2·S_cap/3 holds S ≤ S_cap.
        bound = 2.0 * self.S_cap / 3.0
        xmin = jnp.full(5 * n_total, -bound, dtype=jnp.float32)
        xmax = jnp.full(5 * n_total,  bound, dtype=jnp.float32)

        t0 = time.time()
        x_star, info = gpumma_minimize(
            vg, q5_0, xmin, xmax,
            maxiter=self.relax_maxeval, xtol=self.relax_xtol, gtol=1e-7,
            params=MMAParams())
        elapsed = time.time() - t0
        print(f"[lc_from_field/qtensor_mma3d] {int(info['niter'])} iters, "
              f"f*={float(info['f_star']):.3e}, kkt={float(info['kkt_norm']):.3e}, "
              f"wall={elapsed:.1f}s")

        q5_star = x_star.reshape((5,) + gshape)
        phi_j, theta_j, S_j = qtensor_to_director(q5_star)
        phi = np.asarray(phi_j); theta = np.asarray(theta_j)
        self.phi = phi
        self.theta = theta
        self.Q = np.asarray(q5_star)
        self.S = np.asarray(S_j)
        return phi, theta

    def _compute_3d_shortcut(self, E: np.ndarray, gshape: tuple[int, int, int],
                             eps_floor: float = 1e-12
                             ) -> tuple[np.ndarray, np.ndarray]:
        """3D-mode shortcut: align director with E (both phi+theta from E)."""
        Ex = np.asarray(E[0]); Ey = np.asarray(E[1]); Ez = np.asarray(E[2])
        Emag = np.sqrt(Ex * Ex + Ey * Ey + Ez * Ez)
        Emag_safe = np.where(Emag > eps_floor, Emag, 1.0)
        # phi: in-plane azimuth (atan2(Ey, Ex)); theta: polar from +z.
        phi   = np.where(Emag > eps_floor, np.arctan2(Ey, Ex), 0.0)
        cos_t = np.where(Emag > eps_floor, Ez / Emag_safe, 0.0)
        cos_t = np.clip(cos_t, -1.0, 1.0)
        theta = np.arccos(cos_t)
        # Where the field vanishes (or is degenerate), fall back to planar (π/2).
        theta = np.where(Emag > eps_floor, theta, np.pi / 2.0)
        # Reshape sanity
        assert phi.shape == gshape and theta.shape == gshape, \
            f"shapes {phi.shape}/{theta.shape} != gshape {gshape}"
        self.phi = phi
        self.theta = theta
        return phi, theta

    # ---------------- Internals ----------------

    def _relax_fwdbwd(self, E: np.ndarray, gshape: tuple[int, int, int],
                      spacings: tuple[float, float, float],
                      phi_init: np.ndarray | None) -> np.ndarray:
        """Minimize total energy F = ∫[K|∇n|² − ½ε_a(n·E)²] dV via NLopt LD_MMA.

        Uses fwdbwd Frank-Oseen so the Nyquist mode is correctly penalized
        (prevents spurious 1-cell director stripes the central-diff version
        allows for "free").
        """
        import nlopt
        nx, ny, nz = gshape
        n_total = nx * ny * nz
        theta_flat = jnp.full(n_total, jnp.pi / 2.0)
        constants_5 = self.K
        E_j = jnp.asarray(E)

        # Soft (Rapini-Papoular) anchoring terms: (W/2)·sin²(φ−φ₀)·dA summed
        # over each soft-anchored boundary plane. Hard-pinned faces (face_phi
        # set, W null) are handled below via bound equality instead.
        _face_slices = [(0, slice(None), slice(None)),
                        (-1, slice(None), slice(None)),
                        (slice(None), 0, slice(None)),
                        (slice(None), -1, slice(None)),
                        (slice(None), slice(None), 0),
                        (slice(None), slice(None), -1)]
        _dA = [spacings[1] * spacings[2], spacings[1] * spacings[2],
               spacings[0] * spacings[2], spacings[0] * spacings[2],
               spacings[0] * spacings[1], spacings[0] * spacings[1]]
        soft = [(sl, float(self.face_phi[k]), float(self.face_anchor_W[k]), _dA[k])
                for k, sl in enumerate(_face_slices)
                if self.face_phi[k] is not None and self.face_anchor_W[k]]

        def energy(phi_flat):
            nv = n_sph(phi_flat, theta_flat).reshape((3,) + gshape)
            F = fe_core_director_fwdbwd(nv, constants_5, spacings, fields_=E_j)
            if soft:
                phi3 = phi_flat.reshape(gshape)
                for sl, phi0, W, dA in soft:
                    F = F + 0.5 * W * dA * jnp.sum(jnp.sin(phi3[sl] - phi0) ** 2)
            return F

        E_jit = jax.jit(energy)
        g_jit = jax.jit(jax.grad(energy))

        def objective(x, grad):
            x_j = jnp.asarray(x)
            if grad.size:
                grad[:] = np.asarray(g_jit(x_j), dtype=np.float64)
            return float(E_jit(x_j))

        if phi_init is None:
            x0 = np.zeros(n_total, dtype=np.float64)
        else:
            x0 = np.asarray(phi_init, dtype=np.float64).flatten()
            assert x0.size == n_total, f"phi_init size {x0.size} != n_total {n_total}"

        opt = nlopt.opt(nlopt.LD_MMA, n_total)
        opt.set_min_objective(objective)
        opt.set_maxeval(self.relax_maxeval)
        opt.set_xtol_rel(self.relax_xtol)
        # Generous bounds — director is angle, +8π gives plenty of room.
        bound = 8.0 * np.pi
        lb = np.full(n_total, -bound)
        ub = np.full(n_total,  bound)
        # Strong anchoring: pin boundary-plane phi via bound equality (only
        # faces WITHOUT a soft W — those are already in the energy).
        if any(v is not None for v in self.face_phi):
            pin = np.full(gshape, np.nan)
            for k, sl in enumerate(_face_slices):
                if self.face_phi[k] is not None and not self.face_anchor_W[k]:
                    pin[sl] = float(self.face_phi[k])
            pinf = pin.flatten()
            fixed = ~np.isnan(pinf)
            lb[fixed] = pinf[fixed]
            ub[fixed] = pinf[fixed]
            x0[fixed] = pinf[fixed]
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        try:
            phi_star = opt.optimize(x0)
        except (nlopt.RoundoffLimited, RuntimeError) as e:
            print(f"[lc_from_field] NLopt stopped early: {type(e).__name__}: {e}; "
                  f"using last iterate")
            phi_star = x0
        return np.asarray(phi_star).reshape(gshape)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute LC director from E (shortcut or relax).")
    ap.add_argument("--path", required=True, help="design folder")
    ap.add_argument("--mode", choices=["shortcut", "fwdbwd_relax"], default=None,
                    help="override reservoir.lc_mode")
    args = ap.parse_args()

    # Load E from simulation/poisson.npz (output of class_poisson_2d.py)
    p = Path(args.path) / "simulation" / "poisson.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run class_poisson_2d.py first")
    d = np.load(p)
    E = d["E"]
    _g = [int(x) for x in d["gshape"]]
    _s = [float(x) for x in d["spacings"]]
    assert len(_g) == 3 and len(_s) == 3, f"gshape/spacings must be length 3, got {_g}/{_s}"
    gshape: tuple[int, int, int] = (_g[0], _g[1], _g[2])
    spacings: tuple[float, float, float] = (_s[0], _s[1], _s[2])

    lc = LCFromField(args.path)
    if args.mode is not None:
        lc.mode = args.mode
    import time
    t0 = time.time()
    phi, theta = lc.compute(E, gshape, spacings)
    print(f"[lc_from_field] mode={lc.mode}  "
          f"phi range [{phi.min():+.3f}, {phi.max():+.3f}]  "
          f"({time.time()-t0:.1f}s)")
    out = Path(args.path) / "simulation" / f"lc_{lc.mode}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, phi=phi, theta=theta, mode=lc.mode)
    print(f"[lc_from_field] saved {out}")
