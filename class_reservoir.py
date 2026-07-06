import json
import numpy as np
from pathlib import Path
from alcs_class import XaLCS
from lc_geometry import get_dielectric_3d


class Reservoir:
    """
    LC rectangle/cube that relaxes to minimal Frank free energy under planar
    boundary conditions. Reads parameters from simulation_data.json in the
    given folder (lc key).

    face_phi: 6-tuple (x_min, x_max, y_min, y_max, z_min, z_max) in radians.
              None = free (no anchoring). Default: all faces planar at phi=0.
    """

    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.folder = folder
        with open(folder / "simulation_data.json") as f:
            data = json.load(f)
        cfg = data["reservoir"]

        ec = cfg["elastic_constants"]
        self.dimensions = tuple(cfg["sizes"])
        self.resolution = cfg["resolution"]
        # Back-compat for voltage_reservoir JSON which doesn't carry these keys.
        # They're only used by run_minimization(), which voltage_reservoir runs
        # don't go through (they write lc_fields.npz directly via run_voltage_reservoir.py).
        self.boundary_conditions = tuple(cfg.get("boundary_conditions", ("free",) * 3))
        self.elastic_constants = (ec["K1"], ec["K2"], ec["K3"], ec["q0"])
        self.face_phi   = tuple(x if x is not None else None for x in cfg.get("face_phi", [None]*6))
        _fp = cfg.get("face_theta")
        if _fp is not None:
            self.face_theta = tuple(x if x is not None else None for x in _fp)
        else:
            # backward compat: pin theta=π/2 wherever phi is pinned
            self.face_theta = tuple(np.pi / 2 if p is not None else None for p in self.face_phi)
        self.optimize_phi_theta = tuple(cfg.get("optimize_phi_theta", [True, True]))
        self.boundary_function    = cfg.get("boundary_function", None)
        self.boundary_seed        = cfg.get("boundary_seed", None)
        self.boundary_scale       = float(cfg.get("boundary_scale", 2.0))
        self.boundary_n_periods   = int(cfg.get("boundary_n_periods", 1))
        self.boundary_phase_shift = float(cfg.get("boundary_phase_shift", 0.0))
        self.boundary_noise_level    = float(cfg.get("boundary_noise_level", 0.5))
        self.boundary_same_opposite  = cfg.get("boundary_same_opposite", None)
        self.boundary_mirror         = bool(cfg.get("boundary_mirror", False))
        self.ignore_faces            = cfg.get("ignore_faces", None)
        self.maxeval = cfg.get("maxeval", 2000)
        self.f_tolerance = cfg.get("f_tolerance", 1e-6)
        self.n_o = float(cfg.get("n_o", 1.52))
        self.n_e = float(cfg.get("n_e", 1.71))
        self.S   = float(cfg.get("S", 1.0))
        self.n_background = float(data.get("background_index", 1.0))
        # STED / 4-level gain (nonlinear reservoir). Optional `reservoir.sted` block:
        # pump (λ=lbdA) inverts the medium, signal (λ=lbdE) stimulates emission → gain.
        # Same MultilevelAtom model as the Photoisomerization resonator project.
        self.sted = cfg.get("sted", None)
        # LC model: "director" (Frank, XaLCS) or "Q3D" (Landau-de Gennes, fe_core_qtensor)
        self.lc_param = str(cfg.get("lc_param", "director"))
        self.S_eq  = float(cfg.get("S_eq", 0.80))
        self.S_cap = float(cfg.get("S_cap", 1.05 * self.S_eq))
        self._sim = None
        self._phi_cache: np.ndarray | None = None
        self._theta_cache: np.ndarray | None = None
        self._S_cache: np.ndarray | None = None
        self._meep_center_x: float = 0.0

    def _cell_size(self) -> tuple[float, float, float]:
        if len(self.dimensions) == 2:
            sx, sy = self.dimensions
            return (float(sx), float(sy), 4.0 / self.resolution)
        sx, sy, sz = self.dimensions
        return (float(sx), float(sy), float(sz))

    def run_minimization(self):
        cell = self._cell_size()
        sim = XaLCS(cell_size=cell, resolution=self.resolution,
                    elastic_constants=self.elastic_constants,
                    boundary_conditions=self.boundary_conditions,
                    optimize_phi_theta=self.optimize_phi_theta)
        sim.maxeval = self.maxeval
        sim.f_tolerance = self.f_tolerance

        cell_size = np.asarray(cell)
        res = (cell_size * self.resolution + 1).astype(int)
        nz_pts = int(res[2])

        if self.boundary_function is not None:
            from functions_boundaries import (random_2d_boundaries, random_3d_boundaries,
                                              perlin_3d_boundaries, sinus_3d_boundaries,
                                              sinus_2d_boundaries, sinus_random_2d_boundaries,
                                              smooth_random_2d_boundaries,
                                              defect_2d_boundaries,
                                              competing_3d_boundaries)
            _fn_map = {"random": random_2d_boundaries, "random_3d": random_3d_boundaries,
                       "perlin_3d": perlin_3d_boundaries, "sinus_3d": sinus_3d_boundaries,
                       "sinus_2d": sinus_2d_boundaries,
                       "sinus_random_2d": sinus_random_2d_boundaries,
                       "smooth_random_2d": smooth_random_2d_boundaries,
                       "defect_2d": defect_2d_boundaries,
                       "competing_3d": competing_3d_boundaries}
            fn = _fn_map[self.boundary_function]
            dims = self.dimensions if self.boundary_function in (
                "random_3d", "perlin_3d", "sinus_3d", "competing_3d") else self.dimensions[:2]
            extra_kw = {}
            if self.boundary_function in ("perlin_3d", "competing_3d", "smooth_random_2d"):
                extra_kw["scale"] = self.boundary_scale
            if self.boundary_function == "perlin_3d" and self.boundary_same_opposite is not None:
                extra_kw["same_opposite_faces"] = self.boundary_same_opposite
            if self.boundary_function in ("sinus_2d", "sinus_3d", "sinus_random_2d"):
                extra_kw["n_periods"] = self.boundary_n_periods
            if self.boundary_function in ("sinus_2d", "sinus_random_2d"):
                extra_kw["phase_shift"] = self.boundary_phase_shift
            if self.boundary_function == "sinus_random_2d":
                extra_kw["noise_level"] = self.boundary_noise_level
                extra_kw["scale"] = self.boundary_scale
            gen_dims = (dims[0] / 2,) + tuple(dims[1:]) if self.boundary_mirror else dims
            fp_arr, ft_arr = fn(self.resolution, gen_dims, seed=self.boundary_seed,
                                ignore_faces=self.ignore_faces, **extra_kw)
            if self.boundary_mirror:
                def _mirror_arr(v):
                    return np.concatenate([v, v[-2::-1]]) if v is not None else None
                fp_arr = {k: _mirror_arr(v) for k, v in fp_arr.items()}
                ft_arr = {k: _mirror_arr(v) for k, v in ft_arr.items()}
            if "z_min" in fp_arr:
                # 3D function: all 6 faces have per-pixel arrays sized to match face_mask counts
                active_face_phi = [fp_arr[k] for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")]
                active_face_theta = [ft_arr[k] for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")]
            else:
                # 2D function: repeat each 1D edge array across the z dimension
                def _rep(v): return np.repeat(v, nz_pts) if v is not None else None
                active_face_phi = [
                    _rep(fp_arr["x_min"]), _rep(fp_arr["x_max"]),
                    _rep(fp_arr["y_min"]), _rep(fp_arr["y_max"]),
                    None, None,
                ]
                active_face_theta = [
                    _rep(ft_arr["x_min"]), _rep(ft_arr["x_max"]),
                    _rep(ft_arr["y_min"]), _rep(ft_arr["y_max"]),
                    None, None,
                ]
        else:
            active_face_phi   = list(self.face_phi)
            active_face_theta = list(self.face_theta)
        x = np.linspace(-cell_size[0]/2, cell_size[0]/2, res[0])
        y = np.linspace(-cell_size[1]/2, cell_size[1]/2, res[1])
        z = np.linspace(-cell_size[2]/2, cell_size[2]/2, res[2])
        pos_x, pos_y, pos_z = np.meshgrid(x, y, z)
        pos = np.swapaxes(np.asarray((pos_x, pos_y, pos_z)), 1, 2)
        x_coords, y_coords, z_coords = pos[0].ravel(), pos[1].ravel(), pos[2].ravel()
        n = x_coords.size

        # Boolean masks selecting the outermost grid points on each face.
        # Pinning their bounds enforces anchoring.
        face_masks = [
            x_coords == -cell_size[0]/2,  # x_min
            x_coords ==  cell_size[0]/2,  # x_max
            y_coords == -cell_size[1]/2,  # y_min
            y_coords ==  cell_size[1]/2,  # y_max
            z_coords == -cell_size[2]/2,  # z_min
            z_coords ==  cell_size[2]/2,  # z_max
        ]

        phi0 = np.zeros(n)
        theta0 = np.full(n, np.pi / 2)

        lb_phi = np.full(n, -np.pi)
        ub_phi = np.full(n,  np.pi)
        lb_theta = np.zeros(n)
        ub_theta = np.full(n, np.pi)

        for mask, phi_val, theta_val in zip(face_masks, active_face_phi, active_face_theta):
            if phi_val is not None:
                phi0[mask] = phi_val
                lb_phi[mask] = phi_val
                ub_phi[mask] = phi_val
            if theta_val is not None:
                theta0[mask] = theta_val
                lb_theta[mask] = theta_val
                ub_theta[mask] = theta_val

        sim.initial_state_phi = phi0
        sim.initial_state_theta = theta0
        sim.lower_bounds_phi = lb_phi
        sim.upper_bounds_phi = ub_phi
        sim.lower_bounds_theta = lb_theta
        sim.upper_bounds_theta = ub_theta

        # Q-tensor (Landau-de Gennes) relaxation path: reuse the face anchoring
        # (phi0/theta0 + pinned bounds) but minimize fe_core_qtensor over q5 via
        # GPU_MMA, then recover (phi,theta) by eigendecomposition. Resolution is
        # uniform (dx=dy=dz=1/res) so the elastic energy is axis-permutation-
        # invariant -> reshape order is irrelevant.
        if self.lc_param == "Q3D":
            gshape = tuple(int(v) for v in res)
            # Warm-start the Q relax from the DIRECTOR solution: the director angle
            # is a soft Goldstone mode the q5-MMA won't relax from a cold phi=0
            # interior, so first run the (fwdbwd) Frank relax and seed Q with it.
            # sim._result[:n] is the flat optimized phi in phi0's exact order.
            sim.setup(); sim.minimize()
            n = phi0.size
            r = np.asarray(sim._result)
            opt_phi, opt_theta = self.optimize_phi_theta
            if opt_phi and opt_theta:
                phi_seed, theta_seed = r[:n].copy(), r[n:].copy()
            elif opt_phi:
                phi_seed, theta_seed = r[:n].copy(), theta0.copy()
            else:
                phi_seed, theta_seed = phi0.copy(), r[:n].copy()
            self._relax_qtensor(phi_seed, theta_seed, lb_phi, ub_phi,
                                lb_theta, ub_theta, gshape, 1.0 / self.resolution)
            return

        sim.setup()
        sim.minimize()
        self._sim = sim

    def _relax_qtensor(self, phi0, theta0, lb_phi, ub_phi, lb_theta, ub_theta, gshape, dx):
        """Q-tensor (LdG) relaxation using **BlockOpt's exact machinery** — the same
        `relax_qtensor_3d` + `ldg_constants_5cb` (real 5CB SI Landau coefficients,
        L=K_avg/2S², ξ=ε₀ε_a/S, SI units), so the reservoir relax is identical to
        the BlockOptimization LC stage. Boundary anchoring → Dirichlet boundary_mask.
        Note: physical 5CB defect-core ξ_n≈1.4nm is sub-grid, so S stays ≈S_eq
        except a ~1-pixel core (set artificially soft A,B,C for fat melted cores)."""
        import sys, os, time, numpy as _np, jax.numpy as jnp
        for _p in ("/home/ziga/Orion/BlockOptimization", "/home/cernez/BlockOptimization"):
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)
        from lc_stuff.qtensor_3d import (relax_qtensor_3d, ldg_constants_5cb,
                                         q5_from_director, director_and_S)
        from alcs_jax import n_sph
        ec = self.elastic_constants                                    # (K1,K2,K3,q0)
        eps_a = self.n_e ** 2 - self.n_o ** 2
        cst = ldg_constants_5cb(float(ec[0]), float(ec[1]), float(ec[2]), eps_a, dx_um=dx)
        S = float(cst["S_eq"])
        spac = (dx * 1e-6, dx * 1e-6, dx * 1e-6)                        # µm → m (SI)
        # uniaxial Q seed from the (warm-started) director; n shape (3,)+gshape
        nv = _np.asarray(n_sph(jnp.asarray(phi0), jnp.asarray(theta0))).reshape((3,) + gshape)
        q5_0 = _np.asarray(q5_from_director(jnp.asarray(nv), S))        # (5,)+gshape
        # Dirichlet pin on anchored faces (where phi or theta bound was pinned)
        pin = ((lb_phi == ub_phi) | (lb_theta == ub_theta)).reshape(gshape)
        t0 = time.time()
        q5_star, info = relax_qtensor_3d(
            q5_0, None, A=cst["A"], B=cst["B"], C=cst["C"], L=cst["L"], xi=cst["xi"],
            spacings=spac, boundary_mask=jnp.asarray(pin), maxiter=int(self.maxeval))
        print(f"[reservoir/Q3D BlockOpt-LdG] {int(info['niter'])} iters, "
              f"f*={float(info['f_star']):.3e}, S_eq={S:.4f}, xi_n={cst['xi_n_nm']:.2f}nm, "
              f"wall={time.time()-t0:.1f}s")
        q5 = _np.asarray(q5_star)
        self._Q_cache = q5                       # raw q5 → direct ε(Q) (biaxiality/core preserved)
        n_dir, Sf = director_and_S(jnp.asarray(q5))    # n (3,)+gshape, S gshape
        n_dir = _np.asarray(n_dir)
        self._phi_cache = _np.arctan2(n_dir[1], n_dir[0]).reshape(gshape)
        self._theta_cache = _np.arccos(_np.clip(n_dir[2], -1.0, 1.0)).reshape(gshape)
        self._S_cache = _np.asarray(Sf).reshape(gshape)

    def load_fields(self):
        """Load pre-computed director field from lc_fields.npz (skips minimization)."""
        npz = self.folder / "simulation" / "lc_fields.npz"
        data = np.load(npz)
        self._phi_cache   = data["phi"]
        self._theta_cache = data["theta"]

    def get_results(self):
        """Returns (phi, theta, nx, ny, nz) arrays on the 3D grid."""
        if self._phi_cache is not None and self._theta_cache is not None:
            phi, theta = self._phi_cache, self._theta_cache
            nx = np.sin(theta) * np.cos(phi)
            ny = np.sin(theta) * np.sin(phi)
            nz = np.cos(theta)
            return phi, theta, nx, ny, nz
        if self._sim is None:
            raise RuntimeError("Run run_minimization() or load_fields() first.")
        return self._sim.get_results()

    def _build_sted_susceptibilities(self):
        """Build the 4-level MultilevelAtom gain susceptibility from `reservoir.sted`.

        Identical model to the Photoisomerization resonator (class_resonator):
          1→4  pump absorption   (freq 1/lbdA, gamma gammaA)
          4→3  fast decay        (transition_rate)
          3→2  emission/lasing   (freq 1/lbdE, gamma gammaE)  ← signal stimulates this
          2→1  fast decay        (transition_rate)
        Returns [] when no `sted` block, so the base LC medium is unchanged.
        """
        s = self.sted
        if not s or not s.get("enabled", True):
            return []
        import meep as mp
        lbdA = float(s["lbdA"]); gammaA = float(s["gammaA"])
        lbdE = float(s["lbdE"]); gammaE = float(s["gammaE"])
        SGMA = float(s["SGMA"]); N1_0 = float(s["N1_0"]); N3_0 = float(s.get("N3_0", 0.0))
        r43 = float(s.get("rate_43", 10.0)); r21 = float(s.get("rate_21", 100.0))
        transitions = [
            mp.Transition(1, 4, frequency=1 / lbdA, gamma=gammaA, sigma_diag=mp.Vector3(1, 1, 1)),
            mp.Transition(4, 3, transition_rate=r43),
            mp.Transition(2, 3, frequency=1 / lbdE, gamma=gammaE, sigma_diag=mp.Vector3(1, 1, 1)),
            mp.Transition(2, 1, transition_rate=r21),
        ]
        return [mp.MultilevelAtom(sigma=SGMA, transitions=transitions,
                                  initial_populations=[N1_0, 0, N3_0, 0])]

    def get_geometry_blocks(self):
        import meep as mp

        cell = self._cell_size()
        sx, sy, sz = float(cell[0]), float(cell[1]), float(cell[2])
        n_o_sq = self.n_o ** 2
        n_e_sq = self.n_e ** 2
        S  = self.S
        cx = self._meep_center_x
        _sted_susc = self._build_sted_susceptibilities()   # [] unless reservoir.sted set

        if len(self.dimensions) == 2:
            from scipy.interpolate import RectBivariateSpline
            phi, theta, *_ = self.get_results_2d()
            nx_pts, ny_pts = phi.shape
            x_lc = np.linspace(-sx / 2, sx / 2, nx_pts)
            y_lc = np.linspace(-sy / 2, sy / 2, ny_pts)
            phi_interp   = RectBivariateSpline(x_lc, y_lc, phi)
            theta_interp = RectBivariateSpline(x_lc, y_lc, theta)

            def _mat(v):
                lc_x = float(v.x) - cx
                phi_v   = float(np.asarray(phi_interp(lc_x, float(v.y))).flat[0])
                theta_v = float(np.asarray(theta_interp(lc_x, float(v.y))).flat[0])
                d, od = get_dielectric_3d(n_o_sq, n_e_sq, phi_v, theta_v, S)
                return mp.Medium(epsilon_diag=d, epsilon_offdiag=od,
                                 E_susceptibilities=_sted_susc)

            return [mp.Block(
                center=mp.Vector3(cx, 0, 0),
                size=mp.Vector3(sx, sy, mp.inf),
                material=lambda v: _mat(v),
            )]
        else:
            from scipy.interpolate import RegularGridInterpolator
            phi, theta, *_ = self.get_results()
            nx_pts = int(phi.shape[0])  # type: ignore[index]
            ny_pts = int(phi.shape[1])  # type: ignore[index]
            nz_pts = int(phi.shape[2])  # type: ignore[index]
            x_lc = np.linspace(-sx / 2, sx / 2, nx_pts)
            y_lc = np.linspace(-sy / 2, sy / 2, ny_pts)
            z_lc = np.linspace(-sz / 2, sz / 2, nz_pts)
            phi_interp   = RegularGridInterpolator((x_lc, y_lc, z_lc), phi,
                                                   bounds_error=False, fill_value=np.nan)  # type: ignore[arg-type]
            theta_interp = RegularGridInterpolator((x_lc, y_lc, z_lc), theta,
                                                   bounds_error=False, fill_value=np.nan)  # type: ignore[arg-type]

            def _mat3(v):
                pt = np.array([[float(v.x) - cx, float(v.y), float(v.z)]])
                phi_v   = float(phi_interp(pt)[0])
                theta_v = float(theta_interp(pt)[0])
                d, od = get_dielectric_3d(n_o_sq, n_e_sq, phi_v, theta_v, S)
                return mp.Medium(epsilon_diag=d, epsilon_offdiag=od,
                                 E_susceptibilities=_sted_susc)

            return [mp.Block(
                center=mp.Vector3(cx, 0, 0),
                size=mp.Vector3(sx, sy, sz),
                material=lambda v: _mat3(v),
            )]

    def get_results_2d(self, z_slice=None):
        """Returns (phi, theta, nx, ny, nz) for a single z-slice (default: middle)."""
        phi, theta, nx, ny, nz = self.get_results()
        iz = phi.shape[2] // 2 if z_slice is None else z_slice  # type: ignore[misc]
        return phi[:, :, iz], theta[:, :, iz], nx[:, :, iz], ny[:, :, iz], nz[:, :, iz]

    def save_fields(self):
        """Save director field arrays + grid coordinates to lc_fields.npz."""
        try:
            import meep as mp
            if not mp.am_master():
                return
        except ImportError:
            pass
        phi, theta, *_ = self.get_results()
        sx, sy, sz = self._cell_size()
        nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]  # type: ignore[misc]
        x = np.linspace(-sx / 2, sx / 2, nx)
        y = np.linspace(-sy / 2, sy / 2, ny)
        z = np.linspace(-sz / 2, sz / 2, nz)
        out = self.folder / "simulation"
        out.mkdir(exist_ok=True)
        np.savez(out / "lc_fields.npz", phi=phi, theta=theta, x=x, y=y, z=z)
        self.plot_boundaries()

    def plot_boundaries(self):
        """Plot boundary face values from saved lc_fields.npz as 6×2 grid (face per row)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        npz = self.folder / "simulation" / "lc_fields.npz"
        if not npz.exists():
            print("plot_boundaries: lc_fields.npz not found, skipping")
            return
        d     = np.load(npz)
        phi   = d["phi"]    # (nx, ny, nz)
        theta = d["theta"]
        x, y, z = d["x"], d["y"], d["z"]

        # (phi_slice, theta_slice, h_label, v_label, h_arr, v_arr)
        face_info = {
            "x_min": (phi[0,  :, :], theta[0,  :, :], "y (µm)", "z (µm)", y, z),
            "x_max": (phi[-1, :, :], theta[-1, :, :], "y (µm)", "z (µm)", y, z),
            "y_min": (phi[:,  0, :], theta[:,  0, :], "x (µm)", "z (µm)", x, z),
            "y_max": (phi[:, -1, :], theta[:, -1, :], "x (µm)", "z (µm)", x, z),
            "z_min": (phi[:, :,  0], theta[:, :,  0], "x (µm)", "y (µm)", x, y),
            "z_max": (phi[:, :, -1], theta[:, :, -1], "x (µm)", "y (µm)", x, y),
        }
        keys = list(face_info.keys())

        # row heights proportional to face vertical extents; col widths to horizontal
        wy = float(abs(y[-1] - y[0]))
        wx = float(abs(x[-1] - x[0]))
        wz = float(abs(z[-1] - z[0]))
        # vertical extents per face row (z for x/y faces, y for z faces)
        row_h = [wz, wz, wz, wz, wy, wy]
        # horizontal extents per face (same for both phi/theta columns)
        col_face_w = [wy, wy, wx, wx, wx, wx]
        max_col_w  = max(col_face_w)
        scale = 0.5
        fig_w = max_col_w * 2 * scale + 3.5
        fig_h = sum(row_h)  * scale + 4.0

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = fig.add_gridspec(6, 2, height_ratios=row_h,
                               hspace=0.6, wspace=0.35)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(6)]

        col_meta = [(0, "phi  (0→π)",    "hsv",    np.pi),
                    (1, "theta  (0→π/2)", "plasma", np.pi / 2)]

        for ci, col_label, cmap, vmax in col_meta:
            last_im = None
            for row, key in enumerate(keys):
                ax = axes[row][ci]
                fp_sl, ft_sl, h_lbl, v_lbl, h_arr, v_arr = face_info[key]
                img    = fp_sl if ci == 0 else ft_sl
                extent = [h_arr[0], h_arr[-1], v_arr[0], v_arr[-1]]
                last_im = ax.imshow(img.T, origin="lower", cmap=cmap,
                                    vmin=0, vmax=vmax, aspect="equal",
                                    extent=extent)
                ax.set_xlabel(h_lbl, fontsize=7)
                ax.set_ylabel(v_lbl, fontsize=7)
                ax.tick_params(labelsize=6)
                if ci == 0:
                    ax.set_ylabel(f"{key}\n{v_lbl}", fontsize=7)
            axes[0][ci].set_title(col_label, fontsize=9)
            fig.colorbar(last_im, ax=[axes[r][ci] for r in range(6)],
                         label=col_label, shrink=0.6, pad=0.02)

        fig.suptitle(
            f"Boundaries — {self.boundary_function or 'fixed'}, seed={self.boundary_seed}",
            fontsize=10)
        fig.tight_layout()
        fig_dir = self.folder / "figures"
        fig_dir.mkdir(exist_ok=True)
        out = fig_dir / "boundary_conditions.png"
        fig.savefig(str(out), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/test3D")
    args = parser.parse_args()
    r = Reservoir(args.path)
    r.run_minimization()
    r.save_fields()
    print(f"Done. Fields saved to {r.folder / 'simulation' / 'lc_fields.npz'}")

