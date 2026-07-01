"""
Analytical Fabry-Pérot cavity model for LC reservoir + SLM inside cavity.

Physics (SLM encodes input data, source is fixed uniform illumination):
    source → t₁·mirror_1 → LC_reservoir → SLM(D) → r₂·mirror_2 → detector
                  ↑________ LC_bwd _______ SLM(D) __________________|

    Round-trip operator: M_rt = r₁r₂ · T_LC_bwd · D² · T_LC_fwd
    Cavity steady state: E_cav = (I - M_rt)⁻¹ · t₁ · E_in
    Output field:        E_out = t₂ · D · T_LC_fwd · E_cav
    Detected intensity:  I_out = |E_out|²

    T_LC_bwd = T_LC_fwd^T  (Lorentz reciprocity for symmetric ε tensor)

    Pixel-level round-trip: M_rt = r² · T_Ey^T · diag(exp(2iφ_pixel)) · T_Ey
    Using pixel phases directly (not strip-averaged) preserves
    phase information → M_rt eigenvalues ~0.5 and proper SLM dependence for
    continuous input values (e.g. Iris features normalised to [0,1]).

Prerequisite:
    Build T-matrix for the LC-only path (guides + reservoir, no mirrors/SLM):
        python class_simulation_T.py --path <lc_path> --build-T
    This produces <lc_path>/simulation_T/T_matrix.npz with T_Ey (N_y × N_strips).

Usage:
    python class_cavity_T.py --lc-path data/mirrors --slm-values 1 0 1 0
    python class_cavity_T.py --lc-path data/mirrors --slm-values 0.5 0 0.5 0 --compare
"""
import json
import argparse
import numpy as np
from pathlib import Path


class CavityT:
    """Fabry-Pérot cavity model using precomputed LC strip transfer matrix.

    Strip-level resonance (N_strips × N_strips), pixel-level output (N_y).
    """

    def __init__(self, lc_path: str | Path, cavity_path: str | Path | None = None):
        self.lc_path = Path(lc_path)
        self.cav_path = Path(cavity_path) if cavity_path else self.lc_path
        with open(self.lc_path / "simulation_data.json") as f:
            self.lc_cfg = json.load(f)
        with open(self.cav_path / "simulation_data.json") as f:
            self.cav_cfg = json.load(f)
        self._load_T_matrix()
        self._load_slm_config()
        self._load_mirror_params()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_T_matrix(self):
        t_path = self.lc_path / "simulation_T" / "T_matrix.npz"
        if not t_path.exists():
            raise FileNotFoundError(
                f"T matrix not found: {t_path}\n"
                "Build it first:  python class_simulation_T.py --path <lc_path> --build-T"
            )
        d = np.load(t_path)
        T_Ey = d["T_Ey"]                                # (N_y, N_strips), complex

        # Normalise if built with time-domain FDTD: DFT accumulates E_ss × run_until.
        # Divide by run_until to recover physical field amplitudes (≤1 for passive slab).
        # use_cw and run_until flags are stored in the npz by class_simulation_T.
        built_with_cw  = bool(d["use_cw"])   if "use_cw"    in d.files else False
        run_until_npz  = float(d["run_until"]) if "run_until" in d.files else 0.0
        if not built_with_cw and run_until_npz > 0:
            T_Ey = T_Ey / run_until_npz

        self.T_Ey = T_Ey
        self.N_y, self.N_strips = self.T_Ey.shape

    def _load_slm_config(self):
        for key in self.cav_cfg.get("object_order", []):
            obj = self.cav_cfg.get(key, {})
            if obj.get("class") == "slm":
                self.n_areas = int(obj["number_of_areas"])
                return
        self.n_areas = getattr(self, "N_strips", 4)

    def _load_mirror_params(self):
        for key in self.cav_cfg.get("object_order", []):
            obj = self.cav_cfg.get(key, {})
            if obj.get("class") == "mirror":
                T_m = float(obj.get("transmission", 0.5))
                self.r = np.sqrt(max(0.0, 1.0 - T_m))
                self.t = np.sqrt(T_m)
                return
        self.r, self.t = np.sqrt(0.5), np.sqrt(0.5)

    # ------------------------------------------------------------------
    # SLM phase
    # ------------------------------------------------------------------

    def _strip_phases(self, area_values) -> np.ndarray:
        """area_values[k] ∈ [0,1] → phase[k] = val·π  (val=1 → π half-wave)."""
        return np.array([float(np.clip(v, 0.0, 1.0)) * np.pi
                         for v in list(area_values)[:self.N_strips]])

    def _pixel_phases(self, area_values) -> np.ndarray:
        """Per-pixel phase array (N_y,): each strip block gets its phase."""
        phases = np.zeros(self.N_y)
        edges = np.round(np.linspace(0, self.N_y, self.N_strips + 1)).astype(int)
        for i, val in enumerate(list(area_values)[:self.N_strips]):
            phases[edges[i]:edges[i + 1]] = float(np.clip(val, 0.0, 1.0)) * np.pi
        return phases

    # ------------------------------------------------------------------
    # Cavity evaluation
    # ------------------------------------------------------------------

    def apply_cavity(self, area_values) -> tuple[np.ndarray, np.ndarray]:
        """Compute cavity steady-state output for a given SLM pattern.

        Args:
            area_values: list of N_strips floats in [0, 1]
                         (val=0 → 0 phase, val=1 → π phase)

        Returns:
            E_out : complex output field,  shape (N_y,)
            I_out : intensity |E_out|²,    shape (N_y,)
        """
        phi_p = self._pixel_phases(area_values)   # (N_y,)

        # Pixel-level round-trip: D2_pixel = exp(2i·φ_pixel) applied per detector row.
        # M_rt = r² · T_Ey^T · diag(D2_pixel) · T_Ey  (N_strips × N_strips)
        # Equivalent to T_bwd @ diag(D2) @ T_fwd but avoids lossy strip averaging.
        D2_pixel = np.exp(2j * phi_p)                              # (N_y,)
        M_rt = (self.r ** 2) * (self.T_Ey.T @ (D2_pixel[:, None] * self.T_Ey))

        E_in = self.t * np.ones(self.N_strips, dtype=complex)
        E_cav = np.linalg.solve(np.eye(self.N_strips) - M_rt, E_in)

        # pixel-resolution output: apply per-pixel phase then propagate
        E_out = self.t * np.exp(1j * phi_p) * (self.T_Ey @ E_cav)
        return E_out, np.abs(E_out) ** 2

    def batch_apply(self, all_area_values) -> np.ndarray:
        """Apply cavity to many SLM patterns.

        Args:
            all_area_values: (n_samples, N_strips) array

        Returns:
            I_out: (n_samples, N_y) intensity array
        """
        return np.stack([self.apply_cavity(av)[1] for av in all_area_values])

    # ------------------------------------------------------------------
    # T_LC builder
    # ------------------------------------------------------------------

    def _setup_lc_only_folder(self, lc_only_path: str | Path | None = None) -> Path:
        """Create lc_only/ folder: stripped JSON + lc_fields.npz symlink.

        No MEEP runs — safe to call from Slurm before the mpirun step.
        Returns the lc_only Path.
        """
        import copy
        import os

        lc_only = Path(lc_only_path) if lc_only_path else self.cav_path / "lc_only"
        lc_only.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(self.cav_cfg)
        order = cfg.get("object_order", [])
        skip_classes = {"mirror", "slm"}
        new_order = [k for k in order
                     if cfg.get(k, {}).get("class") not in skip_classes]
        cfg["object_order"] = new_order
        for key in list(order):
            if cfg.get(key, {}).get("class") in skip_classes:
                del cfg[key]
        for key in new_order:
            if cfg.get(key, {}).get("class") == "source":
                cfg[key]["amplitude"] = [1.0] * self.n_areas
                break

        # solve_cw gives actual steady-state field amplitudes (not DFT-integrated)
        # → T_Ey values are physically normalised → M_rt eigenvalues bounded correctly
        cfg["use_cw"] = True
        cfg["cw_init_time"] = int(cfg.get("cw_init_time", 200))

        with open(lc_only / "simulation_data.json", "w") as f:
            json.dump(cfg, f, indent=4)

        sim_dir = lc_only / "simulation"
        sim_dir.mkdir(exist_ok=True)
        src = self.cav_path / "simulation" / "lc_fields.npz"
        dst = sim_dir / "lc_fields.npz"
        if not dst.exists():
            if src.exists():
                os.symlink(src.resolve(), dst)
            else:
                raise FileNotFoundError(
                    f"lc_fields.npz not found: {src}\n"
                    "Run LC minimization first: python class_reservoir.py --path <cavity_path>"
                )
        print(f"lc_only folder ready: {lc_only}")
        return lc_only

    def build_T_LC(self, n_procs: int = 16, lc_only_path: str | Path | None = None):
        """Build T_LC: create lc_only/ folder then run MEEP T-matrix basis runs.

        After completion, reloads T_matrix and updates self.lc_path.

        Args:
            n_procs:      number of MPI processes
            lc_only_path: override output folder (default: <cavity_path>/lc_only)
        """
        import sys
        import shutil
        import subprocess

        lc_only = self._setup_lc_only_folder(lc_only_path)

        python = sys.executable
        mpirun = shutil.which("mpirun") or "mpirun"
        script = Path(__file__).parent / "class_simulation_T.py"
        t_log = lc_only / "simulation_T" / "build_T.log"
        t_log.parent.mkdir(exist_ok=True)
        cmd = [mpirun, "-np", str(n_procs), python, str(script),
               "--path", str(lc_only), "--build-T"]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode != 0:
            raise RuntimeError(f"T-matrix build failed (exit {result.returncode})")

        self.lc_path = lc_only
        with open(lc_only / "simulation_data.json") as f:
            self.lc_cfg = json.load(f)
        self._load_T_matrix()
        print(f"T_LC loaded: {self.T_Ey.shape}  →  {lc_only}/simulation_T/T_matrix.npz")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def compare_to_meep(self, area_values, sim_dir: str | Path | None = None) -> float:
        """Compare I_out against direct MEEP run; return Pearson correlation."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, I_model = self.apply_cavity(area_values)
        sim_dir = Path(sim_dir) if sim_dir else self.cav_path / "simulation"
        mon = np.load(sim_dir / "monitor_2.npz")
        key = "Ey"
        E_meep = mon[key][0]
        I_meep = np.abs(E_meep) ** 2
        # Normalise MEEP to same scale as model: model uses solve_cw amplitudes,
        # MEEP may be time-domain DFT. Scale by RMS ratio so correlation is meaningful.
        rms_model = float(np.sqrt(np.mean(I_model)))
        rms_meep  = float(np.sqrt(np.mean(I_meep)))
        scale = rms_model / rms_meep if rms_meep > 0 else 1.0
        I_meep_scaled = I_meep * scale

        n = min(len(I_model), len(I_meep_scaled))
        corr = float(np.corrcoef(I_model[:n], I_meep_scaled[:n])[0, 1])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        y = np.linspace(-self.cav_cfg.get("cell_size_y", 20) / 2,
                         self.cav_cfg.get("cell_size_y", 20) / 2, n)
        axes[0].plot(y[:n], I_model[:n], label="cavity model")
        axes[0].plot(y[:n], I_meep_scaled[:n], "--", label=f"MEEP (×{scale:.2e})")
        axes[0].set(xlabel="y (µm)", ylabel="|Ey|²", title=f"corr = {corr:.4f}")
        axes[0].legend()

        axes[1].scatter(I_meep_scaled[:n], I_model[:n], s=2, alpha=0.5)
        axes[1].set(xlabel="MEEP (scaled)", ylabel="model", title="scatter")

        fig_path = self.cav_path / "figures" / "cavity_T_vs_meep.png"
        fig_path.parent.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Correlation: {corr:.6f}")
        print(f"Saved: {fig_path}")
        return corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc-path", default=None,
                        help="Folder with LC T-matrix (simulation_T/T_matrix.npz); "
                             "required unless --build-T-LC is set")
    parser.add_argument("--cavity-path", default=None,
                        help="Folder with cavity simulation_data.json (defaults to lc-path)")
    parser.add_argument("--slm-values", nargs="+", type=float, default=[1, 0, 1, 0],
                        help="SLM area values in [0,1]")
    parser.add_argument("--compare", action="store_true",
                        help="Compare cavity model against MEEP monitor_2")
    parser.add_argument("--build-T-LC", action="store_true",
                        help="Build T_LC matrix (LC-only MEEP runs) then exit")
    parser.add_argument("--setup-lc-only", action="store_true",
                        help="Create lc_only/ JSON + symlink only (no MEEP); for Slurm use")
    parser.add_argument("--n-procs", type=int, default=16,
                        help="MPI processes for --build-T-LC (default 16)")
    args = parser.parse_args()

    ct = CavityT.__new__(CavityT)
    ct.cav_path = Path(args.cavity_path) if args.cavity_path else Path(args.lc_path)
    with open(ct.cav_path / "simulation_data.json") as f:
        ct.cav_cfg = json.load(f)
    ct._load_slm_config()
    ct._load_mirror_params()

    if args.build_T_LC or getattr(args, "setup_lc_only", False):
        ct.lc_path = ct.cav_path
        with open(ct.cav_path / "simulation_data.json") as f:
            ct.lc_cfg = json.load(f)
        if getattr(args, "setup_lc_only", False):
            # JSON + symlink only — no MEEP (for use inside Slurm before mpirun step)
            ct._setup_lc_only_folder()
        else:
            ct.build_T_LC(n_procs=args.n_procs)
        import sys; sys.exit(0)

    if args.lc_path is None:
        raise SystemExit("--lc-path is required unless --build-T-LC is set")
    ct.lc_path = Path(args.lc_path)
    with open(ct.lc_path / "simulation_data.json") as f:
        ct.lc_cfg = json.load(f)
    ct._load_T_matrix()
    E_out, I_out = ct.apply_cavity(args.slm_values)
    print(f"N_y = {ct.N_y}, N_strips = {ct.N_strips}")
    print(f"Mirror: r = {ct.r:.3f}, t = {ct.t:.3f}")
    print(f"SLM phases (rad): {ct._strip_phases(args.slm_values).round(3)}")
    print(f"I_out: min={I_out.min():.3f}, max={I_out.max():.3f}, sum={I_out.sum():.1f}")

    if args.compare:
        ct.compare_to_meep(args.slm_values)
