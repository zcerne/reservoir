import gc
import json
import os

import meep as mp
import numpy as np

from class_simulation import Simulation


class SimulationT(Simulation):
    """T-matrix characterisation of the LC reservoir.

    Instead of one MEEP run per input pattern, measure the transfer matrix T
    once (N_strips basis runs), then evaluate any input instantly as E_out = T @ amplitude.

    Output files are saved to  <folder>/simulation_T/  so they never overwrite
    the full time-domain simulation results.

    T_matrix.npz   — T_Ey, T_Ex, T_Ez: shape (N_y, N_strips), complex
    training_data.npz — amplitudes (N_strips × N_strips identity), E_out_*/I_out_* per basis
    result_T.npz   — E_out_*, I_out for a specific apply_T() call
    """

    def __init__(self, args_path: str) -> None:
        super().__init__(args_path)
        self.T_dir = os.path.join(args_path, "simulation_T")
        os.makedirs(self.T_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_sensor(self, name: str):
        for s in self.sensors:
            if s._name == name:
                return s
        raise ValueError(f"Sensor '{name}' not found in simulation")

    def _source_key(self, cfg: dict) -> str:
        return next(
            k for k, v in cfg.items()
            if isinstance(v, dict) and v.get("class") == "source"
        )

    def _reset_state(self) -> None:
        self.objects_args = []
        self.objects      = []
        self.sources      = []
        self.sensors      = []
        self.args         = {}
        self.snapshots    = {"x": [], "y": [], "z": [], "t": []}
        self.simulation   = None
        gc.collect()

    # ------------------------------------------------------------------
    # Basis run
    # ------------------------------------------------------------------

    def _run_basis(self, amplitude_list: list) -> tuple:
        """One MEEP run with given amplitude. Returns complex (Ey, Ex, Ez) at monitor_2."""
        self._reset_state()
        # Load JSON then override amplitude before building objects
        self._set_data()
        key = self._source_key(self.args)
        self.args[key]["amplitude"] = list(amplitude_list)
        # Build and run (no _save_all — we extract directly)
        self._set_simulation_parameters()
        self._set_object_list()
        self._set_pmls()
        self._set_cell()
        self._set_geometry()
        self._set_simulation()
        self._setup_sensors()
        self._run_meep_once()
        # Extract complex DFT field from monitor_2
        m2 = self._find_sensor("monitor_2")
        h  = m2._monitor_handle
        Ey = np.array(self.simulation.get_dft_array(h, mp.Ey, 0))
        Ex = np.array(self.simulation.get_dft_array(h, mp.Ex, 0))
        Ez = np.array(self.simulation.get_dft_array(h, mp.Ez, 0))
        return Ey, Ex, Ez

    # ------------------------------------------------------------------
    # Build T matrix
    # ------------------------------------------------------------------

    def build_T_matrix(self):
        """Run N_strips basis simulations and assemble the transfer matrix.

        Saves:
          simulation_T/T_matrix.npz      — T_Ey/T_Ex/T_Ez, shape (N_y, N_strips)
          simulation_T/training_data.npz — amplitudes + E_out/I_out per basis input

        Returns T_Ey (dominant component for Ey source).
        """
        with open(os.path.join(self.folder_path, "simulation_data.json")) as f:
            cfg = json.load(f)
        amplitude = cfg[self._source_key(cfg)].get("amplitude", [1.0])
        n_strips  = len(amplitude) if isinstance(amplitude, list) else 1

        cols_Ey: list = []
        cols_Ex: list = []
        cols_Ez: list = []

        t_path = os.path.join(self.T_dir, "T_matrix.npz")

        for i in range(n_strips):
            basis = [0.0] * n_strips
            basis[i] = 1.0
            if mp.am_master():
                print(f"[SimulationT] basis run {i+1}/{n_strips}  amplitude={basis}")
            Ey, Ex, Ez = self._run_basis(basis)
            cols_Ey.append(Ey)
            cols_Ex.append(Ex)
            cols_Ez.append(Ez)

            # Save after each basis run so data is not lost on crash
            if mp.am_master():
                T_partial_Ey = np.column_stack(cols_Ey) if len(cols_Ey) > 1 else cols_Ey[0].reshape(-1, 1)
                T_partial_Ex = np.column_stack(cols_Ex) if len(cols_Ex) > 1 else cols_Ex[0].reshape(-1, 1)
                T_partial_Ez = np.column_stack(cols_Ez) if len(cols_Ez) > 1 else cols_Ez[0].reshape(-1, 1)
                np.savez(t_path, T_Ey=T_partial_Ey, T_Ex=T_partial_Ex, T_Ez=T_partial_Ez,
                         n_complete=i + 1, n_total=n_strips)
                print(f"[SimulationT] saved T_matrix.npz after basis {i+1}/{n_strips}")

        T_Ey = np.column_stack(cols_Ey)   # (N_y, N_strips), complex
        T_Ex = np.column_stack(cols_Ex)
        T_Ez = np.column_stack(cols_Ez)

        # Overwrite with final complete matrix (n_complete == n_total)
        if mp.am_master():
            np.savez(t_path, T_Ey=T_Ey, T_Ex=T_Ex, T_Ez=T_Ez,
                     n_complete=n_strips, n_total=n_strips)

        # Training data: basis inputs as rows, corresponding outputs as rows
        inputs = np.eye(n_strips, dtype=float)
        td_path = os.path.join(self.T_dir, "training_data.npz")
        np.savez(
            td_path,
            amplitudes=inputs,            # (N_strips, N_strips) identity
            E_out_Ey=T_Ey.T,             # (N_strips, N_y) complex
            E_out_Ex=T_Ex.T,
            E_out_Ez=T_Ez.T,
            I_out_Ey=np.abs(T_Ey.T)**2,  # (N_strips, N_y) real
            I_out_Ex=np.abs(T_Ex.T)**2,
            I_out_Ez=np.abs(T_Ez.T)**2,
            I_out=np.abs(T_Ey.T)**2 + np.abs(T_Ex.T)**2 + np.abs(T_Ez.T)**2,
        )

        if mp.am_master():
            print(f"Saved T_matrix.npz   shape={T_Ey.shape}")
            print(f"Saved training_data.npz")
        return T_Ey, T_Ex, T_Ez

    # ------------------------------------------------------------------
    # Apply T
    # ------------------------------------------------------------------

    def apply_T(self, amplitude) -> tuple:
        """Multiply T @ amplitude and save result.

        Saves simulation_T/result_T.npz.
        Returns (E_out_Ey, E_out_Ex, E_out_Ez, I_out_total).
        """
        t_path = os.path.join(self.T_dir, "T_matrix.npz")
        if not os.path.exists(t_path):
            raise FileNotFoundError(f"T_matrix.npz not found — run build_T_matrix() first")
        d = np.load(t_path)
        a = np.array(amplitude, dtype=complex)

        E_Ey = d["T_Ey"] @ a
        E_Ex = d["T_Ex"] @ a
        E_Ez = d["T_Ez"] @ a
        I_out = np.abs(E_Ey)**2 + np.abs(E_Ex)**2 + np.abs(E_Ez)**2

        np.savez(
            os.path.join(self.T_dir, "result_T.npz"),
            amplitude=np.array(amplitude),
            E_out_Ey=E_Ey,
            E_out_Ex=E_Ex,
            E_out_Ez=E_Ez,
            I_out=I_out,
        )
        return E_Ey, E_Ex, E_Ez, I_out

    # ------------------------------------------------------------------
    # Override run_simulation → apply T
    # ------------------------------------------------------------------

    def run_simulation(self) -> None:
        """Apply pre-built T matrix to current source amplitude (no MEEP run)."""
        with open(os.path.join(self.folder_path, "simulation_data.json")) as f:
            cfg = json.load(f)
        amplitude = cfg[self._source_key(cfg)].get("amplitude", [1.0])
        self.apply_T(amplitude)
        if mp.am_master():
            print(f"Applied T matrix for amplitude={amplitude}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/test2D")
    parser.add_argument("--build-T", action="store_true", help="Measure T matrix (N basis MEEP runs)")
    parser.add_argument("--apply", action="store_true", help="Apply existing T to current amplitude")
    args = parser.parse_args()

    sim = SimulationT(args.path)
    if args.build_T:
        sim.build_T_matrix()
    elif args.apply:
        sim.run_simulation()
    else:
        print("Use --build-T to measure T matrix, or --apply to apply it.")
