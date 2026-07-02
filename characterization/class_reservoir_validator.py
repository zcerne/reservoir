"""Validator — run the full characterization suite for ONE reservoir.

Loads the generated datasets from <reservoir_path>/datasets/ and runs every
MODES (m1–m3, capacity) + NONLINEARITY (n1–n6) analysis, storing results and
producing a combined report. See [[RC - How good is reservoir]] for the methods.

Dataset → method map (datasets produced by data_gen/generate_*_data.py):
    superposition.npz → n1  (+ its complex base pairs build the linear operator G for MODES)
    amp_sweep.npz     → n3
    harmonics.npz     → n4
    ipc.npz           → n2, n5, n6   (Uniform[-1,1] probes; intensity readout)

MODES need the *field* operator G (linear); nonlinearity n2/n5/n6 use the ipc
{inputs,outputs} set. Field outputs are squared to |E|² on demand for the
intensity views (n1/n3/n4 report both field and |E|²).

  from class_reservoir_validator import Validator
  v = Validator("data/reservoir_clasifications/01_2D_director")
  v.run_all(); print(v.report())
"""
import os
import numpy as np

import m1_best_linear_approx as m1
import m2_covariance_PCA as m2
import m3_sum_rule_and_mixing as m3
import n1_superposition as n1
import n2_linear_residual as n2
import n3_amplitude_dependant as n3
import n4_harmonics_distortion as n4
import n5_voltera_series as n5
import n6_dambre as n6


class Validator:
    def __init__(self, reservoir_path):
        self.path = reservoir_path
        self.datasets = os.path.join(reservoir_path, "datasets")
        self.results = {}

    # ------------------------------------------------------------------ io
    def _load(self, name):
        p = os.path.join(self.datasets, name)
        return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None

    @staticmethod
    def _to_intensity(d, keys):
        """Copy of d with the given output keys squared to |E|² (no-op if already real)."""
        out = dict(d)
        for k in keys:
            if k in out and np.iscomplexobj(out[k]):
                out[k] = np.abs(out[k]) ** 2
        return out

    # ------------------------------------------------------- MODES (capacity)
    def modes(self):
        """Build the linear field operator G from the superposition base pairs (complex
        field) → m1 (BLA+SVD), m2 (PCA), m3 (sum-rule + mixing). Falls back to ipc if
        its outputs are complex (field readout)."""
        Xin = Yout = None
        d = self._load("superposition.npz")
        if d is not None and d.get("out1") is not None and np.iscomplexobj(d["out1"]):
            Xin = np.concatenate([d["E1"], d["E2"]], axis=0)      # E1→out1, E2→out2 field pairs
            Yout = np.concatenate([d["out1"], d["out2"]], axis=0)
        else:
            ipc = self._load("ipc.npz")
            if ipc is not None and ipc.get("outputs") is not None and np.iscomplexobj(ipc["outputs"]):
                Xin, Yout = ipc["inputs"], ipc["outputs"]
        if Xin is None:
            return None                                           # no field data → skip MODES
        res = m1.best_linear_approx({"inputs": Xin, "outputs": Yout}, test_frac=0.3)
        self.results["m1_bla"] = res
        self.results["m2_pca"] = m2.covariance_pca({"inputs": Xin, "outputs": Yout})
        self.results["m3_sum"] = m3.sum_rule(res["G"])
        self.results["m3_mix"] = m3.mixing(res["G"], s=res["s"], Vh=res["Vh"])
        return res

    # -------------------------------------------------- NONLINEARITY (n1–n6)
    def superposition(self):
        d = self._load("superposition.npz")
        if d is None:
            return None
        self.results["n1_field"] = n1.super_position_test(d)
        self.results["n1_intensity"] = n1.super_position_test(
            self._to_intensity(d, ("out1", "out2", "out_combo")))
        return self.results["n1_intensity"]

    def linear_residual(self):
        d = self._load("ipc.npz")
        if d is None:
            return None
        self.results["n2"] = n2.linear_residual(d)
        return self.results["n2"]

    def amplitude(self):
        d = self._load("amp_sweep.npz")
        if d is None:
            return None
        self.results["n3_field"] = n3.amplitude_dependance(d)
        self.results["n3_intensity"] = n3.amplitude_dependance(self._to_intensity(d, ("outputs",)))
        return self.results["n3_intensity"]

    def harmonics(self):
        d = self._load("harmonics.npz")
        if d is None:
            return None
        self.results["n4_field"] = n4.harmonic_specter(d)
        self.results["n4_intensity"] = n4.harmonic_specter(self._to_intensity(d, ("outputs",)))
        return self.results["n4_intensity"]

    def volterra(self):
        d = self._load("ipc.npz")
        if d is None:
            return None
        self.results["n5"] = n5.volterra_series(d, degree=2)
        return self.results["n5"]

    def dambre(self):
        d = self._load("ipc.npz")
        if d is None:
            return None
        self.results["n6"] = n6.dambre_ipc(d, max_degree=3)
        return self.results["n6"]

    # ------------------------------------------------------------- orchestrate
    def run_all(self):
        for step in (self.modes, self.superposition, self.linear_residual,
                     self.amplitude, self.harmonics, self.volterra, self.dambre):
            try:
                if step() is None:
                    print(f"[validator] {step.__name__}: dataset missing — skipped", flush=True)
            except Exception as e:
                print(f"[validator] {step.__name__} FAILED: {e}", flush=True)
        return self.results

    def report(self):
        R = self.results
        rep = {"m1_bla": m1.report, "m2_pca": m2.report,
               "n1_field": n1.report, "n1_intensity": n1.report, "n2": n2.report,
               "n3_field": n3.report, "n3_intensity": n3.report,
               "n4_field": n4.report, "n4_intensity": n4.report,
               "n5": n5.report, "n6": n6.report}
        lines = [f"=== Reservoir characterization: {self.path} ==="]
        for k in ("m1_bla", "m2_pca"):
            if k in R:
                lines.append(f"\n[MODES {k}]\n" + rep[k](R[k]))
        if "m3_sum" in R:
            lines.append("\n[MODES m3]\n" + m3.report(R["m3_sum"], R.get("m3_mix")))
        for k in ("n1_field", "n1_intensity", "n2", "n3_field", "n3_intensity",
                  "n4_field", "n4_intensity", "n5", "n6"):
            if k in R:
                lines.append(f"\n[{k}]\n" + rep[k](R[k]))
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="reservoir dir (has datasets/)")
    a = ap.parse_args()
    v = Validator(a.path)
    v.run_all()
    print(v.report())
