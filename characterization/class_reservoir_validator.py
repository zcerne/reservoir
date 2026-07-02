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
import n7_dimention_expansion as n7


class Validator:
    def __init__(self, reservoir_path):
        self.path = reservoir_path
        self.datasets = os.path.join(reservoir_path, "datasets")
        self.stats_dir = os.path.join(reservoir_path, "stats_data")
        self.results = {}

    # ------------------------------------------------------------------ io
    def _load(self, name):
        p = os.path.join(self.datasets, name)
        return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None

    def _load_stats(self, name):
        """Load cached analysis results from stats_data/<name>.npz."""
        p = os.path.join(self.stats_dir, f"{name}.npz")
        if not os.path.exists(p):
            return None
        return dict(np.load(p, allow_pickle=True))

    def _save_stats(self, name, **kwargs):
        """Save analysis results to stats_data/<name>.npz (skips None values)."""
        os.makedirs(self.stats_dir, exist_ok=True)
        clean = {k: v for k, v in kwargs.items() if v is not None}
        if clean:
            np.savez(os.path.join(self.stats_dir, f"{name}.npz"), **clean)

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
        cached = self._load_stats("modes")
        if cached is not None:
            for k, v in cached.items():
                self.results[k] = v
            return cached.get("m1_bla")
        Xin = Yout = None
        ipc = self._load("ipc.npz")
        if ipc is not None and ipc.get("outputs") is not None and np.iscomplexobj(ipc["outputs"]):
            Xin, Yout = ipc["inputs"], ipc["outputs"]
        else:
            d = self._load("superposition.npz")
            if d is not None and d.get("out1") is not None and np.iscomplexobj(d["out1"]):
                Xin = np.concatenate([d["E1"], d["E2"]], axis=0)
                Yout = np.concatenate([d["out1"], d["out2"]], axis=0)
        if Xin is None:
            return None
        res = m1.best_linear_approx({"inputs": Xin, "outputs": Yout}, test_frac=0.3)
        self.results["m1_bla"] = res
        self.results["m2_pca"] = m2.covariance_pca({"inputs": Xin, "outputs": Yout})
        self.results["m3_sum"] = m3.sum_rule(res["G"])
        self.results["m3_mix"] = m3.mixing(res["G"], s=res["s"], Vh=res["Vh"])
        self._save_stats("modes", m1_bla=res,
                         m2_pca=self.results["m2_pca"],
                         m3_sum=self.results["m3_sum"],
                         m3_mix=self.results["m3_mix"])
        return res

    # -------------------------------------------------- NONLINEARITY (n1–n7)
    def superposition(self):
        cached = self._load_stats("n1")
        if cached is not None:
            for k, v in cached.items(): self.results[k] = v
            return cached.get("n1_intensity")
        d = self._load("superposition.npz")
        if d is None: return None
        self.results["n1_field"] = n1.super_position_test(d)
        self.results["n1_intensity"] = n1.super_position_test(
            self._to_intensity(d, ("out1", "out2", "out_combo")))
        self._save_stats("n1", n1_field=self.results["n1_field"],
                         n1_intensity=self.results["n1_intensity"])
        return self.results["n1_intensity"]

    def linear_residual(self):
        d = self._load("ipc.npz")
        if d is None: return None
        cache_name = "n2_field" if np.iscomplexobj(d["outputs"]) else "n2"
        cached = self._load_stats(cache_name)
        if cached is not None:
            for k, v in cached.items(): self.results[k] = v
            return cached.get("n2_intensity" if "field" in cache_name else "n2")
        if np.iscomplexobj(d["outputs"]):
            self.results["n2_field"] = n2.linear_residual(d)
            self.results["n2_intensity"] = n2.linear_residual(self._to_intensity(d, ("outputs",)))
            self._save_stats("n2_field", n2_field=self.results["n2_field"],
                             n2_intensity=self.results["n2_intensity"])
            return self.results["n2_intensity"]
        self.results["n2"] = n2.linear_residual(d)
        self._save_stats("n2", n2=self.results["n2"])
        return self.results["n2"]

    def amplitude(self):
        cached = self._load_stats("n3")
        if cached is not None:
            for k, v in cached.items(): self.results[k] = v
            return cached.get("n3_intensity")
        d = self._load("amp_sweep.npz")
        if d is None: return None
        self.results["n3_field"] = n3.amplitude_dependance(d)
        self.results["n3_intensity"] = n3.amplitude_dependance(self._to_intensity(d, ("outputs",)))
        self._save_stats("n3", n3_field=self.results["n3_field"],
                         n3_intensity=self.results["n3_intensity"])
        return self.results["n3_intensity"]

    def harmonics(self):
        cached = self._load_stats("n4")
        if cached is not None:
            for k, v in cached.items(): self.results[k] = v
            return cached.get("n4_intensity")
        d = self._load("harmonics.npz")
        if d is None: return None
        self.results["n4_field"] = n4.harmonic_specter(d)
        self.results["n4_intensity"] = n4.harmonic_specter(self._to_intensity(d, ("outputs",)))
        self._save_stats("n4", n4_field=self.results["n4_field"],
                         n4_intensity=self.results["n4_intensity"])
        return self.results["n4_intensity"]

    def volterra(self):
        d = self._load("ipc.npz")
        if d is None: return None
        cache_name = "n5_field" if np.iscomplexobj(d["outputs"]) else "n5"
        cached = self._load_stats(cache_name)
        if cached is not None:
            for k, v in cached.items(): self.results[k] = v
            return cached.get("n5_intensity" if "field" in cache_name else "n5")
        if np.iscomplexobj(d["outputs"]):
            self.results["n5_field"] = n5.volterra_series(d, degree=2)
            self.results["n5_intensity"] = n5.volterra_series(self._to_intensity(d, ("outputs",)), degree=2)
            self._save_stats("n5_field", n5_field=self.results["n5_field"],
                             n5_intensity=self.results["n5_intensity"])
            return self.results["n5_intensity"]
        self.results["n5"] = n5.volterra_series(d, degree=2)
        self._save_stats("n5", n5=self.results["n5"])
        return self.results["n5"]

    def dambre(self):
        cached = self._load_stats("n6")
        if cached is not None:
            for k, v in cached.items(): self.results[k] = v
            return cached.get("n6")
        d = self._load("ipc.npz")
        if d is None: return None
        di = self._to_intensity(d, ("outputs",)) if np.iscomplexobj(d["outputs"]) else d
        self.results["n6"] = n6.dambre_ipc(di, max_degree=3)
        self._save_stats("n6", n6=self.results["n6"])
        return self.results["n6"]

    def dimension_expansion(self):
        d = self._load("ipc.npz")
        if d is None: return None
        cache_name = "n7_field" if np.iscomplexobj(d["outputs"]) else "n7"
        cached = self._load_stats(cache_name)
        if cached is not None:
            for k, v in cached.items(): self.results[k] = v
            return cached.get("n7_intensity" if "field" in cache_name else "n7")
        if np.iscomplexobj(d["outputs"]):
            self.results["n7_field"] = n7.dimension_expansion(d)
            self.results["n7_intensity"] = n7.dimension_expansion(
                self._to_intensity(d, ("outputs",)))
            self._save_stats("n7_field", n7_field=self.results["n7_field"],
                             n7_intensity=self.results["n7_intensity"])
            return self.results["n7_intensity"]
        self.results["n7"] = n7.dimension_expansion(d)
        self._save_stats("n7", n7=self.results["n7"])
        return self.results["n7"]

    # ------------------------------------------------------------- orchestrate
    def run_all(self):
        for step in (self.modes, self.superposition, self.linear_residual,
                     self.amplitude, self.harmonics, self.volterra, self.dambre,
                     self.dimension_expansion):
            try:
                if step() is None:
                    print(f"[validator] {step.__name__}: dataset missing — skipped", flush=True)
            except Exception as e:
                print(f"[validator] {step.__name__} FAILED: {e}", flush=True)
        return self.results

    def report(self):
        R = self.results
        rep = {"m1_bla": m1.report, "m2_pca": m2.report,
               "n1_field": n1.report, "n1_intensity": n1.report,
               "n2": n2.report, "n2_field": n2.report, "n2_intensity": n2.report,
               "n3_field": n3.report, "n3_intensity": n3.report,
               "n4_field": n4.report, "n4_intensity": n4.report,
               "n5": n5.report, "n5_field": n5.report, "n5_intensity": n5.report,
               "n6": n6.report, "n7": n7.report, "n7_field": n7.report,
               "n7_intensity": n7.report}
        lines = [f"=== Reservoir characterization: {self.path} ==="]
        for k in ("m1_bla", "m2_pca"):
            if k in R:
                lines.append(f"\n[MODES {k}]\n" + rep[k](R[k]))
        if "m3_sum" in R:
            lines.append("\n[MODES m3]\n" + m3.report(R["m3_sum"], R.get("m3_mix")))
        for k in ("n1_field", "n1_intensity", "n2", "n2_field", "n2_intensity",
                  "n3_field", "n3_intensity", "n4_field", "n4_intensity",
                  "n5", "n5_field", "n5_intensity", "n6",
                  "n7", "n7_field", "n7_intensity"):
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
