"""Validator — run the full characterization suite for ONE reservoir.

Loads the generated datasets from <reservoir_path>/datasets/ and runs every
MODES (m1–m3, capacity) + NONLINEARITY (n1–n6) analysis, storing results and
producing a combined report. See [[RC - How good is reservoir]] for the methods.

Dataset → method map (datasets produced by data_gen/generate_*_data.py):
    superposition.npz → n1  (+ its complex base pairs build the linear operator G for MODES)
    amp_sweep.npz     → n3
    harmonics.npz     → n4
    ipc.npz           → n2, n5, n6   (Uniform[-1,1] probes; n6 reports BOTH
                        the field and the intensity readout — they carry
                        opposite parities, see dambre())

MODES need the *field* operator G (linear); nonlinearity n2/n5/n6 use the ipc
{inputs,outputs} set. Field outputs are squared to |E|² on demand for the
intensity views (n1/n3/n4 report both field and |E|²).

  from class_reservoir_validator import Validator
  v = Validator("data/reservoir_clasifications/01_2D_director")
  v.run_all(); print(v.report())
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    def __init__(self, reservoir_path, component=None):
        #: restrict analysis to ONE stored polarization (e.g. "Ey"). Datasets
        #: written with --components Ex,Ey,Ez concatenate equal-width blocks
        #: and record the order in their `components` key; without slicing,
        #: every method analyses the pump channel (Ez) mixed in with the
        #: signal — that is how the ad-hoc Ey-only figures on 04/03b differ
        #: from plot_main's. None keeps the historical use-everything
        #: behaviour (also right for datasets with no `components` key,
        #: e.g. the NN references, whose state has no polarizations).
        self.component = component
        self.path = reservoir_path
        self.datasets = self._resolve_datasets(reservoir_path)
        self.stats_dir = os.path.join(reservoir_path, "stats_data")
        self.results = {}

    # ------------------------------------------------------- dataset location
    #: Repo roots searched for a design's datasets/ when the local copy holds no
    #: data. The Nextcloud checkout versions only simulation_data.json and
    #: .gitkeep (data/** is gitignored), so running from there must fall back to
    #: the copy that actually has the .npz files, and vice versa. Override with
    #: RESERVOIR_DATA_ROOTS (os.pathsep-separated).
    DATA_ROOTS = ("~/Orion/resevoir",
                  "~/Nextcloud/Doktorski/Projects/Reservoir/gitcode",
                  "~/resevoir")

    @staticmethod
    def _has_data(d):
        """True only if the dir holds actual datasets. A git-tracked skeleton
        contains just .gitkeep — that must NOT outrank a mirror that has data."""
        return os.path.isdir(d) and any(
            n.endswith((".npz", ".npz.parts")) for n in os.listdir(d))

    @classmethod
    def _mirror_candidates(cls, reservoir_path):
        """The same design under every configured root: a design path always
        contains .../data/<...>/<design>, so mirror on the 'data/' segment."""
        parts = os.path.normpath(os.path.abspath(reservoir_path)).split(os.sep)
        if "data" not in parts:
            return []
        rel = os.path.join(*parts[len(parts) - 1 - parts[::-1].index("data"):])
        roots = os.environ.get("RESERVOIR_DATA_ROOTS")
        roots = roots.split(os.pathsep) if roots else cls.DATA_ROOTS
        return [os.path.join(os.path.expanduser(r), rel) for r in roots]

    @classmethod
    def _resolve_datasets(cls, reservoir_path):
        """The design's primary datasets dir (first one holding data). Individual
        files are still resolved per-file by `_dataset_path`."""
        local = os.path.join(reservoir_path, "datasets")
        if cls._has_data(local):
            return local
        for cand in cls._mirror_candidates(reservoir_path):
            ds = os.path.join(cand, "datasets")
            if os.path.abspath(ds) != os.path.abspath(local) and cls._has_data(ds):
                print(f"[validator] datasets ← {ds}", flush=True)
                return ds
        return local  # keep local for clear "missing" messages

    def _dataset_path(self, name):
        """Locate ONE dataset, searching the primary dir and then every mirror.
        Resolution is per FILE, not per directory, so a design whose data is
        split across machines — say ipc.npz already synced here while
        harmonics.npz is still only on the cluster — still analyses completely."""
        primary = os.path.join(self.datasets, name)
        if os.path.exists(primary):
            return primary
        seen = {os.path.abspath(primary)}
        for cand in self._mirror_candidates(self.path):
            p = os.path.join(cand, "datasets", name)
            if os.path.abspath(p) in seen:
                continue
            seen.add(os.path.abspath(p))
            if os.path.exists(p):
                print(f"[validator] {name} ← {os.path.dirname(p)}", flush=True)
                return p
        return primary  # keep primary for clear "missing" messages

    # ------------------------------------------------------------------ io
    def _load(self, name):
        p = self._dataset_path(name)
        d = dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None
        return self._slice_component(d) if d is not None else None

    def _slice_component(self, d):
        """Cut one polarization's block out of every output-like array."""
        if not self.component or "components" not in d:
            return d
        comps = [str(c) for c in np.asarray(d["components"]).reshape(-1)]
        if self.component not in comps or len(comps) < 2:
            return d
        k = comps.index(self.component)
        out = dict(d)
        for key in ("outputs", "out1", "out2", "out_combo"):
            if key in out:
                a = np.asarray(out[key])
                n = a.shape[-1] // len(comps)
                out[key] = a[..., k * n:(k + 1) * n]
        out["components"] = np.asarray([self.component])
        return out

    def _load_stats(self, name):
        """Load cached analysis results from stats_data/<name>.npz. Each stored value
        is an analysis-result dict; np.savez wraps it as a 0-d object array, so unwrap
        with .item() to recover the nested dict (power_by_order, gain_by_order, …)."""
        p = os.path.join(self.stats_dir, f"{name}.npz")
        if not os.path.exists(p):
            return None
        raw = np.load(p, allow_pickle=True)
        out = {}
        for k in raw.files:
            v = raw[k]
            out[k] = v.item() if getattr(v, "ndim", None) == 0 and v.dtype == object else v
        return out

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
        # MODES = SVD of the input→state operator G. Prefer the ipc probe set (complex
        # FIELD for LC reservoirs, or REAL state for NN references — both give a valid
        # linear operator). Fall back to the superposition base pairs.
        Xin = Yout = None
        ipc = self._load("ipc.npz")
        if ipc is not None and ipc.get("outputs") is not None:
            Xin, Yout = ipc["inputs"], ipc["outputs"]          # complex OR real state
        else:
            d = self._load("superposition.npz")
            if d is not None and d.get("out1") is not None:
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

    @staticmethod
    def _reduce_state(di, label=""):
        """IPC needs a WELL-DETERMINED readout: if the raw state has more channels F
        than probes M, the linear fit reconstructs any target perfectly (spurious
        capacity) and the significance threshold (~2F/M) then rejects everything →
        IPC=0. The physical state is low-rank anyway, so PCA-reduce to K≪M leading
        components (independent readout channels) before the Dambre estimate."""
        X = np.asarray(di["outputs"]); M = X.shape[0]; Xf = X.reshape(M, -1)
        F = Xf.shape[1]
        # Reduce straight to the size the estimate actually wants (M//10, the same
        # cap dambre_ipc would apply). Reducing only to M/4 and letting dambre_ipc
        # subsample the scores afterwards is WRONG: it selects evenly spaced
        # indices, which is right for spatial detectors but not for PCA scores —
        # those are ordered by variance, so skipping leading components throws
        # away the very modes carrying the linear response (03b deg1 fell 2.00 →
        # 0.76 that way).
        k = int(min(F, max(4, M // 10)))
        if F <= k:
            return di
        Xc = Xf - Xf.mean(0, keepdims=True)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        print(f"[validator] dambre{label}: PCA-reduced state {F}→{k} leading "
              f"channels (M={M} probes)", flush=True)
        return {**di, "outputs": Xc @ Vt[:k].conj().T}     # (M, k) PCA scores

    def dambre(self):
        d = self._load("ipc.npz")
        if d is None: return None
        is_cx = np.iscomplexobj(d["outputs"])
        # Field and intensity see DIFFERENT halves of the spectrum: |E|² is even in
        # the drive so it annihilates every odd (linear, cubic…) target, while the
        # field is odd and annihilates the even ones. Reporting only the intensity
        # view hides exactly the orders the medium contributes, so do both — same
        # split as n1/n3/n4. (--component slices polarization; it does not choose
        # the readout.)
        keys = ("n6_field", "n6_intensity") if is_cx else ("n6",)
        cached = self._load_stats("n6")
        if cached is not None and all(k in cached for k in keys):
            for k, v in cached.items(): self.results[k] = v
            return cached.get(keys[-1])
        variants = ([("n6_field", d),
                     ("n6_intensity", self._to_intensity(d, ("outputs",)))]
                    if is_cx else [("n6", d)])
        for key, dv in variants:
            dr = self._reduce_state(dv, f" {key.split('_')[-1]}")
            # _reduce_state already sized the readout; opt out of dambre_ipc's own
            # even-spaced cap so it cannot re-cut PCA scores out of variance order.
            n_ch = np.asarray(dr["outputs"]).reshape(dr["inputs"].shape[0], -1).shape[1]
            self.results[key] = n6.dambre_ipc(dr, max_degree=3, max_features=n_ch)
        self._save_stats("n6", **{k: self.results[k] for k, _ in variants})
        return self.results[keys[-1]]

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
