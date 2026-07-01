"""Generate a dataset for the superposition (linearity) test — Nonlinearity Method A.

For a FIXED reservoir (relaxed LC texture), the optical input→output map is probed
at random input light fields and their linear combinations. Per trial we need three
forward runs: f(E₁), f(E₂), and f(αE₁+βE₂). The dataset this writes is consumed by
`characterization/n1_superposition.super_position_test`.

Efficiency: instead of 3 sims/trial, draw a POOL of `--n_base` random input vectors,
run each ONCE (n_base sims), cache their outputs, then form `--n_trials` combinations
by picking two pool members + random α, β and running only the combined input
(1 sim/trial). Total = n_base + n_trials sims. At save time out1/out2 are expanded
from the cached pool outputs so the file matches the test's expected keys.

Each forward run = one MEEP run via `SimulationT._run_basis(amplitude_list)` on the
area source (per-strip complex amplitudes), reading the complex sensor field at
monitor_2. The reservoir design (`simulation_data.json` + relaxed LC) must already
exist at --path (build it with run_voltage_reservoir / the T-matrix pipeline).

  python data_gen/generate_superposition_data.py --path data/test2D \
      --n_base 8 --n_trials 40 --out data/test2D/superposition.npz --seed 0

Then:  from n1_superposition import super_position_test, report
        d = dict(np.load("data/test2D/superposition.npz"))
        print(report(super_position_test(d)))          # field linearity
        # intensity (actual readout): square the outputs first
        for k in ("out1","out2","out_combo"): d[k] = np.abs(d[k])**2
        print(report(super_position_test(d)))          # |E|² nonlinearity
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np


def _real(rng, shape, scale):
    # REAL amplitudes: the reservoir source (class_source) casts amplitude to float,
    # so complex phase is discarded. Real inputs fully test linearity/superposition.
    return rng.normal(size=shape) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="reservoir design dir (simulation_data.json + relaxed LC)")
    ap.add_argument("--out", default=None, help="output npz (default <path>/superposition.npz)")
    ap.add_argument("--n_base", type=int, default=8, help="pool of base input vectors (1 sim each)")
    ap.add_argument("--n_trials", type=int, default=40, help="combination trials (1 sim each)")
    ap.add_argument("--scale", type=float, default=1.0, help="input field amplitude")
    ap.add_argument("--components", default="Ey", help="comma list of sensor components to save (Ey[,Ex,Ez])")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from class_simulation_T import SimulationT
    try:
        import meep as mp
        is_master = bool(mp.am_master())
    except Exception:
        is_master = True
    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    out_path = args.out or os.path.join(args.path, "superposition.npz")

    sim = SimulationT(args.path)
    # infer #source strips from the JSON amplitude length
    sim._set_data()
    src_key = sim._source_key(sim.args)
    amp = sim.args[src_key].get("amplitude", [1.0])
    n_strips = len(amp) if isinstance(amp, (list, tuple)) else 1
    print(f"[supdata] reservoir={args.path}  n_strips={n_strips}  "
          f"base={args.n_base} trials={args.n_trials} comps={comps}", flush=True)

    def forward(E):
        """E (n_strips,) complex → stacked sensor field (len(comps)*N_y,) complex."""
        Ey, Ex, Ez = sim._run_basis(list(E))
        fields = {"Ey": Ey, "Ex": Ex, "Ez": Ez}
        return np.concatenate([np.asarray(fields[c]).ravel() for c in comps])

    rng = np.random.default_rng(args.seed)

    # --- base pool: one sim each ---
    E_base = _real(rng, (args.n_base, n_strips), args.scale)
    base_out = []
    for i in range(args.n_base):
        base_out.append(forward(E_base[i]))
        print(f"[supdata] base {i+1}/{args.n_base}", flush=True)
    base_out = np.stack(base_out)                              # (n_base, f_out)

    # --- combination trials: one sim each ---
    E1, E2, alpha, beta, out1, out2, out_combo = [], [], [], [], [], [], []
    for t in range(args.n_trials):
        i, j = rng.choice(args.n_base, size=2, replace=False)
        a, b = _real(rng, (), args.scale / max(args.scale, 1.0)), _real(rng, (), 1.0)
        combo_in = a * E_base[i] + b * E_base[j]
        oc = forward(combo_in)
        E1.append(E_base[i]); E2.append(E_base[j]); alpha.append(a); beta.append(b)
        out1.append(base_out[i]); out2.append(base_out[j]); out_combo.append(oc)
        print(f"[supdata] trial {t+1}/{args.n_trials}", flush=True)

    if is_master:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(out_path,
                 E1=np.stack(E1), E2=np.stack(E2),
                 alpha=np.asarray(alpha), beta=np.asarray(beta),
                 out1=np.stack(out1), out2=np.stack(out2), out_combo=np.stack(out_combo),
                 components=np.asarray(comps), n_strips=n_strips)
        print(f"[supdata] DONE → {out_path}  ({args.n_trials} trials, "
          f"{args.n_base + args.n_trials} sims total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
