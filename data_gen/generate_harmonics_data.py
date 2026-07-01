"""Generate harmonic/intermodulation-distortion data — Nonlinearity Method D.

Drive the reservoir with one or two tones via a PHASE-SWEEP parameter t and forward-
run at each t:
    E_in(t) = Σₖ Aₖ · e^{i·toneₖ·t} · uₖ ,   t = 2π·j/N_t ,  j = 0 .. N_t−1
where uₖ is the input pattern each tone rides on (a chosen source strip by default).
Sample N_t points over one period, one MEEP run each, and save the output at every t.
The analysis (`characterization/n4_harmonics_distortion.harmonic_specter`) DFTs the
output over t: a linear field map shows only the fundamental tones; the |E|² readout
adds DC + 2nd-order harmonics/intermod.

Sampling: use well-separated integer tones (e.g. 5,7) and N_t > 2·max_order·max_tone
to avoid aliasing the harmonics (default max_order 6 ⇒ for tones ≤7, N_t ≥ ~90).

  python data_gen/generate_harmonics_data.py --path data/test2D \
      --tones 5,7 --n_t 128 --amps 1,1 --out data/test2D/harmonics.npz

Then:  from n4_harmonics_distortion import harmonic_specter, report
        d = dict(np.load("data/test2D/harmonics.npz"))
        print(report(harmonic_specter(d)))              # field: linear
        d["outputs"] = np.abs(d["outputs"])**2          # |E|² readout
        print(report(harmonic_specter(d)))              # order-2 distortion
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="reservoir design dir (simulation_data.json + relaxed LC)")
    ap.add_argument("--out", default=None, help="output npz (default <path>/harmonics.npz)")
    ap.add_argument("--tones", default="5,7", help="1 or 2 integer tone frequencies, comma-sep")
    ap.add_argument("--amps", default=None, help="per-tone amplitudes (comma-sep); default all 1.0")
    ap.add_argument("--channels", default=None,
                    help="per-tone source-strip index each tone rides on (comma-sep); "
                         "default distinct strips 0,1,...")
    ap.add_argument("--n_t", type=int, default=128, help="phase-sweep samples over one period (1 sim each)")
    ap.add_argument("--components", default="Ey", help="sensor components to save (Ey[,Ex,Ez])")
    args = ap.parse_args()

    from class_simulation_T import SimulationT
    try:
        import meep as mp
        is_master = bool(mp.am_master())
    except Exception:
        is_master = True
    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    tones = [int(t) for t in args.tones.split(",") if t.strip()]
    amps = ([float(a) for a in args.amps.split(",")] if args.amps else [1.0] * len(tones))
    if len(amps) != len(tones):
        raise SystemExit("--amps must have one value per tone")
    out_path = args.out or os.path.join(args.path, "harmonics.npz")

    sim = SimulationT(os.path.join(args.path, "simulation_data.json"))
    sim._set_data()
    src_key = sim._source_key(sim.args)
    amp0 = sim.args[src_key].get("amplitude", [1.0])
    n_strips = len(amp0) if isinstance(amp0, (list, tuple)) else 1
    chans = ([int(c) for c in args.channels.split(",")] if args.channels
             else list(range(len(tones))))
    if len(chans) != len(tones) or max(chans) >= n_strips:
        raise SystemExit(f"--channels must have one valid strip index (<{n_strips}) per tone")
    print(f"[harmdata] reservoir={args.path}  n_strips={n_strips}  tones={tones}  "
          f"amps={amps}  channels={chans}  N_t={args.n_t}  comps={comps}", flush=True)

    def forward(E):
        Ey, Ex, Ez = sim._run_basis(list(E))
        fields = {"Ey": Ey, "Ex": Ex, "Ez": Ez}
        return np.concatenate([np.asarray(fields[c]).ravel() for c in comps])

    # unit input patterns: tone k rides on strip chans[k]
    U = np.zeros((len(tones), n_strips), dtype=complex)
    for k, s in enumerate(chans):
        U[k, s] = 1.0

    t_grid = 2.0 * np.pi * np.arange(args.n_t) / args.n_t
    inputs, outputs = [], []
    for j, t in enumerate(t_grid):
        E = np.zeros(n_strips, dtype=complex)
        for k in range(len(tones)):
            E += amps[k] * np.exp(1j * tones[k] * t) * U[k]
        inputs.append(E)
        outputs.append(forward(E))
        if j % 16 == 0:
            print(f"[harmdata] sweep {j+1}/{args.n_t}", flush=True)

    if is_master:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(out_path,
                 outputs=np.stack(outputs), inputs=np.stack(inputs),
                 t=t_grid, tones=np.asarray(tones), amps=np.asarray(amps),
                 channels=np.asarray(chans), components=np.asarray(comps), n_strips=n_strips)
        print(f"[harmdata] DONE → {out_path}  ({args.n_t} sims)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
