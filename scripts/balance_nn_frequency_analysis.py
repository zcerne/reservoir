"""Frequency analysis (Method D harmonics) of the trained Balance-Scale NN.

Loads the trained balance-scale MLP (weights in data/NN_data, see
scripts/train_balance_scale_nn.py) and drives its 4 inputs with two integer
tones through a phase-sweep parameter t, SAME tone on each side of the scale:

    tone f1 -> inputs 0,1  (left weight, left distance)
    tone f2 -> inputs 2,3  (right weight, right distance)
    x(t) = A·cos(f1·t)·[1,1,0,0] + A·cos(f2·t)·[0,0,1,1]

The hidden layer is the "reservoir state". DFT over t and classify the bins
with n4_harmonics_distortion.harmonic_specter: the identity net may only keep
the fundamentals; the sigmoid net puts power at harmonics (2f1, 3f1, ...) and
intermod products (f1±f2, ...). The balance rule itself is degree-2
(LW·LD − RW·RD), whose signature under this drive sits at DC, 2f1 and 2f2.

  python scripts/balance_nn_frequency_analysis.py
  python scripts/balance_nn_frequency_analysis.py --model data/NN_data/6_linearNN_balance/balance_scale_model.npz \
      --out data/reservoir_clasifications/19_linear_NN_Balance
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "characterization"))
sys.path.insert(0, os.path.join(ROOT, "plotting"))
import n4_harmonics_distortion as n4
from plot_function_n4_harmonics_distortion import plot_n4_harmonics_distortion


def load_forward(model_path):
    """forward(x) -> hidden activations of the trained balance-scale MLP."""
    m = np.load(model_path, allow_pickle=True)
    W1, b1 = m["W1"], m["b1"]
    activation = str(m["activation"])
    act = (lambda z: z) if activation == "identity" else \
          (lambda z: 1.0 / (1.0 + np.exp(-z)))

    def forward(x):
        x = np.real(np.asarray(x)).ravel()
        return act(x @ W1.T + b1)
    return forward, W1.shape[1], activation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        ROOT, "data", "NN_data", "6_nonlinearNN_balance", "balance_scale_model.npz"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "data", "reservoir_clasifications", "19_nonlinear_NN_Balance"))
    ap.add_argument("--tones", default="3,5")
    ap.add_argument("--amp", type=float, default=1.0,
                    help="cosine amplitude in STANDARDIZED input units (~sigma)")
    ap.add_argument("--n_t", type=int, default=64,
                    help="phase samples; needs n_t > 2*max_order*max_tone")
    a = ap.parse_args()

    forward, n_in, activation = load_forward(a.model)
    tones = [int(t) for t in a.tones.split(",") if t.strip()]
    if n_in % len(tones):
        raise SystemExit(f"{n_in} inputs not divisible into {len(tones)} tone groups")
    print(f"[freq] model={a.model}  activation={activation}  n_in={n_in}")

    # same tone on each contiguous input pair: [0,1]->f1, [2,3]->f2
    U = np.zeros((len(tones), n_in))
    b = np.linspace(0, n_in, len(tones) + 1).astype(int)
    for k in range(len(tones)):
        U[k, b[k]:b[k + 1]] = a.amp
        print(f"[freq] tone {tones[k]} -> inputs {list(range(b[k], b[k + 1]))}")

    t = 2.0 * np.pi * np.arange(a.n_t) / a.n_t
    inputs = np.stack([sum(np.cos(tones[k] * tj) * U[k] for k in range(len(tones)))
                       for tj in t])
    outputs = np.stack([forward(x) for x in inputs]).astype(np.float64)

    ds = os.path.join(a.out, "datasets"); os.makedirs(ds, exist_ok=True)
    ds_path = os.path.join(ds, "harmonics.npz")
    np.savez(ds_path, outputs=outputs, inputs=inputs, t=t,
             tones=np.asarray(tones), amps=np.full(len(tones), a.amp),
             components=np.asarray("hidden"))
    print(f"[freq] dataset ({a.n_t} phase samples) -> {ds_path}")

    d = dict(outputs=outputs, tones=np.asarray(tones))
    res_state = n4.harmonic_specter(d)
    res_int = n4.harmonic_specter(dict(outputs=np.abs(outputs) ** 2,
                                       tones=np.asarray(tones)))
    stats = os.path.join(a.out, "stats_data"); os.makedirs(stats, exist_ok=True)
    np.savez(os.path.join(stats, "n4.npz"), n4_field=res_state, n4_intensity=res_int)

    figs = os.path.join(a.out, "figures")
    p = plot_n4_harmonics_distortion(res_state, figs, suffix="_state", data=d)
    print(f"[freq] figure -> {p}")

    print(f"\n=== hidden state ({activation}) ===")
    print(n4.report(res_state))
    print("\n=== |hidden|^2 readout ===")
    print(n4.report(res_int))


if __name__ == "__main__":
    main()
