#!/usr/bin/env python
"""Echo energies E0 (transmitted pulse), E1, E2, ... from decay-run point snapshots.

E_k = ∫ env(t) dt over round-trip window k, env = |Ey|² smoothed over ~2 optical
periods. Window 0 is centered on the transmitted-pulse peak (located in the
first `--t-first` t.u.); window k is T_rt later. Used for the pump ladder
(gain vs pump) and the probe-amplitude ladder (saturation: E1/E0, E2/E1 vs amp).

    python analysis_T/echo_energies.py \
        data/lasing_testing/05_adding_mirror/decayamp_p150_R0.5_a* \
        data/lasing_testing/05_adding_mirror/decay_p150_R0.5 \
        --label-re 'a([0-9.]+)$|p150' --n-echoes 3
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np


def envelope(ey, dt, periods=2.0, lam=0.55):
    w = max(3, int(round(periods * lam / dt)))
    return np.convolve(ey ** 2, np.ones(w) / w, mode="same")


def echo_energies(folder, T_rt=95.0, t_first=120.0, n_echoes=3):
    z = np.load(os.path.join(folder, "simulation_meep", "point_snap.npz"))
    t, ey = z["t"], z["Ey"][:, 0]
    dt = float(np.median(np.diff(t)))
    env = envelope(ey, dt)
    t0 = t[np.argmax(env[t <= t_first])]        # transmitted-pulse peak
    E = []
    for k in range(n_echoes + 1):
        sel = (t >= t0 + k * T_rt - T_rt / 2) & (t < t0 + k * T_rt + T_rt / 2)
        E.append(float(np.sum(env[sel]) * dt))
    return t0, E


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--T-rt", type=float, default=95.0)
    ap.add_argument("--t-first", type=float, default=120.0)
    ap.add_argument("--n-echoes", type=int, default=3)
    a = ap.parse_args()

    print(f"{'design':<38} {'t0':>6} " +
          " ".join(f"{'E'+str(k):>10}" for k in range(a.n_echoes + 1)) +
          f" {'E1/E0':>7} {'E2/E1':>7}")
    for f in a.folders:
        name = os.path.basename(f.rstrip("/"))
        try:
            t0, E = echo_energies(f, a.T_rt, a.t_first, a.n_echoes)
        except FileNotFoundError:
            print(f"{name:<38} (no point_snap)")
            continue
        r10 = E[1] / E[0] if E[0] > 0 else float("nan")
        r21 = E[2] / E[1] if len(E) > 2 and E[1] > 0 else float("nan")
        print(f"{name:<38} {t0:>6.1f} " +
              " ".join(f"{e:>10.3g}" for e in E) +
              f" {r10:>7.3f} {r21:>7.3f}")


if __name__ == "__main__":
    main()
