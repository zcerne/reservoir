#!/usr/bin/env python
"""signal_modulation: all four Nd:YAG level populations over the whole run.

Level map (4-level scheme, JSON order):
  1 = ground (4I9/2)          - refilled by the drain, emptied by the pump
  2 = lower laser (4I11/2)    - filled by stimulated emission, drained fast
  3 = upper laser (4F3/2)     - the stored energy; starts full (pre-inverted)
  4 = pump band (4F5/2)       - transient, fed by 808 and dumped into 3
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1]
OUT = sys.argv[2]
AMPS = [int(a) for a in sys.argv[3].split(",")]
LAB = {0: ("1  ground (4I9/2)", "tab:blue"), 1: ("2  lower laser (4I11/2)", "tab:orange"),
       2: ("3  upper laser (4F3/2)", "tab:red"), 3: ("4  pump band (4F5/2)", "tab:green")}

fig, axes = plt.subplots(len(AMPS), 2, figsize=(14, 3.1 * len(AMPS)), squeeze=False)
for row, amp in enumerate(AMPS):
    d = np.load(os.path.join(RES, f"pop_monitor_a{amp}.npz"))
    t, N = d["t"], d["N"]
    tot = N[0].sum()

    ax = axes[row][0]                       # all four, linear
    for k in range(4):
        lab, c = LAB[k]
        ax.plot(t, N[:, k] / tot, lw=1.4, color=c, label=lab)
    ax.set_ylabel(f"drive {amp}\npopulation / total")
    ax.grid(alpha=0.25)
    if row == 0:
        ax.set_title("all four levels (fraction of total)")
        ax.legend(fontsize=7, loc="center right")

    ax = axes[row][1]                       # levels 1 and 4, log
    for k in (0, 3):
        lab, c = LAB[k]
        ax.semilogy(t, np.maximum(N[:, k] / tot, 1e-9), lw=1.4, color=c, label=lab)
    ax.grid(alpha=0.25)
    ax.set_ylabel("fraction (log)")
    if row == 0:
        ax.set_title("levels 1 and 4 alone (log) — the pump cycle")
        ax.legend(fontsize=7)
    f1, f4 = N[-1, 0] / tot, N[-1, 3] / tot
    ax.text(0.015, 0.12, f"end: L1 {f1:.3f}   L4 {f4:.2e}", transform=ax.transAxes,
            fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("time [MEEP units]")
fig.suptitle("signal_modulation/design01 — Nd:YAG level populations, whole run",
             fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
for amp in AMPS:
    d = np.load(os.path.join(RES, f"pop_monitor_a{amp}.npz"))
    N = d["N"]; tot = N[0].sum()
    print(f"  drive {amp:4d}: end fractions  L1 {N[-1,0]/tot:.4f}  L2 {N[-1,1]/tot:.2e}"
          f"  L3 {N[-1,2]/tot:.4f}  L4 {N[-1,3]/tot:.2e}")
