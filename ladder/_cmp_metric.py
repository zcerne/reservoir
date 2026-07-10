"""Phase-aware MEEP-vs-gpumeep sensor comparison for one ladder config.
Usage: python _cmp_metric.py <config_dir>
Compares monitor_2_meep.npz (reference) with monitor_2_gpumeep.npz.
"""
import os, sys
import numpy as np


def load(p):
    d = np.load(p)
    Ey = np.asarray(d["Ey"])
    return Ey.reshape(-1) if Ey.ndim == 1 else Ey[0]


def main():
    sim = os.path.join(sys.argv[1], "simulation")
    a = load(os.path.join(sim, "monitor_2_meep.npz"))     # reference (MEEP)
    b = load(os.path.join(sim, "monitor_2_gpumeep.npz"))  # gpumeep
    a = a.astype(complex); b = b.astype(complex)
    # MEEP get_array and the gpumeep monitor can differ by 1-2 samples at the
    # y-endpoints; center-crop both to the common length before comparing.
    if len(a) != len(b):
        n = min(len(a), len(b))
        a = a[(len(a) - n) // 2:(len(a) - n) // 2 + n]
        b = b[(len(b) - n) // 2:(len(b) - n) // 2 + n]
    inner = np.vdot(a, b)                       # <a,b> = Σ conj(a)·b
    corr = np.abs(inner) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)
    phase = np.degrees(np.angle(inner))
    max_ratio = np.abs(b).max() / (np.abs(a).max() + 1e-300)
    relL2 = np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-300)
    derot = b * np.exp(-1j * np.angle(inner))   # remove global phase
    relL2_derot = np.linalg.norm(derot - a) / (np.linalg.norm(a) + 1e-300)
    print(f"cfg4 |corr|={corr:.4f} phase={phase:.2f}deg max-ratio={max_ratio:.4f} "
          f"rel-L2={relL2:.3f} rel-L2(derot)={relL2_derot:.3f}")


if __name__ == "__main__":
    main()
