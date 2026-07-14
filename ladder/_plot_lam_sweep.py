"""Per-config wavelength-sweep comparison plot: |Ey|(y) at each signal lambda,
MEEP (black) vs GPUmeep (red dashed), with |corr| + rel-L2 annotations."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/cernez/resevoir/ladder/_plots/data"
OUT = "/home/cernez/resevoir/ladder/_plots"
LAMS = ["0.45", "0.50", "0.55", "0.60"]
TITLES = {1: "air + PML", 2: "LC reservoir", 3: "LC + dye (gain)",
          4: "DBR mirrors + air", 5: "mirrors + LC", 6: "mirrors + LC + dye"}


def load(p):
    ey = np.load(p)["Ey"]
    return (ey.reshape(-1) if ey.ndim == 1 else ey[0]).astype(complex)


for c in (1, 2, 3, 4, 5, 6):
    fig, axes = plt.subplots(2, len(LAMS), figsize=(4 * len(LAMS), 6.5),
                             sharex=True)
    for k, lam in enumerate(LAMS):
        try:
            a = load(f"{DATA}/cfg{c}_lam{lam}_meep.npz")
            b = load(f"{DATA}/cfg{c}_lam{lam}_gpumeep.npz")
        except FileNotFoundError:
            axes[0, k].set_title(f"λ={lam}: missing")
            continue
        n = min(len(a), len(b))
        a = a[(len(a) - n) // 2:][:n]; b = b[(len(b) - n) // 2:][:n]
        inner = np.vdot(a, b)
        corr = abs(inner) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)
        dr = b * np.exp(-1j * np.angle(inner))
        rel = np.linalg.norm(dr - a) / (np.linalg.norm(a) + 1e-300)
        y = np.linspace(-3, 3, n)
        ax = axes[0, k]
        ax.plot(y, np.abs(a), "k-", lw=2.2, label="MEEP")
        ax.plot(y, np.abs(b), "r--", lw=1.3, label="GPUmeep")
        ax.set_title(f"λ = {lam} µm\n|corr|={corr:.6f}  relL2={rel:.1e}", fontsize=10)
        if k == 0:
            ax.set_ylabel("|Ey|"); ax.legend(fontsize=8)
        ax2 = axes[1, k]
        ax2.semilogy(y, np.abs(b - a) + 1e-300, "m-", lw=1.0)
        ax2.set_xlabel("y [µm]")
        if k == 0:
            ax2.set_ylabel("|GPU − MEEP|")
    fig.suptitle(f"Config {c}: {TITLES[c]} — signal wavelength sweep", fontsize=14)
    fig.tight_layout()
    out = f"{OUT}/ladder_cfg{c}_lamsweep.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)
