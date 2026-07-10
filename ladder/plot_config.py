"""Plot MEEP vs GPUmeep sensor comparison for a ladder config (runs locally, numpy+mpl)."""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LADDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "ladder")
NAMES = {1: "air", 2: "LC", 3: "LC_dye", 4: "mirrors_air", 5: "mirrors_LC", 6: "mirrors_LC_dye"}


def load(p):
    Ey = np.asarray(np.load(p)["Ey"])
    return Ey.reshape(-1) if Ey.ndim == 1 else Ey[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, required=True)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    d = args.dir or os.path.join(LADDER, f"config_{args.config}_{NAMES[args.config]}")
    sim = os.path.join(d, "simulation")
    m = load(os.path.join(sim, "monitor_2_meep.npz"))
    g = load(os.path.join(sim, "monitor_2_gpumeep.npz"))
    # align lengths (MEEP sometimes has ±1-2 boundary samples)
    n = min(len(m), len(g))
    # center-align
    m = m[(len(m) - n) // 2: (len(m) - n) // 2 + n]
    g = g[(len(g) - n) // 2: (len(g) - n) // 2 + n]
    y = np.linspace(-3, 3, n)
    am, ag = np.abs(m), np.abs(g)
    corr = float(np.corrcoef(am, ag)[0, 1])
    ccorr = float(np.abs(np.vdot(g, m)) / (np.linalg.norm(g) * np.linalg.norm(m)))
    rel = float(np.linalg.norm(ag - am) / np.linalg.norm(am))
    ratio = float(ag.max() / am.max())

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(y, am, "b-", label="MEEP", lw=2)
    ax[0].plot(y, ag, "r--", label="gpumeep", lw=1.5)
    ax[0].set_title(f"|Ey|(y)  — abs value\nshape-corr={corr:.4f}  max-ratio={ratio:.3f}")
    ax[0].set_xlabel("y (µm)"); ax[0].set_ylabel("|Ey|"); ax[0].legend()
    ax[1].plot(y, m.real, "b-", label="MEEP Re", lw=2)
    ax[1].plot(y, g.real, "r--", label="gpumeep Re", lw=1.5)
    ax[1].set_title(f"Re(Ey)\ncomplex-corr={ccorr:.4f}  rel-L2-err={rel:.3f}")
    ax[1].set_xlabel("y (µm)"); ax[1].legend()
    ax[2].plot(y, m.imag, "b-", label="MEEP Im", lw=2)
    ax[2].plot(y, g.imag, "r--", label="gpumeep Im", lw=1.5)
    ax[2].set_title("Im(Ey)"); ax[2].set_xlabel("y (µm)"); ax[2].legend()
    cfg = os.path.basename(d)
    fig.suptitle(f"MEEP vs GPUmeep — {cfg}   "
                 f"(MEEP max {am.max():.3g}, gpumeep max {ag.max():.3g})", fontsize=13)
    plt.tight_layout()
    fig_dir = os.environ.get("LADDER_FIG_DIR",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"))
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"ladder_{cfg}.png")
    plt.savefig(out, dpi=120)
    print(f"config {args.config} ({cfg}): shape-corr={corr:.4f} complex-corr={ccorr:.4f} "
          f"rel-L2={rel:.3f} max-ratio={ratio:.3f}")
    print(f"  MEEP |Ey| max={am.max():.4g} mean={am.mean():.4g}")
    print(f"  gpumeep |Ey| max={ag.max():.4g} mean={ag.mean():.4g}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
