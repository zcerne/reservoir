"""Plot MEEP vs GPUmeep sensor comparison for all ladder configs -> one PNG each."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/cernez/resevoir/data/ladder"
OUT = "/home/cernez/resevoir/ladder/_plots"
os.makedirs(OUT, exist_ok=True)

CONFIGS = [
    ("1", "config_1_air", "air + PML"),
    ("1.2", "config_1.2_air_periodic", "air, periodic, no PML"),
    ("2", "config_2_LC", "LC reservoir"),
    ("3", "config_3_LC_dye", "LC + dye (gain)"),
    ("4", "config_4_mirrors_air", "DBR mirrors + air"),
    ("5", "config_5_mirrors_LC", "mirrors + LC"),
    ("6", "config_6_mirrors_LC_dye", "mirrors + LC + dye"),
]

for num, d, title in CONFIGS:
    sim = os.path.join(BASE, d, "simulation")
    try:
        a = np.load(os.path.join(sim, "monitor_2_meep.npz"))["Ey"][0]
        b = np.load(os.path.join(sim, "monitor_2_gpumeep.npz"))["Ey"][0]
    except FileNotFoundError as e:
        print(f"cfg{num}: missing {e.filename}")
        continue
    n = min(len(a), len(b))
    a = a[(len(a) - n) // 2:][:n]
    b = b[(len(b) - n) // 2:][:n]
    inner = np.vdot(a, b)
    corr = abs(inner) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)
    dr = b * np.exp(-1j * np.angle(inner))
    rel = np.linalg.norm(dr - a) / (np.linalg.norm(a) + 1e-300)
    y = np.linspace(-3, 3, n)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax[0, 0].plot(y, np.abs(a), "k-", lw=2.2, label="MEEP")
    ax[0, 0].plot(y, np.abs(b), "r--", lw=1.3, label="GPUmeep")
    ax[0, 0].set_ylabel("|Ey|"); ax[0, 0].legend(); ax[0, 0].set_title("magnitude")
    ax[0, 1].plot(y, np.angle(a), "k-", lw=2.2)
    ax[0, 1].plot(y, np.angle(b), "r--", lw=1.3)
    ax[0, 1].set_ylabel("arg Ey [rad]"); ax[0, 1].set_title("phase")
    ax[1, 0].plot(y, np.real(a), "k-", lw=2.2, label="Re MEEP")
    ax[1, 0].plot(y, np.real(b), "r--", lw=1.3, label="Re GPU")
    ax[1, 0].plot(y, np.imag(a), "b-", lw=2.2, alpha=0.6, label="Im MEEP")
    ax[1, 0].plot(y, np.imag(b), "c--", lw=1.3, label="Im GPU")
    ax[1, 0].set_xlabel("y [um]"); ax[1, 0].set_ylabel("Ey"); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("real / imag")
    err = np.abs(b - a)
    ax[1, 1].semilogy(y, err + 1e-300, "m-", lw=1.2)
    ax[1, 1].set_xlabel("y [um]"); ax[1, 1].set_ylabel("|GPU − MEEP|")
    ax[1, 1].set_title("abs error (log)")
    fig.suptitle(f"Config {num}: {title}   —   |corr|={corr:.8f}, rel-L2(derot)={rel:.2e}",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, f"ladder_cfg{num.replace('.', '_')}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")
