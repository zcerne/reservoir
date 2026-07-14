"""Spectrum comparison plots: |Ey(f)| at the sensor, MEEP vs GPUmeep, 2D+3D."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/cernez/resevoir/data/ladder"
OUT = "/home/cernez/resevoir/ladder/_plots"
CASES = [("2D cfg1 (air)", "config_1_air"), ("2D cfg4 (mirrors)", "config_4_mirrors_air"),
         ("3D cfg1 (air)", "config_1_air_3d"), ("3D cfg4 (mirrors)", "config_4_mirrors_air_3d")]

fig, axes = plt.subplots(2, 4, figsize=(19, 7))
for col, (title, d) in enumerate(CASES):
    A = np.load(f"{BASE}/{d}/simulation/monitor_2_meep.npz")
    B = np.load(f"{BASE}/{d}/simulation/monitor_2_gpumeep.npz")
    fr = A["freqs"]; lam = 1.0 / fr
    a = A["Ey"]; b = B["Ey"]
    # spectral amplitude: L2 over the sensor plane/line per frequency (crop to common)
    if a.ndim == 2:
        n = min(a.shape[1], b.shape[1])
        ac = a[:, (a.shape[1]-n)//2:][:, :n]; bc = b[:, (b.shape[1]-n)//2:][:, :n]
        axsum = 1
    else:
        ny = min(a.shape[1], b.shape[1]); nz = min(a.shape[2], b.shape[2])
        ac = a[:, (a.shape[1]-ny)//2:, :][:, :ny, (a.shape[2]-nz)//2:][:, :, :nz]
        bc = b[:, (b.shape[1]-ny)//2:, :][:, :ny, (b.shape[2]-nz)//2:][:, :, :nz]
        axsum = (1, 2)
    Sm = np.sqrt((np.abs(ac) ** 2).sum(axis=axsum))
    Sg = np.sqrt((np.abs(bc) ** 2).sum(axis=axsum))
    rel = np.array([np.linalg.norm((bc[i] * np.exp(-1j*np.angle(np.vdot(ac[i], bc[i])))) - ac[i])
                    / (np.linalg.norm(ac[i]) + 1e-300) for i in range(len(fr))])
    ax = axes[0, col]
    ax.plot(lam * 1000, Sm, "ko-", lw=2, ms=5, label="MEEP")
    ax.plot(lam * 1000, Sg, "r^--", lw=1.2, ms=5, label="GPUmeep")
    ax.set_title(title); ax.set_ylabel("|Ey(f)| (sensor L2)")
    ax.legend(fontsize=8)
    if "mirror" in d:
        ax.axvspan(500, 620, alpha=0.12, color="b")  # DBR stopband vicinity
    ax2 = axes[1, col]
    ax2.semilogy(lam * 1000, rel, "m.-")
    ax2.set_xlabel("wavelength [nm]"); ax2.set_ylabel("rel-L2(derot) per freq")
    ax2.set_ylim(1e-15, 1e-5)
fig.suptitle("Gaussian-pulse spectra at the sensor: MEEP vs GPUmeep "
             "(15 DFT frequencies, λ 420–650 nm)", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/ladder_spectra.png", dpi=110)
print("saved", f"{OUT}/ladder_spectra.png")
