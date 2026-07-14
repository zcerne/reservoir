"""3D ladder comparison figures: |Ey|(y,z) sensor plane, MEEP vs GPU vs |diff|."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/cernez/resevoir/data/ladder"
OUT = "/home/cernez/resevoir/ladder/_plots"
CONFIGS = [("1", "config_1_air_3d", "air + PML"),
           ("2", "config_2_LC_3d", "LC reservoir"),
           ("3", "config_3_LC_dye_3d", "LC + dye (gain)"),
           ("4", "config_4_mirrors_air_3d", "DBR mirrors + air"),
           ("5", "config_5_mirrors_LC_3d", "mirrors + LC"),
           ("6", "config_6_mirrors_LC_dye_3d", "mirrors + LC + dye")]

fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(13, 3.1 * len(CONFIGS)))
for row, (num, d, title) in enumerate(CONFIGS):
    A = np.load(f"{BASE}/{d}/simulation/monitor_2_meep.npz")["Ey"]
    B = np.load(f"{BASE}/{d}/simulation/monitor_2_gpumeep.npz")["Ey"]
    a = A[A.shape[0] // 2]; b = B[B.shape[0] // 2]   # center frequency
    ny = min(a.shape[0], b.shape[0]); nz = min(a.shape[1], b.shape[1])
    ac = a[(a.shape[0]-ny)//2:, :][:ny, (a.shape[1]-nz)//2:][:, :nz]
    bc = b[(b.shape[0]-ny)//2:, :][:ny, (b.shape[1]-nz)//2:][:, :nz]
    inner = np.vdot(ac, bc)
    corr = abs(inner) / (np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-300)
    dr = bc * np.exp(-1j * np.angle(inner))
    rel = np.linalg.norm(dr - ac) / (np.linalg.norm(ac) + 1e-300)
    vmax = np.abs(ac).max()
    ext = (-1.75, 1.75, -1.5, 1.5)   # z, y extents (cell_z=3.5, int_y=3)
    for col, (data, lab) in enumerate([(np.abs(ac), "MEEP |Ey|"),
                                       (np.abs(bc), "GPUmeep |Ey|"),
                                       (np.abs(bc - ac), "|GPU − MEEP|")]):
        ax = axes[row, col]
        im = ax.imshow(data, origin="lower", aspect="auto", extent=ext,
                       cmap="inferno" if col < 2 else "viridis",
                       vmax=vmax if col < 2 else None)
        plt.colorbar(im, ax=ax, shrink=0.85)
        if col == 0:
            ax.set_ylabel(f"cfg {num}: {title}\ny [µm]", fontsize=9)
        if row == 0:
            ax.set_title(lab)
        if row == len(CONFIGS) - 1:
            ax.set_xlabel("z [µm]")
    axes[row, 2].set_title(f"rel-L2={rel:.1e}  |corr|={corr:.8f}", fontsize=9)
fig.suptitle("3D ladder: sensor plane Ey (yz) — MEEP vs GPUmeep", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/ladder_3d_all.png", dpi=100)
print("saved", f"{OUT}/ladder_3d_all.png")
