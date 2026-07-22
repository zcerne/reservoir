import json
import numpy as np
import os

data_path = "/home/ziga/Orion/resevoir/data/lasing_testing/01_basic_test/"

concentration_data = np.load(os.path.join(data_path, "simulation_gpumeep/conc_monitor.npz"))

print(concentration_data)

times = concentration_data["times"]
levels = concentration_data["levels"]
N = concentration_data["N"]  # (n_times, 4, Nx, Ny) — gain_populations() returns the FULL
                              # sim grid (not cropped to the reservoir), so the physical
                              # extent below is the whole cell, not just the reservoir.

with open(os.path.join(data_path, "simulation_data.json")) as f:
    _cfg = json.load(f)
_dx = 1.0 / _cfg["resolution"]
_nx, _ny = N.shape[2], N.shape[3]
cell_x, cell_y = _nx * _dx, _ny * _dx
extent = [-cell_x / 2, cell_x / 2, -cell_y / 2, cell_y / 2]

import matplotlib.pyplot as plt

snap_id = 110

fig, axs = plt.subplots(2, 2, figsize=(9, 9))
axs = axs.flatten()
for i in range(4):
    # .T + origin="lower": x (propagation direction, array axis 0) on the
    # horizontal axis, y (transverse, array axis 1) on the vertical — same
    # convention simplesim's own sensor plots use.
    im = axs[i].imshow(N[snap_id, i].T, origin="lower", extent=extent, aspect="equal")
    fig.colorbar(im, ax=axs[i])
    axs[i].set_title(f"{levels[i]}  (t={times[snap_id]:.1f})")
    axs[i].set_xlabel("x [µm]")
    axs[i].set_ylabel("y [µm]")
fig.tight_layout()
plt.savefig(os.path.join(data_path, "figures/conc_snapshot.png"), dpi=140)
plt.show()

sums = np.sum(N, axis=(2, 3)).T

plt.figure()
for i in range(4):
    plt.plot(times, sums[i], label=str(levels[i]))
plt.xlabel("t [MEEP units]")
plt.ylabel("total population (summed over grid)")
plt.legend()
plt.yscale("log")
plt.savefig(os.path.join(data_path, "figures/conc_totals.png"), dpi=140)
plt.show()

snapshot_1_data = np.load(os.path.join(data_path, "simulation_gpumeep/snapshot_1.npz"))
print(snapshot_1_data)

snap_t = snapshot_1_data["t"]
snap_Ex = snapshot_1_data["Ex"]
snap_Ey = snapshot_1_data["Ey"]
snap_Ez = snapshot_1_data["Ez"]  # (n_snaps, nx, ny) — 2Dsnap, cropped to the JSON
                                 # position.size box already (unlike conc_monitor,
                                 # this sensor type honors size at record time).
snap_I = np.abs(snap_Ex) ** 2 + np.abs(snap_Ey) ** 2 + np.abs(snap_Ez) ** 2

# extent relative to the box's OWN center (not absolute lab-frame x) — this
# script only has the npz, not the full layout, so ticks are "distance from
# the reservoir center" rather than absolute simulation coordinates.
_snx, _sny = snap_Ex.shape[1], snap_Ex.shape[2]
snap_extent = [-_snx * _dx / 2, _snx * _dx / 2, -_sny * _dx / 2, _sny * _dx / 2]

n_snaps = len(snap_t)
ncols = 4
nrows = -(-n_snaps // ncols)  # ceil
fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
axs = axs.flatten()
vmax = snap_I.max()
for i in range(n_snaps):
    im = axs[i].imshow(snap_I[i].T, origin="lower", extent=snap_extent,
                       aspect="equal", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=axs[i])
    axs[i].set_title(f"|E|²  (t={snap_t[i]:.1f})")
    axs[i].set_xlabel("x rel. to reservoir center [µm]")
    axs[i].set_ylabel("y [µm]")
for i in range(n_snaps, len(axs)):
    axs[i].axis("off")
fig.tight_layout()
plt.savefig(os.path.join(data_path, "figures/snapshot_intensity.png"), dpi=140)
plt.show()

monitor_1_data = np.load(os.path.join(data_path, "simulation_gpumeep/monitor_1.npz"))
monitor_2_data = np.load(os.path.join(data_path, "simulation_gpumeep/monitor_2.npz"))
print(monitor_1_data)
print(monitor_2_data)

# Spectral power (Σ|E|² integrated along the monitor line) vs wavelength —
# input (guide_1, before the reservoir) vs output (guide_2, after it).
plt.figure()
for label, mon in (("monitor_1 (input, guide_1)", monitor_1_data),
                   ("monitor_2 (output, guide_2)", monitor_2_data)):
    freqs = mon["freqs"]
    lam = 1.0 / freqs
    power = np.sum(np.abs(mon["Ex"]) ** 2 + np.abs(mon["Ey"]) ** 2
                   + np.abs(mon["Ez"]) ** 2, axis=1)
    order = np.argsort(lam)
    plt.plot(lam[order], power[order], label=label)
plt.xlabel("λ [µm]")
plt.ylabel("Σ|E|² along monitor line")
plt.legend()
plt.yscale("log")
plt.savefig(os.path.join(data_path, "figures/monitor_spectra.png"), dpi=140)
plt.show()

# Spatial |E(y)|² profile at the source wavelength (500nm) — input vs output.
plt.figure()
for label, mon in (("monitor_1 (input, guide_1)", monitor_1_data),
                   ("monitor_2 (output, guide_2)", monitor_2_data)):
    freqs = mon["freqs"]
    k = int(np.argmin(np.abs(freqs - 1.0 / 0.5)))
    I_y = (np.abs(mon["Ex"][k]) ** 2 + np.abs(mon["Ey"][k]) ** 2
           + np.abs(mon["Ez"][k]) ** 2)
    y = np.linspace(-len(I_y) / 2 * _dx, len(I_y) / 2 * _dx, len(I_y))
    plt.plot(y, I_y, label=label)
plt.xlabel("y [µm]")
plt.ylabel("|E|² at λ=500nm")
plt.legend()
plt.savefig(os.path.join(data_path, "figures/monitor_profiles.png"), dpi=140)
plt.show()
