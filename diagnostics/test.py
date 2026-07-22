import json
import numpy as np
import os

data_path = "data/lasing_testing/01_basic_test/"

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
