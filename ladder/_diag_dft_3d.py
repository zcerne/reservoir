"""Pin down the 3D cfg2 DFT discrepancy: manual DFT of the MEEP time trace at
the monitor point vs both saved sensors."""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
import ladder  # noqa: E402

path = ladder.build_json(int(os.environ.get("DIAG_CFG", "2")))
sim_dir = os.path.join(path, "simulation")

a = np.load(os.path.join(sim_dir, "monitor_2_meep.npz"))
b = np.load(os.path.join(sim_dir, "monitor_2_gpumeep.npz"))
Am = a["Ey"][0]; Bg = b["Ey"][0]
print("meep sensor", Am.shape, "max", np.abs(Am).max(),
      "argmax", np.unravel_index(np.abs(Am).argmax(), Am.shape))
print("gpu  sensor", Bg.shape, "max", np.abs(Bg).max(),
      "argmax", np.unravel_index(np.abs(Bg).argmax(), Bg.shape))
print("meep freqs", a["freqs"], "gpu freqs", b["freqs"])
# center entries
cm = Am[Am.shape[0] // 2, Am.shape[1] // 2]
cg = Bg[Bg.shape[0] // 2, Bg.shape[1] // 2]
print(f"center: meep={cm:.6g}  gpu={cg:.6g}  ratio={cm/cg:.6g}")

# --- manual DFT from a fresh MEEP run's time trace at the monitor point ---
import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

simm = Simulation(path)
simm._set_everything()
s = simm.simulation
s.init_sim()
dt = s.fields.dt
# monitor is at guide_2 center; read its x from the sensor object
sens = simm.sensors[0]
print("sensor center:", sens.center, "size:", sens.size)
xm = float(sens.center.x)
trace = []


def _rec(sim_):
    trace.append(np.real(complex(sim_.get_field_point(mp.Ey, mp.Vector3(xm, 0.025, 0.0)))))


run_until = float(simm.args.get("run_until", 60))
s.run(_rec, until=run_until)
s.change_sources([])
s.run(_rec, until=50)
trace = np.array(trace)
n = len(trace)
print("trace steps:", n, "max", np.abs(trace).max())

# get_dft_array from THIS run's own sensor (same process, same fields)
h = sens._monitor_handle
Ey_own = np.array(s.get_dft_array(h, mp.Ey, 0))
print("own get_dft_array:", Ey_own.shape, "max", np.abs(Ey_own).max())
print("own center value:", Ey_own[Ey_own.shape[0]//2, Ey_own.shape[1]//2])
# constancy of the ratio vs the saved ladder npz
print("saved-vs-own max|diff|:", np.abs(Ey_own - Am).max() if Ey_own.shape == Am.shape else "shape mismatch")
f0 = float(a["freqs"][0]); omega = 2 * np.pi * f0
mth = np.arange(1, n + 1)
for q in (1, 9):
    sel = (mth % q) == 0
    val = np.sum(trace[sel] * np.exp(1j * omega * mth[sel] * dt)) * q * dt / np.sqrt(2 * np.pi)
    print(f"manual DFT (decim {q}): {val:.6g}  |.|={abs(val):.6g}")
