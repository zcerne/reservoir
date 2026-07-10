"""DECISIVE TEST: run config 4 with gpumeep but feed it MEEP's EXACT averaged ε
(sampled at gpu's own Yee-face locations). If the sensor then matches MEEP, the
residual is confirmed to be the mirror-boundary ε averaging (not the engine).

MEEP ε is isotropic for config 4, so a scalar ε(x,y) suffices; we sample MEEP's
snap epsilon (raw per-pixel) at each gpu Yee face via bilinear interpolation and
build gpu's material as diag(1/ε).
"""
import os, sys, importlib
import numpy as np
RESV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RESV); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_meep, _load_sensor  # noqa: E402
import shutil  # noqa: E402

path = build_json(4)
sim_dir = os.path.join(path, "simulation")

# 1. MEEP reference run + its exact averaged epsilon on the Yee grid
import meep as mp
from class_simulation import Simulation
smeep = Simulation(path); smeep._set_everything(); smeep.simulation.init_sim()
eps_meep = np.asarray(smeep.simulation.get_epsilon(snap=True))     # (Nx, Ny) scalar
md = smeep.simulation.get_array_metadata()
xm = np.asarray(md[0]); ym = np.asarray(md[1])
xm = xm[:eps_meep.shape[0]]; ym = ym[:eps_meep.shape[1]]
smeep.simulation.run(until=smeep.args["run_until"])
ey_meep = _load_sensor(os.path.join(sim_dir, "monitor_2.npz"))
shutil.copy(os.path.join(sim_dir, "monitor_2.npz"), os.path.join(sim_dir, "monitor_2_meep.npz"))
print(f"MEEP done: |Ey|max={np.abs(ey_meep).max():.4g}")

# 2. gpumeep run with MEEP's epsilon injected
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
import fdtd_2d as f2
from scipy.interpolate import RegularGridInterpolator
g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
g._set_data(); g._update_all_args()

interp = RegularGridInterpolator((xm, ym), eps_meep, bounds_error=False, fill_value=1.0)
def eps_at(x_off, y_off):
    i = np.arange(g.Nx); j = np.arange(g.Ny)
    X = (i + x_off) * g.dx - g.gx / 2
    Y = (j + y_off) * g.dx - g.gy / 2
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    return interp(np.stack([XX.ravel(), YY.ravel()], -1)).reshape(g.Nx, g.Ny)

eEx = eps_at(0.5, 0.0)   # ε at Ex face
eEy = eps_at(0.0, 0.5)   # ε at Ey face
end = eps_at(0.0, 0.0)   # ε at node
J = lambda a: jnp.asarray(a, jnp.float64)
z = J(np.zeros((g.Nx, g.Ny)))
g._build_material = lambda: None      # disable the normal build
g.material = f2.AnisoFull2D(
    ixx_Ex=J(1.0 / eEx), ixy_Ex=z, ixz_Ex=z,
    ixy_Ey=z, iyy_Ey=J(1.0 / eEy), iyz_Ey=z,
    ixz_nd=z, iyz_nd=z, izz_nd=J(1.0 / end))
g.eps_inv_zz = J(1.0 / end)
g._n_max = float(np.sqrt(max(eEx.max(), eEy.max(), end.max())))
g.run()
ey_gpu = _load_sensor(os.path.join(sim_dir, "monitor_2.npz"))
shutil.copy(os.path.join(sim_dir, "monitor_2.npz"), os.path.join(sim_dir, "monitor_2_gpumeep.npz"))

# 3. compare
a = ey_meep.astype(complex); b = ey_gpu.astype(complex)
n = min(len(a), len(b)); a = a[(len(a)-n)//2:(len(a)-n)//2+n]; b = b[(len(b)-n)//2:(len(b)-n)//2+n]
inner = np.vdot(a, b)
corr = abs(inner)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-300)
print(f"INJECTED-EPS cfg4  |corr|={corr:.4f}  phase={np.degrees(np.angle(inner)):.2f}deg  "
      f"max-ratio={np.abs(b).max()/np.abs(a).max():.4f}  "
      f"rel-L2(derot)={np.linalg.norm(b*np.exp(-1j*np.angle(inner))-a)/np.linalg.norm(a):.3f}")
print("DONE_INJECT")
