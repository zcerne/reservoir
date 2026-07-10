"""Characterize config 1 (air) gpu-vs-MEEP: is the amplitude ratio a constant
across y (→ source-scale / DFT-norm scalar) or y-dependent (→ PML/dispersion)?"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import _load_sensor  # noqa: E402

sim = "/home/cernez/resevoir/data/ladder/config_1_air/simulation"
a = _load_sensor(os.path.join(sim, "monitor_2_meep.npz")).astype(complex)
b = _load_sensor(os.path.join(sim, "monitor_2_gpumeep.npz")).astype(complex)
n = min(len(a), len(b))
a = a[(len(a)-n)//2:(len(a)-n)//2+n]; b = b[(len(b)-n)//2:(len(b)-n)//2+n]
r = np.abs(b) / (np.abs(a) + 1e-30)
ph = np.degrees(np.angle(b / a))
# central region (avoid edge/PML rolloff)
c0, c1 = n//4, 3*n//4
print(f"config 1: n={n}")
print(f"|gpu|/|meep| ratio: mean={r.mean():.5f} std={r.std():.5f}  "
      f"central mean={r[c0:c1].mean():.5f} std={r[c0:c1].std():.5f}")
print(f"  ratio min={r.min():.5f} max={r.max():.5f}  (edge vs center spread)")
print(f"phase(gpu/meep) deg: central mean={ph[c0:c1].mean():.4f} std={ph[c0:c1].std():.4f}")
print(f"|meep|max={np.abs(a).max():.6g}  |gpu|max={np.abs(b).max():.6g}  "
      f"ratio_of_max={np.abs(b).max()/np.abs(a).max():.5f}")
# is it constant? print ratio at 5 y-positions
for k in np.linspace(c0, c1-1, 5).astype(int):
    print(f"    y-idx {k}: |meep|={np.abs(a[k]):.5g} |gpu|={np.abs(b[k]):.5g} ratio={r[k]:.5f}")
print("DONE_CFG1")
