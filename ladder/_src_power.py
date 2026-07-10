"""Decisive, calibration-free measurement of the gpu-vs-MEEP structural ratio.

Runs config 1 (air, pure vacuum line source) with gpu src_scale = 1 EXACTLY
(no C, no res factor) and MEEP native, across resolutions. The ratio
  Q(res) = |Ey|_gpu / |Ey|_meep
is the raw 'field per unit injected current' between the two codes with no
calibration. Its resolution power (log-log slope) reveals the exact structural
difference: slope 0 -> pure constant (codes equivalent, only a scalar differs),
slope -1 -> gpu field is 1/res smaller (a missing dx / dt somewhere), etc.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_meep, run_gpumeep  # noqa: E402

RESES = [20, 40, 80]
rows = []
for res in RESES:
    os.environ["LADDER_RES"] = str(res)
    path = build_json(1)
    ey_m = np.asarray(run_meep(path)).reshape(-1)
    # gpu with src_scale = 1 exactly (C=1, res^0) — zero calibration
    os.environ["GPUMEEP_SRC_C"] = "1"
    os.environ["GPUMEEP_SRC_POW"] = "0"
    ey_g = np.asarray(run_gpumeep(path)).reshape(-1)
    Mm = np.abs(ey_m).max()
    Gg = np.abs(ey_g).max()
    Q = Gg / Mm
    rows.append((res, Mm, Gg, Q))
    print(f"RES={res}: |Ey|meep={Mm:.6g}  |Ey|gpu(scale1)={Gg:.6g}  Q=gpu/meep={Q:.6g}",
          flush=True)

print("\n--- structural ratio Q(res) = field-per-unit-current gpu/meep ---")
r = np.array([x[0] for x in rows], float)
q = np.array([x[3] for x in rows], float)
# log-log slope between consecutive resolutions
for i in range(1, len(r)):
    slope = np.log(q[i] / q[i - 1]) / np.log(r[i] / r[i - 1])
    print(f"  slope log(Q)/log(res) [{int(r[i-1])}->{int(r[i])}] = {slope:+.4f}")
# global fit Q = a * res^b
b, loga = np.polyfit(np.log(r), np.log(q), 1)
print(f"  global fit: Q = {np.exp(loga):.5g} * res^({b:+.4f})")
print(f"  => to make ratio res-flat, src_scale must scale as res^({-b:+.4f}), "
      f"const = {1.0/np.exp(loga):.6g}")
print("DONE_SRC_POWER")
