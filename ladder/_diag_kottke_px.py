"""Trace the GPU Kottke acceptance-loop decision at the cfg5 disagreeing pixels."""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
import ladder  # noqa: E402

path = ladder.build_json(5)

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src"))
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")

g = csg.SimulationGPU(folder_path=path)
g.force_fullvector = True
g._set_data(); g._update_all_args(); g._setup_lc_interp()

rects = g._iso_rects()
print(f"Nx={g.Nx} Ny={g.Ny} dx={g.dx}")
for k, r in enumerate(rects):
    print(f"  rect {k}: cx={r[0]:.10f} sx={r[1]:.10f} cy={r[2]} sy={r[3]} n2={r[4]}")

half = 0.5 * g.dx
icx = g.Nx - g.Nx % 2; icy = g.Ny - g.Ny % 2

# the 4 disagreeing pixels: (comp, i, j)
PIX = [("ixx", 103, 60), ("ixx", 103, 300), ("iyy", 304, 60), ("izz", 304, 60)]
OFF = {"ixx": (1, 0), "iyy": (0, 1), "izz": (0, 0)}

for comp, i, j in PIX:
    sx_off, sy_off = OFF[comp]
    X = (2 * i + sx_off - icx) * half
    Y = (2 * j + sy_off - icy) * half
    print(f"\n{comp} (i={i}, j={j}) X={X:.10f} Y={Y:.10f}")
    offs = ((0.0, 0.0), (-half, -half), (half, half), (-half, half), (half, -half))
    sids = []
    for (ox, oy) in offs:
        qx, qy = X + ox, Y + oy
        owner = -1
        for k, (cx_, sx_, cy_, sy_, _n2) in enumerate(rects):
            if abs(qx - cx_) <= 0.5 * sx_ and abs(qy - cy_) <= 0.5 * sy_:
                owner = k
        sids.append(owner)
        print(f"   sample ({qx:+.6f},{qy:+.6f}) -> owner {owner} "
              f"(n2={'bg' if owner < 0 else rects[owner][4]})")
    # replicate acceptance loop
    n2s = [1.0 if r[4] is None else r[4] for r in rects]
    id1 = id2 = None; mat1 = mat2 = None; fail = False
    for oid in sids:
        if oid == id1 or oid == id2:
            continue
        m = 1.0 if oid < 0 else float(n2s[oid])
        if id1 is None:
            id1, mat1 = oid, m
        elif id2 is None or ((oid >= id1 and oid >= id2) and (id1 == id2 or mat1 == mat2)):
            id2, mat2 = oid, m
        elif not (id1 < id2 and (id1 == oid or mat1 == m)) and \
             not (id2 < id1 and (id2 == oid or mat2 == m)):
            fail = True
            break
    print(f"   accept-loop: id1={id1} id2={id2} mat1={mat1} mat2={mat2} fail={fail}")
