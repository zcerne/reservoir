"""Locate GPUmeep and import the gpumeep API safely.

gpumeep.py internally does `from class_simulation_gpu import ...` — meaning
its OWN engine-support module in GPUmeep/src, which THIS repo's
class_simulation_gpu.py shadows on sys.path. Importing gpumeep while the
reservoir module owns (or is mid-import under) that name would bind the
wrong module. Fix: temporarily seat GPUmeep's genuine engine module under
the name, import gpumeep, then restore whatever was there.

Usage (all GPU class files):  from gpumeep_setup import gm, FS_PER_MEEP
"""
from __future__ import annotations

import importlib.util
import os
import sys

FS_PER_MEEP = 3.335640952        # fs per MEEP time unit (a = 1 µm)

_candidates = [
    os.environ.get("GPUMEEP_PATH"),
    os.path.expanduser("~/Nextcloud/Doktorski/Projects/GPUmeep/gitcode/src"),
    os.path.expanduser("~/GPUmeep/gitcode/src"),
    os.path.expanduser("~/GPUmeep/src"),
    os.path.dirname(os.path.abspath(__file__)),
]
for _p in _candidates:
    if _p and os.path.exists(os.path.join(_p, "gpumeep.py")):
        GM_SRC = _p
        break
else:
    raise ImportError("Could not find GPUmeep src/ (set GPUMEEP_PATH)")
if GM_SRC not in sys.path:
    sys.path.insert(0, GM_SRC)

if "gpumeep" not in sys.modules:
    if "_gpumeep_engine_csg" in sys.modules:
        _engine = sys.modules["_gpumeep_engine_csg"]
    else:
        _spec = importlib.util.spec_from_file_location(
            "_gpumeep_engine_csg", os.path.join(GM_SRC, "class_simulation_gpu.py"))
        _engine = importlib.util.module_from_spec(_spec)
        sys.modules["_gpumeep_engine_csg"] = _engine
        _spec.loader.exec_module(_engine)
    _prev = sys.modules.get("class_simulation_gpu")
    sys.modules["class_simulation_gpu"] = _engine
    try:
        import gpumeep  # noqa: F401
    finally:
        if _prev is not None:
            sys.modules["class_simulation_gpu"] = _prev
        else:
            del sys.modules["class_simulation_gpu"]

import gpumeep as gm  # noqa: E402

JDTYPE = sys.modules["_gpumeep_engine_csg"]._JDTYPE
