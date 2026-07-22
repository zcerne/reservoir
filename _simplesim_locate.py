"""Locate the canonical SimpleSim package so `import simplesim` resolves.

Same pattern as _lcrelax_locate: walk UP from this repo for a SimpleSim
checkout (gitcode layout or plain), else SIMPLESIM_PATH, else the Nextcloud
projects dir. SimpleSim itself locates LCrelax + GPUmeep.
"""
from __future__ import annotations

import os
import sys


def ensure() -> None:
    if "simplesim" in sys.modules:
        return
    cands = [os.environ.get("SIMPLESIM_PATH")]
    d = os.path.dirname(os.path.abspath(__file__))
    while d not in ("/", ""):
        cands += [os.path.join(d, "SimpleSim", "gitcode"),
                  os.path.join(d, "SimpleSim")]
        d = os.path.dirname(d)
    cands += [os.path.expanduser("~/Nextcloud/Doktorski/Projects/SimpleSim/gitcode"),
              "/home/ziga/Orion/SimpleSim", "/home/cernez/SimpleSim"]
    for p in cands:
        if p and os.path.isdir(os.path.join(p, "simplesim")):
            if p not in sys.path:
                sys.path.append(p)     # append: never shadow this repo's modules
            return
    raise ModuleNotFoundError(
        "SimpleSim package not found: set SIMPLESIM_PATH to its gitcode dir "
        "or keep a checkout next to Reservoir.")


ensure()
