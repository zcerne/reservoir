"""Pump × reflectivity grid on 05_adding_mirror — find the cavity's operating point.

For each (pump, R) the design JSON is copied into
05_adding_mirror/grid/p<pump>_R<R>/ with both mirrors set to R and the pump
amplitude replaced; R="none" removes the mirrors entirely (the no-cavity
baseline at that pump, needed for enhancement = cavity / no-cavity at equal
pump). run_until is raised to 200 so the cavity ring-down isn't cut as hard
as the design default 100 would (photon lifetime at R≈0.9 is several hundred
time units — even 200 undershoots at the top of the grid; treat those points
as lower bounds).

Each variant is a self-contained design dir because the overrides mechanism
only patches dict-valued JSON keys, so run_until/reflectivity cannot be
overridden through it.

  python scripts/grid05_cavity.py --combos p200_Rnone,p200_R0.5,...   # worker
  python scripts/grid05_cavity.py --report                            # summarize
"""
from __future__ import annotations
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "lasing_testing", "05_adding_mirror")
RUN_UNTIL = 200.0


def make_variant(pump: float, R):
    tag = f"p{pump:g}_R{'none' if R is None else f'{R:g}'}"
    dst = os.path.join(BASE, "grid", tag)
    cfg = json.load(open(os.path.join(BASE, "simulation_data.json")),
                    object_pairs_hook=collections.OrderedDict)
    cfg["run_until"] = RUN_UNTIL
    cfg["dft_t_end"] = RUN_UNTIL * 2
    cfg["source_pump"]["amplitude"] = float(pump)
    if R is None:
        for m in ("mirror_1", "mirror_2"):
            cfg.pop(m, None)
        cfg["object_order"] = [k for k in cfg["object_order"]
                               if k not in ("mirror_1", "mirror_2")]
    else:
        for m in ("mirror_1", "mirror_2"):
            cfg[m]["reflectivity"] = float(R)
    os.makedirs(dst, exist_ok=True)
    json.dump(cfg, open(os.path.join(dst, "simulation_data.json"), "w"), indent=2)
    return tag, dst


def run_combo(spec: str):
    pump_s, r_s = spec.split("_R")
    pump = float(pump_s[1:])
    R = None if r_s == "none" else float(r_s)
    tag, dst = make_variant(pump, R)
    out = os.path.join(dst, "simulation_gpumeep", "monitor_2.npz")
    if os.path.exists(out):
        print(f"[grid] {tag}: exists, skipping", flush=True)
        return
    import run as _run
    _run._ensure_simplesim()
    from simplesim import Simulation
    print(f"[grid] {tag}: running", flush=True)
    Simulation(dst, backend="gpumeep").run(empty=False)


def report():
    rows = []
    gdir = os.path.join(BASE, "grid")
    for tag in sorted(os.listdir(gdir)) if os.path.isdir(gdir) else []:
        f = os.path.join(gdir, tag, "simulation_gpumeep", "monitor_2.npz")
        if not os.path.exists(f):
            rows.append((tag, None)); continue
        d = np.load(f)
        rows.append((tag, float(np.sum(np.abs(d["Ey"]) ** 2))))
    base_p = {t.split("_R")[0]: p for t, p in rows if t.endswith("Rnone") and p}
    print(f"{'combo':14} {'P(monitor_2)':>13} {'vs no-mirror':>12} {'vs passive':>11}")
    for tag, p in rows:
        if p is None:
            print(f"{tag:14} {'pending':>13}"); continue
        pump_tag = tag.split("_R")[0]
        ref = base_p.get(pump_tag)
        enh = p / ref if ref else float("nan")
        r_s = tag.split("_R")[1]
        note = ""
        if r_s != "none":
            # quantised achieved R for the passive expectation
            from simplesim.mirror import _n_layers_for_reflectivity, _achieved_reflectivity
            n = _n_layers_for_reflectivity(float(r_s), 1.46, 2.4)
            Ra = _achieved_reflectivity(n, 1.46, 2.4)
            note = f"{enh / (1 - Ra) ** 2:11.2f}"
        print(f"{tag:14} {p:13.4g} {enh:12.3f} {note:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default=None,
                    help="comma list like p200_Rnone,p300_R0.8")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    for spec in [c.strip() for c in (a.combos or "").split(",") if c.strip()]:
        run_combo(spec)


if __name__ == "__main__":
    main()
