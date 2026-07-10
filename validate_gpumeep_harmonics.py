"""Single-point gpumeep-vs-MEEP validation for the STED reservoir harmonics.

Runs ONE harmonics work item through the GPUmeep full-vector STED engine and
compares monitor_2 |Ey|(y) against the MEEP result already stored in
datasets/harmonics.npz (outputs[item]). Use before committing both smaugs to
the full 32-sample sweep.

  python validate_gpumeep_harmonics.py --path <design> [--item 0] [--run_until 450]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import class_simulation_gpu as cg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--item", type=int, default=0)
    ap.add_argument("--run_until", type=float, default=None,
                    help="override JSON run_until (fast smoke test)")
    ap.add_argument("--source_key", default="source_1")
    args = ap.parse_args()

    h = np.load(os.path.join(args.path, "datasets", "harmonics.npz"))
    E_in = h["inputs"][args.item]          # complex per-strip amplitude
    meep_out = np.asarray(h["outputs"][args.item]).ravel()   # complex Ey(y), MEEP

    sim = cg.SimulationGPU(folder_path=args.path)
    if args.run_until:
        sim.run_until_override = args.run_until
    Ey, Ex, Ez = sim.run_basis([complex(v).real for v in E_in],
                                source_key=args.source_key)
    gp = np.asarray(Ey).ravel()

    n = min(len(gp), len(meep_out))
    gp = gp[:n]; mo = meep_out[:n]
    a = np.abs(gp); b = np.abs(mo)
    corr = float(np.corrcoef(a, b)[0, 1]) if n > 1 and b.max() > 0 else float("nan")
    ratio = float(a.max() / b.max()) if b.max() > 0 else float("nan")
    # complex correlation (phase-aware)
    ccorr = (np.abs(np.vdot(gp, mo)) / (np.linalg.norm(gp) * np.linalg.norm(mo))
             if np.linalg.norm(gp) > 0 and np.linalg.norm(mo) > 0 else float("nan"))

    print("=" * 60)
    print(f"ITEM {args.item}  input={E_in.real.tolist()}")
    print(f"len: gpumeep={len(Ey)}  meep={len(meep_out)}")
    print(f"|Ey| shape-corr = {corr:.4f}")
    print(f"|Ey| complex-corr = {ccorr:.4f}")
    print(f"max-ratio gp/meep = {ratio:.4f}")
    print(f"gpumeep |Ey|: max {a.max():.4g}  mean {a.mean():.4g}")
    print(f"MEEP    |Ey|: max {b.max():.4g}  mean {b.mean():.4g}")
    print("=" * 60)
    out = f"/tmp/gpumeep_vs_meep_item{args.item}.npz"
    np.savez(out, gp=gp, meep=mo)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
