"""Shared helpers for the characterization data generators (index/assemble pattern).

Each generator enumerates a deterministic list of input vectors (from a seed) and
supports three modes:
  --index K   : run ONE forward (input K) → write <out>.parts/part_K.npz immediately
                (this IS the incremental save; a killed array loses only unfinished tasks).
  --assemble  : gather all parts → the final <out>.npz with the analysis's keys.
  --serial    : loop all indices in one process, part-saving each (incremental), then
                assemble. Fallback when not array-parallelizing.
  --count     : print the number of work items (for `sbatch --array=0-(N-1)`).

Array-parallel: `slurm_char_array.sh` runs `--index $SLURM_ARRAY_TASK_ID` per task,
then a final `--assemble`. Wall-clock ≈ one forward run, not N of them.
"""
from __future__ import annotations
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# JAX persistent compilation cache — must be set BEFORE jax is imported anywhere.
#
# gpumeep runs the FDTD as one lax.scan, and XLA needs 20+ MINUTES to compile it
# for the heavier designs (DBR mirrors + MultilevelAtom gain + monitors). Since
# forward() rebuilds the Simulation for every sample, that compile was being paid
# per sample: a py-spy dump of a "hung" worker (2026-07-30) sat in
# backend_compile_and_load, which is what made 05_adding_mirror look deadlocked on
# both smaug and lips. With the cache, the first sample pays it once and every
# later sample — plus every other worker sharing the directory — loads the
# executable instead. Override the location with JAX_COMPILATION_CACHE_DIR
# (on lips point it at /project/cerneziga/.jax_cache).
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.expanduser("~/.cache/jax_compile"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
import numpy as np


def _load_extras(out_dir, out_sensor, suffix):
    """Everything worth keeping per sample beyond the readout vector:

      m2_*  monitor_2's complex DFT fields (61 freqs x 402 points x Ex/Ey/Ez,
            ~1.2 MB) — the in-cell spectral readout
      eq_*  the near2far EQUIVALENCE CURRENTS (~46 kB) — the raw near-field
            data the far field is computed FROM. With these the far field can
            be re-rendered at any distance, angle or resolution without ever
            re-running FDTD, which the far-field map alone cannot do because
            its screen geometry is baked in.

    Missing files are skipped rather than fatal (the MEEP backend does not
    emit equivalence currents at all)."""
    extras = {}
    m2 = os.path.join(out_dir, f"monitor_2_{suffix}.npz")
    if os.path.exists(m2):
        with np.load(m2) as d:
            extras.update({f"m2_{k}": d[k] for k in d.files})
    if out_sensor:
        eq = os.path.join(out_dir, f"{out_sensor}_n2f_eq_{suffix}.npz")
        if not os.path.exists(eq):                      # pre-suffix fallback
            eq = os.path.join(out_dir, f"{out_sensor}_n2f_eq.npz")
        if os.path.exists(eq):
            with np.load(eq) as d:
                extras.update({f"eq_{k}": d[k] for k in d.files})
    return extras


def open_reservoir(path, components, out_sensor=None, full_sensor=False,
                   with_extras=False, n_sources=None):
    """Load the fixed reservoir; return (forward_fn, n_strips, is_master).

    forward(E): real/complex input amplitudes (n_strips,) → stacked complex sensor
    field over `components` (Ey[,Ex,Ez]). NOTE the source casts amplitude to real,
    so pass REAL amplitudes unless a tone's imaginary part is intended as a phase.

    with_extras: after each run, populate `forward.extras` with monitor_2's
    fields and the near2far equivalence currents, so a generator can store them
    in the part file (see _load_extras).

    full_sensor: with out_sensor, return the sensor's ENTIRE array per component
    (the whole near2far map, nx*ny values each) instead of just its last column.
    Much larger — a 200x200 map over 3 components is ~1.9 MB per sample — but it
    keeps every spatial channel available, so a readout can be chosen later.
    The assembled npz records `sensor_shape` so the flat vector can be reshaped.

    out_sensor: alternative readout — key of a sensor whose npz to use instead of
    monitor_2. For a near2far area map the output is the COMPLEX field of the
    map's LAST COLUMN, i.e. EH[-1, :, c] for each requested component (the far
    screen at max x, 1D over y). NOT the stored `E2` intensity: parts must keep
    the raw complex field so `--readout intensity` can apply |E|² at assemble
    time, and so a field→field map stays linear-testable (an intensity readout
    annihilates the fundamentals and manufactures order-2 products of its own).
    Other npz types fall back to the `components` stacking above.

    Runs through SimpleSim's ReservoirSimulation (class_simulation.py) — the
    current, actively-maintained engine (same one the interactive run()/plot()
    workflow uses) — for BOTH backends. The old standalone class_simulation_gpu.py
    run_basis()/_run_2d_sted() path and the (stale, broken since the SimpleSim
    migration) class_simulation_T.py path are no longer used here: they're a
    separate, unsynced reimplementation that was never validated against
    ReservoirSimulation and gave measurably wrong gain for STED designs
    (2026-07-25: ~2.5x vs SimpleSim's ~4.1x at the same amplitude/pump).
    """
    import json as _json
    with open(os.path.join(path, "simulation_data.json")) as _f:
        cfg = _json.load(_f)
    backend = str(cfg.get("solver", "meep")).lower()
    # Env override: the same design JSON can run MEEP on Orion (CPU/MPI) and
    # GPUmeep on smaug without editing the file.
    backend = os.environ.get("RESERVOIR_SOLVER", backend).lower()
    backend = "gpumeep" if backend in ("gpumeep", "gpu", "gpumma") else "meep"

    # SimpleSim lives in a sibling checkout, not on PYTHONPATH — resolve it the
    # same way run.py does (SIMPLESIM_PATH, then a walk up from the repo for
    # SimpleSim/gitcode or SimpleSim). Without this every data_gen script dies
    # with ModuleNotFoundError on any machine that doesn't happen to export
    # PYTHONPATH (e.g. smaug over non-login ssh).
    import run as _run
    _run._ensure_simplesim()
    from simplesim import Simulation as ReservoirSimulation

    # First source in JSON key order that isn't "source_2" (same convention the
    # old pipeline used) -- for 02_adding_pump this is source_1 (signal), not
    # source_pump (fixed CW pump, comes later in the JSON).
    src_key = next(k for k, v in cfg.items()
                   if isinstance(v, dict) and v.get("class") == "source"
                   and k != "source_2")
    amp0 = cfg[src_key].get("amplitude", [1.0])
    n_strips = len(amp0) if isinstance(amp0, (list, tuple)) else 1
    # Source-count variant WITHOUT a design-folder copy (standing convention,
    # 2026-08-03): forward() replaces the amplitude list wholesale on every run,
    # and SimpleSim derives the strip layout from that list's length — so a
    # different n_strips here IS the whole "4src" variant.
    if n_sources is not None:
        n_strips = int(n_sources)

    # Per-process scratch tag (matches the old GPUMEEP_SCRATCH_TAG use): two
    # concurrent characterization batches (e.g. split across smaug1/smaug2)
    # must not share one suffix, or they'd race on the same output npz.
    suffix = os.environ.get("SIMPLESIM_SCRATCH_TAG", "fwd")
    out_dir = os.path.join(path, f"simulation_{backend}")

    from simplesim.simulation import resolve_engine
    _mp = resolve_engine(backend)
    is_master = bool(_mp.am_master()) if hasattr(_mp, "am_master") else True

    # Dataset forwards never use population SNAPSHOTS (whole-cell E + N at
    # snap_interval → 2.1 GB/run on 04_adding_LC, rewritten every forward):
    # force snap_interval=0 on every population monitor, keeping only the
    # small N(t) trace.
    pop_off = {k: {"snap_interval": 0}
               for k, v in cfg.items()
               if isinstance(v, dict) and v.get("class") == "monitor"
               and str(v.get("type", "")) == "population"}

    def forward(E):
        ov = {src_key: {"amplitude": list(E)}}
        ov.update(pop_off)
        sim = ReservoirSimulation(path, backend=backend, suffix=suffix,
                                  overrides=ov)
        sim.relax()
        sim.run(empty=False)
        # MEEP writes monitor and near2far output from the MASTER rank only, so
        # a non-master rank has nothing to read back. Every rank still has to
        # run the FDTD above (it is an MPI collective), but only master's return
        # value is ever used — save_part() discards the others. Without this
        # guard each non-master rank raises FileNotFoundError on
        # <sensor>_<tag>.npz, some die, the survivors block in the next
        # collective, and the job deadlocks: observed 2026-07-30 on the 05
        # amp_sweep, one item written in 2h44m with 16 ranks pinned at 100%.
        # (single_source_sweep escaped it only because monitor_2 is written by
        # every rank.)
        if not is_master:
            return None
        if with_extras:
            forward.extras = _load_extras(out_dir, out_sensor, suffix)
        if out_sensor:
            d = np.load(os.path.join(out_dir, f"{out_sensor}_{suffix}.npz"))
            if "EH" in d.files:
                ci = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
                EH = np.asarray(d["EH"])
                if full_sensor:
                    # the WHOLE far-field map per component, (nx, ny) each,
                    # flattened C-order and concatenated in `components` order
                    # (reshape with sensor_shape from the assembled npz)
                    return np.concatenate([EH[:, :, ci[c]].ravel()
                                           for c in components])
                return np.concatenate([EH[-1, :, ci[c]].ravel()
                                       for c in components])
            m2 = d
        else:
            m2 = np.load(os.path.join(out_dir, f"monitor_2_{suffix}.npz"))
        zeros = None
        vals = []
        for c in components:
            if c in m2.files:
                v = np.asarray(m2[c])
                zeros = np.zeros_like(v)
            vals.append(m2[c] if c in m2.files else None)
        vals = [v if v is not None else zeros for v in vals]
        return np.concatenate([np.asarray(v).ravel() for v in vals])

    return forward, n_strips, is_master


def _parts_dir(out_path):
    return out_path + ".parts"


def save_part(out_path, k, is_master, **arrays):
    """Write one part file (master rank only, MPI-safe)."""
    if not is_master:
        return
    d = _parts_dir(out_path)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"part_{int(k):06d}.npz"), idx=int(k), **arrays)


def part_exists(out_path, k):
    """True if work item k already has a part file."""
    return os.path.exists(os.path.join(_parts_dir(out_path), f"part_{int(k):06d}.npz"))


def load_parts(out_path):
    """Return parts as a list of dicts sorted by idx. Errors if any are missing/gapped."""
    d = _parts_dir(out_path)
    files = sorted(glob.glob(os.path.join(d, "part_*.npz")))
    if not files:
        raise SystemExit(f"no parts in {d} — run --index tasks (or --serial) first")
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    parts.sort(key=lambda p: int(p["idx"]))
    idxs = [int(p["idx"]) for p in parts]
    if idxs != list(range(len(idxs))):
        missing = sorted(set(range(max(idxs) + 1)) - set(idxs))
        raise SystemExit(f"parts incomplete in {d}: {len(idxs)} present, missing idx {missing[:10]}...")
    return parts


def run_mode(args, n_items, run_one, assemble, is_master, out_path=None):
    """Dispatch --count / --index / --serial / --assemble. `run_one(k)` executes one
    forward + save_part; `assemble()` builds the final npz. Returns an exit code.

    `out_path` enables `--skip_existing` for EVERY generator: work items whose part
    file already exists are skipped before the (expensive) forward run. The flag is
    registered for all generators by add_common_args but used to be implemented only
    inside generate_ipc_data's own run_one, so every other probe set silently redid
    finished work on a restart — which is exactly what a restart is for (found
    2026-07-29 after a VRAM crash cost a partly-finished superposition set)."""
    skip = getattr(args, "skip_existing", False) and out_path is not None

    def _run(k):
        if skip and part_exists(out_path, k):
            return False
        run_one(k)
        return True

    if getattr(args, "count", False):
        print(n_items)                                        # for sbatch --array
        return 0
    if getattr(args, "index", None) is not None:
        k = int(args.index)
        if not (0 <= k < n_items):
            raise SystemExit(f"--index {k} out of range [0,{n_items})")
        _run(k)
        return 0
    if getattr(args, "batch", None) is not None:
        S = int(args.batch_size); lo = int(args.batch) * S; hi = min(lo + S, n_items)
        if lo >= n_items:
            raise SystemExit(f"--batch {args.batch} (size {S}) starts at {lo} ≥ n_items {n_items}")
        for k in range(lo, hi):
            ran = _run(k)
            if is_master:
                print(f"[gen] batch {args.batch}: {k-lo+1}/{hi-lo} (idx {k}/{n_items})"
                      f"{'' if ran else ' [skipped, part exists]'}", flush=True)
        return 0
    if getattr(args, "assemble", False):
        if is_master:
            assemble()
        return 0
    # --serial (default): loop all, part-save each (incremental), then assemble.
    # --reverse iterates high→low so a second (backward) worker can share the job
    # with a forward array run and meet in the middle (run_one skips done parts).
    order = range(n_items - 1, -1, -1) if getattr(args, "reverse", False) else range(n_items)
    for i, k in enumerate(order):
        ran = _run(k)
        if is_master:
            print(f"[gen] serial {i+1}/{n_items} (idx {k})"
                  f"{'' if ran else ' [skipped, part exists]'}", flush=True)
    if is_master:
        assemble()
    return 0


def add_common_args(ap):
    ap.add_argument("--index", type=int, default=None, help="run one work item K → part file")
    ap.add_argument("--batch", type=int, default=None, help="run item batch B: indices [B*batch_size, +batch_size)")
    ap.add_argument("--batch_size", type=int, default=50, help="items per batch (with --batch)")
    ap.add_argument("--assemble", action="store_true", help="combine parts → final npz")
    ap.add_argument("--count", action="store_true", help="print #work items (for sbatch --array)")
    ap.add_argument("--components", default="Ey", help="sensor components to save (Ey[,Ex,Ez])")
    ap.add_argument("--out_sensor", default=None,
                    help="sensor key to read as output instead of monitor_2 "
                         "(near2far map -> E2 last column at max x, 1D over y)")
    ap.add_argument("--n_sources", type=int, default=None,
                    help="override the input strip count (e.g. 4 on a 2-strip "
                         "design) — in-memory only, no design copy; SimpleSim "
                         "lays out strips from the per-run amplitude list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="serial: iterate indices high→low")
    ap.add_argument("--skip_existing", action="store_true", help="skip an index whose part file already exists")
