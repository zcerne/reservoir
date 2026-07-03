"""Sync the data/ FOLDER STRUCTURE + a generated-datasets MANIFEST into git, so you
can tell from the repo (any machine) what's already been generated on Orion — without
committing the heavy .npz.

What it does, scanning a data root (default: the Orion mount):
  1. writes `.gitkeep` into every subdirectory  → the empty folder tree is versioned
     (git can't track empty dirs otherwise; the .npz themselves stay gitignored).
  2. writes `data/GENERATED.md` — a tree of every design showing which datasets /
     stats / sim files are present (✓) with sizes → the "is it already generated?" view.

  # scan Orion, write .gitkeep + manifest INTO the local repo's data/ dir
  python scripts/sync_data_structure.py \
      --data-root /home/ziga/Orion/resevoir/data \
      --repo-data /home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode/data

Then commit — .gitkeep + GENERATED.md are whitelisted in .gitignore.
"""
from __future__ import annotations
import argparse, os


def _human(n):
    x = float(n)
    for u in ("B", "K", "M", "G"):
        if x < 1024 or u == "G":
            return f"{x:.0f}{u}" if u == "B" else f"{x:.1f}{u}"
        x /= 1024.0


def _sz(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="data dir to SCAN (e.g. Orion mount)")
    ap.add_argument("--repo-data", required=True, help="repo data/ dir to write .gitkeep + GENERATED.md")
    ap.add_argument("--keep-ext", default=".npz,.png,.pt",
                    help="file extensions to list in the manifest")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.data_root))
    repo = os.path.abspath(os.path.expanduser(args.repo_data))
    exts = tuple(e.strip() for e in args.keep_ext.split(",") if e.strip())

    n_keep = 0
    lines = ["# Generated data manifest",
             "",
             f"_Scanned `{root}` — ✓ = file present (sizes shown). Heavy data is gitignored;",
             "this manifest + the folder tree are what's versioned so you can see what exists._",
             ""]

    # walk designs at data-root; mirror empty dirs into repo via .gitkeep.
    # Skip .parts dirs — they hold thousands of transient part files; the assembled
    # dataset + the manifest's part-count is what matters, not the per-part tree.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.endswith(".parts")]
        rel = os.path.relpath(dirpath, root)
        if rel != "." and os.path.basename(dirpath).endswith(".parts"):
            continue
        repo_dir = repo if rel == "." else os.path.join(repo, rel)
        os.makedirs(repo_dir, exist_ok=True)
        gk = os.path.join(repo_dir, ".gitkeep")
        if not os.path.exists(gk):
            open(gk, "a").close()
        n_keep += 1

    # build the manifest: every directory that contains data files, listed with
    # its relative path + present files (+ .parts part-counts). Grouped by top-level.
    n_dirs = 0
    # NB: iterate os.walk directly (not sorted(os.walk)) so the in-place dirnames[:]
    # pruning actually stops descent into .parts dirs; sort within each level instead.
    for dirpath, dirnames, filenames in os.walk(root):
        parts_dirs = sorted(d for d in dirnames if d.endswith(".parts"))
        # summarize .parts as a count at the parent; don't descend into them (noise)
        dirnames[:] = sorted(d for d in dirnames if not d.endswith(".parts"))
        rel = os.path.relpath(dirpath, root)
        data_files = sorted(f for f in filenames if f.endswith(exts))
        if not data_files and not parts_dirs:
            continue
        n_dirs += 1
        header = "root" if rel == "." else rel
        lines.append(f"\n### `{header}/`")
        for f in data_files:
            lines.append(f"- ✓ `{f}`  ({_human(_sz(os.path.join(dirpath, f)))})")
        for pd in parts_dirs:
            pp = os.path.join(dirpath, pd)
            n = len([x for x in os.listdir(pp) if x.endswith(".npz")])
            lines.append(f"- `{pd}/`  ({n} part files)")

    out = os.path.join(repo, "GENERATED.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[sync] {n_keep} .gitkeep written under {repo}")
    print(f"[sync] manifest → {out}  ({n_dirs} dirs with data)")


if __name__ == "__main__":
    raise SystemExit(main())
