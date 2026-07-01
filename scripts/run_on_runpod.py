"""Run LC reservoir designs on a RunPod GPU instance using GPUmeep.

Spins up a pod, clones both the reservoir repo and GPUmeep, uploads the chosen
data folder (with its cached LC director field), runs the GPU FDTD simulation,
pulls results back, and terminates the pod.

This is the reservoir-project-specific runner. It assumes:
  * the reservoir repo is at github.com:zcerne/reservoir.git
  * GPUmeep is at github.com:zcerne/GPUmeep.git
  * `class_simulation_gpu.py` (in this repo) drives the GPU FDTD

API key: read from `~/.runpod/api_key` or env RUNPOD_API_KEY. Never commit it.

Usage:
    python run_on_runpod.py --data data/test2D --gpu "NVIDIA GeForce RTX 4090" \
        --precision fp32

    python run_on_runpod.py --data data/source_mnist --gpu "NVIDIA H100 80GB HBM3" \
        --precision fp64 --empty       # vacuum reference run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def load_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    key_path = Path.home() / ".runpod" / "api_key"
    if key_path.exists():
        return key_path.read_text().strip()
    sys.exit("No RunPod API key. Set RUNPOD_API_KEY or save to ~/.runpod/api_key")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Local path to data folder (e.g. data/test2D). Uploaded to the pod.")
    p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090",
                   help="GPU type ID. RTX 4090 for fp32; H100 80GB for fp64.")
    p.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="COMMUNITY")
    p.add_argument("--image",
                   default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    p.add_argument("--disk", type=int, default=30, help="Container disk GB")
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--empty", action="store_true",
                   help="Vacuum reference run (no reservoir material)")
    p.add_argument("--script", default="class_simulation_gpu.py",
                   help="Reservoir script to run on the pod")
    p.add_argument("--no-terminate", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--ssh_key", default="~/.ssh/id_ed25519.pub",
                   help="Local SSH PUBLIC key; injected into the pod via PUBLIC_KEY env")
    p.add_argument("--gpumeep_dir",
                   default="~/Nextcloud/Doktorski/Projects/GPUmeep/gitcode",
                   help="Local GPUmeep checkout (its src/ is rsynced to the pod)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import runpod
    except ImportError:
        sys.exit("pip install runpod")
    runpod.api_key = load_api_key()

    repo_root = Path(__file__).resolve().parent
    data_path = (repo_root / args.data).resolve() if not os.path.isabs(args.data) else Path(args.data)
    if not data_path.exists():
        sys.exit(f"Data path not found: {data_path}")
    data_name = data_path.name

    # Read the local SSH public key — injected into the pod via the PUBLIC_KEY
    # env var (RunPod images write it to authorized_keys at boot). This avoids
    # needing to register the key in the RunPod dashboard.
    pubkey_path = Path(args.ssh_key).expanduser()
    if not pubkey_path.exists():
        sys.exit(f"SSH public key not found: {pubkey_path}. "
                 f"Generate one with: ssh-keygen -t ed25519")
    pubkey = pubkey_path.read_text().strip()
    privkey_path = str(pubkey_path)[:-4] if str(pubkey_path).endswith(".pub") else str(pubkey_path)

    pod_name = args.name or f"reservoir-{int(time.time())}"
    # --gpu may be a comma-separated list; try each until one has capacity.
    gpu_candidates = [g.strip() for g in args.gpu.split(",") if g.strip()]
    # ports="22/tcp" requests a directly-mapped public TCP port for SSH. Unlike
    # the ssh.runpod.io proxy (interactive only, NO scp/sftp/rsync), a real TCP
    # port supports rsync, which we need for upload/download.
    pod = None
    for gpu in gpu_candidates:
        print(f"[*] Trying GPU '{gpu}' ({args.cloud})...")
        try:
            pod = runpod.create_pod(
                name=pod_name, image_name=args.image, gpu_type_id=gpu,
                cloud_type=args.cloud, container_disk_in_gb=args.disk,
                support_public_ip=True, start_ssh=True,
                ports="22/tcp",
                env={"PUBLIC_KEY": pubkey},
            )
            print(f"[*] Got '{gpu}'")
            break
        except Exception as e:
            if "no longer any instances" in str(e).lower() or "instances available" in str(e).lower():
                print(f"    no capacity for '{gpu}', trying next...")
                continue
            raise
    if pod is None:
        sys.exit(f"No capacity on {args.cloud} for any of: {gpu_candidates}. "
                 f"Try --cloud SECURE, retry later, or add more GPU types to --gpu.")
    pod_id = pod["id"]
    print(f"[*] Pod ID: {pod_id}")

    try:
        # Wait for a directly-mapped TCP port 22 (privatePort 22 with a public
        # ip+port). The HTTP proxy port that community pods always expose is NOT
        # usable for rsync, so we specifically require the TCP-22 mapping.
        ssh_ip = ssh_port = None
        for attempt in range(72):  # up to 6 min
            time.sleep(5)
            info = runpod.get_pod(pod_id)
            ports = (info.get("runtime") or {}).get("ports") or []
            sp = next((p for p in ports if p.get("privatePort") == 22
                       and p.get("publicPort") and p.get("ip")), None)
            if sp:
                ssh_ip, ssh_port = sp["ip"], sp["publicPort"]
                break
            kinds = [(p.get("privatePort"), p.get("type")) for p in ports]
            print(f"  ... waiting for TCP-22 mapping ({attempt+1}); ports so far: {kinds}")
        if not ssh_ip:
            sys.exit("Timed out waiting for a public TCP port 22. The provider may "
                     "not support public IP — try --cloud SECURE or a different GPU.")
        print(f"[*] SSH (TCP): root@{ssh_ip}:{ssh_port}")

        ssh = ["ssh", "-i", privkey_path, "-p", str(ssh_port),
               "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null", f"root@{ssh_ip}"]
        rsync_ssh = (f"ssh -i {privkey_path} -p {ssh_port} "
                     f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")

        def rrun(cmd, check=True):
            print(f"  $ {cmd}")
            return subprocess.run(ssh + [cmd], check=check)

        def rsync_to(local_dir, remote_dir, extra_args=()):
            print(f"[*] rsync {local_dir} -> pod:{remote_dir}")
            subprocess.run([
                "rsync", "-a", "--info=progress2", *extra_args, "-e", rsync_ssh,
                str(local_dir).rstrip("/") + "/",
                f"root@{ssh_ip}:{remote_dir}/",
            ], check=True)

        # Install deps. rsync must exist on the pod too (the pytorch image
        # often lacks it) — rsync exit code 12 = "rsync not found on remote".
        rrun("which rsync || (apt-get update -qq && apt-get install -y -qq rsync)")
        rrun("pip install --upgrade 'jax[cuda12]' scipy matplotlib nlopt")

        # Upload code by rsync (NOT git clone). This avoids the public/private
        # repo + auth problem (GPUmeep is private) and the "must push first"
        # requirement — the pod always gets the exact LOCAL code.
        rrun("mkdir -p /workspace/reservoir /workspace/GPUmeep/src")
        # reservoir: just the .py files (skip data/, figures/, .git, etc.)
        rsync_to(repo_root, "/workspace/reservoir",
                 extra_args=("--include=*.py", "--include=*/", "--exclude=*"))
        # GPUmeep src
        gpumeep_src = Path(args.gpumeep_dir).expanduser() / "src"
        if not (gpumeep_src / "fdtd_core.py").exists():
            sys.exit(f"GPUmeep src not found at {gpumeep_src}. Set --gpumeep_dir.")
        rsync_to(gpumeep_src, "/workspace/GPUmeep/src")

        # Upload the data folder (includes cached lc_fields.npz) to a CLEAN
        # relative location on the pod, regardless of whether --data was a
        # relative or absolute local path. The pod always sees data/<name>.
        remote_data = f"data/{data_name}"
        rrun(f"mkdir -p /workspace/reservoir/data")
        print(f"[*] Uploading {data_path} -> pod:/workspace/reservoir/{remote_data}")
        subprocess.run([
            "rsync", "-a", "--info=progress2", "-e", rsync_ssh,
            str(data_path).rstrip("/") + "/",
            f"root@{ssh_ip}:/workspace/reservoir/{remote_data}/",
        ], check=True)

        # Run (pod-side path is the clean relative remote_data)
        extra = "--precision " + args.precision + (" --empty" if args.empty else "")
        runner = (
            f"cd /workspace/reservoir && "
            f"GPUMEEP_PATH=/workspace/GPUmeep/src "
            f"python {args.script} --path {remote_data} {extra}"
        )
        print("[*] Running simulation on pod...")
        rrun(runner)

        # Pull results back to the original local folder
        print("[*] Syncing results back")
        subprocess.run([
            "rsync", "-a", "--info=progress2", "-e", rsync_ssh,
            f"root@{ssh_ip}:/workspace/reservoir/{remote_data}/",
            str(data_path).rstrip("/") + "/",
        ], check=True)
        print(f"[*] Done. Results in {data_path}/")
    finally:
        if args.no_terminate:
            print(f"[*] Pod {pod_id} left running (--no-terminate)")
        else:
            print(f"[*] Terminating pod {pod_id}")
            try:
                runpod.terminate_pod(pod_id)
            except Exception as e:
                print(f"  warn: terminate failed: {e}")


if __name__ == "__main__":
    main()
