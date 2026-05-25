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

RESERVOIR_REPO = "https://github.com/zcerne/reservoir.git"
GPUMEEP_REPO = "https://github.com/zcerne/GPUmeep.git"


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

    pod_name = args.name or f"reservoir-{int(time.time())}"
    print(f"[*] Creating pod '{pod_name}' ({args.gpu}, {args.cloud})")
    pod = runpod.create_pod(
        name=pod_name, image_name=args.image, gpu_type_id=args.gpu,
        cloud_type=args.cloud, container_disk_in_gb=args.disk,
        support_public_ip=True, start_ssh=True,
    )
    pod_id = pod["id"]
    print(f"[*] Pod ID: {pod_id}")

    try:
        # Wait for SSH endpoint
        ssh_ip = ssh_port = None
        for attempt in range(60):
            time.sleep(5)
            info = runpod.get_pod(pod_id)
            ports = (info.get("runtime") or {}).get("ports") or []
            sp = next((p for p in ports if p.get("privatePort") == 22), None)
            if sp and sp.get("publicPort"):
                ssh_ip, ssh_port = sp["ip"], sp["publicPort"]
                break
            print(f"  ... waiting for SSH ({attempt+1})")
        if not ssh_ip:
            sys.exit("Timed out waiting for SSH")
        print(f"[*] SSH: root@{ssh_ip}:{ssh_port}")

        ssh = ["ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null", f"root@{ssh_ip}"]
        rsync_ssh = (f"ssh -p {ssh_port} -o StrictHostKeyChecking=no "
                     f"-o UserKnownHostsFile=/dev/null")

        def rrun(cmd, check=True):
            print(f"  $ {cmd}")
            return subprocess.run(ssh + [cmd], check=check)

        # Install deps + clone both repos on the pod
        rrun("pip install --upgrade 'jax[cuda12]' scipy matplotlib nlopt")
        rrun(f"cd /workspace && git clone --depth 1 {RESERVOIR_REPO} reservoir")
        rrun(f"cd /workspace && git clone --depth 1 {GPUMEEP_REPO} GPUmeep")

        # Upload the data folder (includes cached lc_fields.npz)
        rrun(f"mkdir -p /workspace/reservoir/{os.path.dirname(args.data) or '.'}")
        print(f"[*] Uploading {data_path} -> pod")
        subprocess.run([
            "rsync", "-a", "--info=progress2", "-e", rsync_ssh,
            str(data_path) + "/",
            f"root@{ssh_ip}:/workspace/reservoir/{args.data}/",
        ], check=True)

        # Run
        extra = "--precision " + args.precision + (" --empty" if args.empty else "")
        # GPUmeep repo root IS its gitcode contents (src/ at top level after clone)
        runner = (
            f"cd /workspace/reservoir && "
            f"GPUMEEP_PATH=/workspace/GPUmeep/src "
            f"python {args.script} --path {args.data} {extra}"
        )
        print("[*] Running simulation on pod...")
        rrun(runner)

        # Pull results back
        print("[*] Syncing results back")
        subprocess.run([
            "rsync", "-a", "--info=progress2", "-e", rsync_ssh,
            f"root@{ssh_ip}:/workspace/reservoir/{args.data}/",
            str(data_path) + "/",
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
