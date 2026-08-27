#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def module_info(name: str) -> dict:
    try:
        module = __import__(name)
    except Exception as exc:
        return {"error": repr(exc)}
    return {
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "commit": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "branch": command("git", "-C", str(path), "branch", "--show-current"),
        "status": command("git", "-C", str(path), "status", "--short"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-stats", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    payload = {
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "MUJOCO_GL",
                "MUJOCO_EGL_DEVICE_ID",
                "MAGICK_HOME",
                "LD_LIBRARY_PATH",
                "LIBERO_CONFIG_PATH",
                "DIFFSYNTH_DOWNLOAD_SOURCE",
            )
        },
        "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "gpus": command(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ).splitlines(),
        "modules": {
            name: module_info(name)
            for name in ("torch", "torchvision", "mujoco", "robosuite", "libero", "fastwam")
        },
        "repositories": {"rankwam": git_info(root)},
        "artifacts": {},
    }
    if args.libero_root:
        payload["repositories"]["libero"] = git_info(args.libero_root)
    for key, value in (("checkpoint", args.checkpoint), ("dataset_stats", args.dataset_stats)):
        if value:
            resolved = value.resolve()
            payload["artifacts"][key] = {
                "path": str(resolved),
                "exists": resolved.is_file(),
                "sha256": sha256(resolved) if resolved.is_file() else None,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
