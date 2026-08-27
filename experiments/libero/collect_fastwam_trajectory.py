#!/usr/bin/env python3
"""Collect a bounded, resumable FastWAM baseline trajectory for LIBERO."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from libero.libero import benchmark
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, LIBERO_EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.libero.collect_fastwam_candidates import (
    PROPRIO_KEYS,
    _model_dimensions,
    _predict,
)
from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from experiments.libero.task_progress import score_task_progress
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--max-policy-steps", type=int, default=400)
    parser.add_argument("--max-states", type=int, default=40)
    parser.add_argument("--max-cache-mb", type=float, default=500.0)
    parser.add_argument(
        "--image-stride",
        type=int,
        default=1,
        help="Save JPEG observations every N checkpoints; 0 disables images.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--resume", action="store_true")
    args, hydra_overrides = parser.parse_known_args()
    if args.max_policy_steps < 0:
        raise ValueError("max-policy-steps cannot be negative")
    if args.max_states <= 0:
        raise ValueError("max-states must be positive")
    if args.max_cache_mb <= 0:
        raise ValueError("max-cache-mb must be positive")
    if args.image_stride < 0:
        raise ValueError("image-stride cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be in [1, 100]")
    return args, hydra_overrides


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def save_jpeg(path: Path, array: np.ndarray, quality: int) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(
        temporary, format="JPEG", quality=quality, optimize=True
    )
    os.replace(temporary, path)


def check_budget(output_dir: Path, max_cache_bytes: int) -> int:
    used = directory_size_bytes(output_dir)
    if used > max_cache_bytes:
        raise RuntimeError(
            f"trajectory cache budget exceeded: {used} > {max_cache_bytes} bytes"
        )
    return used


def main() -> None:
    args, hydra_overrides = parse_args()
    trajectory_path = args.output_dir / "trajectory.npz"
    metadata_path = args.output_dir / "metadata.json"
    progress_path = args.output_dir / "progress.json"
    image_dir = args.output_dir / "images"

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --resume or choose a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.image_stride > 0:
        image_dir.mkdir(exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="sim_libero", overrides=hydra_overrides)
    cfg.EVALUATION.output_dir = str(args.output_dir.resolve())
    if cfg.ckpt is None:
        raise ValueError("pass ckpt=/path/to/checkpoint as a Hydra override")

    started = time.time()
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    action_horizon, input_w, input_h = _model_dimensions(cfg)

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    if not 0 <= args.trial_id < len(initial_states):
        raise ValueError("trial-id is outside the benchmark initial-state range")
    initial_state = np.asarray(initial_states[args.trial_id]).copy()
    env_seed = int(cfg.get("seed", 42))
    dummy_steps = int(cfg.EVALUATION.num_steps_wait)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    max_cache_bytes = int(args.max_cache_mb * 1024 * 1024)

    prefix_actions: list[np.ndarray] = []
    checkpoint_steps: list[int] = []
    checkpoint_action_offsets: list[int] = []
    checkpoint_states: list[np.ndarray] = []
    checkpoint_proprio: dict[str, list[np.ndarray]] = {key: [] for key in PROPRIO_KEYS}
    progress_records: list[dict[str, Any] | None] = []
    resumed = False
    if args.resume and trajectory_path.exists():
        with np.load(trajectory_path, allow_pickle=False) as stored:
            stored_initial = np.asarray(stored["initial_state"])
            initial_state_max_abs = float(np.max(np.abs(stored_initial - initial_state)))
            if initial_state_max_abs > 1e-7:
                raise RuntimeError(
                    "saved initial state does not match requested trial: "
                    f"max_abs={initial_state_max_abs}"
                )
            prefix_actions = [row.copy() for row in stored["prefix_actions"]]
            checkpoint_steps = stored["checkpoint_policy_steps"].astype(int).tolist()
            checkpoint_action_offsets = stored["checkpoint_action_offsets"].astype(int).tolist()
            checkpoint_states = [row.copy() for row in stored["checkpoint_states"]]
            for key in PROPRIO_KEYS:
                checkpoint_proprio[key] = [
                    row.copy() for row in stored[f"checkpoint_{key}"]
                ]
        if progress_path.exists():
            progress_records = json.loads(progress_path.read_text(encoding="utf-8"))
        else:
            progress_records = [None] * len(checkpoint_steps)
        if len(progress_records) != len(checkpoint_steps):
            raise RuntimeError("progress record count does not match trajectory checkpoints")
        resumed = True

    np.random.seed(env_seed)
    env, task_description = get_libero_env(
        task, LIBERO_ENV_RESOLUTION, env_seed, env_num=1
    )
    done = False
    try:
        env.reset()
        obs = env.set_init_state(initial_state.copy())
        if resumed:
            for action in prefix_actions:
                obs, _, done, _ = env.step(action)
                if done:
                    break
            if checkpoint_states:
                replay_max_abs = float(
                    np.max(np.abs(env.get_sim_state() - checkpoint_states[-1]))
                )
                if replay_max_abs > 1e-7:
                    raise RuntimeError(
                        "resume replay did not reconstruct the last checkpoint: "
                        f"max_abs={replay_max_abs}"
                    )
            policy_steps = checkpoint_steps[-1] if checkpoint_steps else 0
        else:
            for _ in range(dummy_steps):
                action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
                obs, _, done, _ = env.step(action)
                prefix_actions.append(action)
                if done:
                    break
            policy_steps = 0

        while not done and policy_steps <= args.max_policy_steps:
            needs_checkpoint = not checkpoint_steps or checkpoint_steps[-1] != policy_steps
            if needs_checkpoint:
                checkpoint_index = len(checkpoint_steps)
                checkpoint_steps.append(policy_steps)
                checkpoint_action_offsets.append(len(prefix_actions))
                checkpoint_states.append(env.get_sim_state().copy())
                for key in PROPRIO_KEYS:
                    checkpoint_proprio[key].append(np.asarray(obs[key]).copy())
                progress_records.append(
                    score_task_progress(env, obs, suite=suite_name, task_id=task_id)
                )
                if args.image_stride > 0 and checkpoint_index % args.image_stride == 0:
                    images = get_libero_image(obs)
                    save_jpeg(
                        image_dir / f"state{checkpoint_index:04d}_agentview.jpg",
                        images["image"],
                        args.jpeg_quality,
                    )
                    save_jpeg(
                        image_dir / f"state{checkpoint_index:04d}_wrist.jpg",
                        images["wrist_image"],
                        args.jpeg_quality,
                    )

                atomic_npz(
                    trajectory_path,
                    initial_state=initial_state,
                    prefix_actions=np.asarray(prefix_actions, dtype=np.float32),
                    checkpoint_policy_steps=np.asarray(checkpoint_steps, dtype=np.int64),
                    checkpoint_action_offsets=np.asarray(
                        checkpoint_action_offsets, dtype=np.int64
                    ),
                    checkpoint_states=np.stack(checkpoint_states),
                    **{
                        f"checkpoint_{key}": np.stack(values)
                        for key, values in checkpoint_proprio.items()
                    },
                )
                atomic_json(progress_path, progress_records)
                used_bytes = check_budget(args.output_dir, max_cache_bytes)
                print(
                    json.dumps(
                        {
                            "checkpoint_index": checkpoint_index,
                            "policy_step": policy_steps,
                            "progress": (
                                None
                                if progress_records[-1] is None
                                else progress_records[-1]["score"]
                            ),
                            "cache_bytes": used_bytes,
                        }
                    ),
                    flush=True,
                )

            if done or policy_steps >= args.max_policy_steps:
                break
            if len(checkpoint_steps) >= args.max_states:
                break

            chunk = _predict(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                seed=env_seed,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            execute = min(replan_steps, args.max_policy_steps - policy_steps)
            for action in chunk[:execute]:
                obs, _, done, _ = env.step(action)
                prefix_actions.append(np.asarray(action, dtype=np.float32).copy())
                policy_steps += 1
                if done:
                    break

        if not checkpoint_steps or checkpoint_steps[-1] != policy_steps:
            checkpoint_steps.append(policy_steps)
            checkpoint_action_offsets.append(len(prefix_actions))
            checkpoint_states.append(env.get_sim_state().copy())
            for key in PROPRIO_KEYS:
                checkpoint_proprio[key].append(np.asarray(obs[key]).copy())
            progress_records.append(
                score_task_progress(env, obs, suite=suite_name, task_id=task_id)
            )
            atomic_npz(
                trajectory_path,
                initial_state=initial_state,
                prefix_actions=np.asarray(prefix_actions, dtype=np.float32),
                checkpoint_policy_steps=np.asarray(checkpoint_steps, dtype=np.int64),
                checkpoint_action_offsets=np.asarray(checkpoint_action_offsets, dtype=np.int64),
                checkpoint_states=np.stack(checkpoint_states),
                **{
                    f"checkpoint_{key}": np.stack(values)
                    for key, values in checkpoint_proprio.items()
                },
            )
            atomic_json(progress_path, progress_records)
    finally:
        env.close()

    used_bytes = check_budget(args.output_dir, max_cache_bytes)
    metadata = {
        "schema_version": 1,
        "suite": suite_name,
        "task_id": task_id,
        "task": task_description,
        "trial_id": args.trial_id,
        "env_seed": env_seed,
        "dummy_steps": dummy_steps,
        "replan_steps": replan_steps,
        "max_policy_steps": args.max_policy_steps,
        "max_states": args.max_states,
        "max_cache_mb": args.max_cache_mb,
        "image_stride": args.image_stride,
        "jpeg_quality": args.jpeg_quality,
        "num_checkpoints": len(checkpoint_steps),
        "final_policy_step": checkpoint_steps[-1],
        "success": bool(done),
        "stopped_by_state_limit": (
            len(checkpoint_steps) >= args.max_states
            and checkpoint_steps[-1] < args.max_policy_steps
            and not done
        ),
        "cache_bytes": used_bytes,
        "checkpoint": str(Path(str(cfg.ckpt)).resolve()),
        "dataset_stats": str(dataset_stats_path),
        "model_device": model_device,
        "duration_seconds": time.time() - started,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    atomic_json(metadata_path, metadata)
    print(json.dumps({**metadata, "resolved_config": "stored in metadata"}, indent=2))


if __name__ == "__main__":
    main()
