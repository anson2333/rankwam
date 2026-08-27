#!/usr/bin/env python3
"""Evaluate a whole candidate group with one FastWAM model load."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from libero.libero import benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.collect_fastwam_candidates import PROPRIO_KEYS, _model_dimensions, _predict
from experiments.libero.continuation_targets import summarize_continuation
from experiments.libero.counterfactual_harness import candidate_diversity, summarize_headroom
from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env, get_libero_image
from experiments.libero.task_progress import score_task_progress
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--max-continuation-steps", type=int, default=60)
    parser.add_argument("--terminal-policy-step", type=int, default=None)
    parser.add_argument("--save-future-observations", action="store_true")
    parser.add_argument("--max-cache-mb", type=float, default=50.0)
    args, overrides = parser.parse_known_args()
    if args.max_continuation_steps <= 0 or args.max_cache_mb <= 0:
        raise ValueError("continuation steps and cache budget must be positive")
    if args.terminal_policy_step is not None and args.terminal_policy_step <= 0:
        raise ValueError("terminal-policy-step must be positive")
    return args, overrides


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def progress_record(step: int, env: Any, obs: dict[str, Any], suite: str, task_id: int) -> dict[str, Any]:
    return {
        "continuation_step": step,
        "progress": score_task_progress(env, obs, suite=suite, task_id=task_id),
    }


def main() -> None:
    args, overrides = parse_args()
    with np.load(args.input, allow_pickle=False) as bundle:
        initial_state = np.asarray(bundle["initial_state"]).copy()
        prefix_actions = np.asarray(bundle["prefix_actions"], dtype=np.float32).copy()
        candidates = np.asarray(bundle["candidate_actions"], dtype=np.float32).copy()
        saved_branch_state = (
            np.asarray(bundle["branch_state"]).copy() if "branch_state" in bundle.files else None
        )
    if candidates.ndim != 3 or candidates.shape[-1] != 7:
        raise ValueError(f"candidate actions must be [K,H,7], got {candidates.shape}")

    metadata_path = args.input.with_suffix(".json")
    candidate_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    branch_policy_step = candidate_metadata.get("prefix_policy_steps")
    continuation_horizon = args.max_continuation_steps
    if args.terminal_policy_step is not None:
        if branch_policy_step is None:
            raise ValueError("terminal-policy-step requires candidate prefix_policy_steps metadata")
        continuation_horizon = (
            args.terminal_policy_step - int(branch_policy_step) - int(candidates.shape[1])
        )
        if continuation_horizon <= 0:
            raise ValueError(
                "terminal policy step leaves no continuation budget: "
                f"T={args.terminal_policy_step}, branch={branch_policy_step}, "
                f"candidate_horizon={candidates.shape[1]}"
            )
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="sim_libero", overrides=overrides)
    if cfg.ckpt is None:
        raise ValueError("pass ckpt=/path/to/checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env_seed = int(cfg.get("seed", 42))
    set_global_seed(env_seed, get_worker_init_fn=False)
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    stats_path = _resolve_dataset_stats_path(cfg)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    action_horizon, input_w, input_h = _model_dimensions(cfg)

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    expected = np.asarray(suite.get_task_init_states(task_id)[args.trial_id])
    if float(np.max(np.abs(initial_state - expected))) > 1e-7:
        raise ValueError("candidate initial state does not match selected trial")
    replan_steps = int(cfg.EVALUATION.replan_steps)
    partial_path = args.output_dir / "partial_results.json"
    records: list[dict[str, Any]] = []
    if partial_path.exists():
        loaded = json.loads(partial_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"partial results must be a list: {partial_path}")
        candidate_ids = [row.get("candidate_id") for row in loaded]
        expected_ids = list(range(len(loaded)))
        if candidate_ids != expected_ids or len(loaded) > len(candidates):
            raise ValueError(
                "partial results are not a contiguous prefix of this candidate group: "
                f"got {candidate_ids}, expected {expected_ids}"
            )
        records = loaded
        print(json.dumps({"resuming_after_candidates": candidate_ids}), flush=True)
    max_cache_bytes = int(args.max_cache_mb * 1024 * 1024)

    for candidate_id, candidate_actions in enumerate(candidates):
        if candidate_id < len(records):
            continue
        candidate_started = time.time()
        np.random.seed(env_seed)
        env, task_description = get_libero_env(
            task, LIBERO_ENV_RESOLUTION, env_seed, env_num=1
        )
        continuation_step = 0
        done = False
        observations: list[dict[str, np.ndarray]] = []
        proprios: dict[str, list[np.ndarray]] = {key: [] for key in PROPRIO_KEYS}
        try:
            env.reset()
            obs = env.set_init_state(initial_state.copy())
            for action in prefix_actions:
                obs, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError("episode ended while replaying prefix")
            replay_error = (
                0.0
                if saved_branch_state is None
                else float(np.max(np.abs(env.get_sim_state() - saved_branch_state)))
            )
            if replay_error > 1e-7:
                raise RuntimeError(f"branch replay mismatch: {replay_error}")
            for action in candidate_actions:
                obs, _, done, _ = env.step(action)
                if done:
                    break

            progress = [progress_record(0, env, obs, suite_name, task_id)]
            if args.save_future_observations:
                observations.append(get_libero_image(obs))
                for key in PROPRIO_KEYS:
                    proprios[key].append(np.asarray(obs[key]).copy())

            while not done and continuation_step < continuation_horizon:
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
                execute = min(replan_steps, continuation_horizon - continuation_step)
                for action in chunk[:execute]:
                    obs, _, done, _ = env.step(action)
                    continuation_step += 1
                    if done:
                        break
                progress.append(progress_record(continuation_step, env, obs, suite_name, task_id))
                if args.save_future_observations:
                    observations.append(get_libero_image(obs))
                    for key in PROPRIO_KEYS:
                        proprios[key].append(np.asarray(obs[key]).copy())

            success = bool(env.check_success())
            terminal_state = env.get_sim_state().copy()
        finally:
            env.close()

        if args.save_future_observations:
            atomic_npz(
                args.output_dir / f"candidate_{candidate_id:02d}_future.npz",
                continuation_steps=np.asarray(
                    [row["continuation_step"] for row in progress], dtype=np.int64
                ),
                agentview=np.stack([row["image"] for row in observations]),
                wrist=np.stack([row["wrist_image"] for row in observations]),
                **{key: np.stack(value) for key, value in proprios.items()},
            )
        record = {
            "candidate_id": candidate_id,
            "source": candidate_metadata.get("candidate_sources", [None] * len(candidates))[candidate_id],
            "success": success,
            "done": bool(done),
            "candidate_steps": int(len(candidate_actions)),
            "continuation_steps": continuation_step,
            "branch_replay_max_abs": replay_error,
            "progress": progress,
            "targets": summarize_continuation(progress),
            "terminal_state": terminal_state.tolist(),
            "duration_seconds": time.time() - candidate_started,
        }
        records.append(record)
        atomic_json(partial_path, records)
        used = directory_size(args.output_dir)
        if used > max_cache_bytes:
            raise RuntimeError(f"candidate cache budget exceeded: {used} > {max_cache_bytes}")
        print(json.dumps({"candidate": candidate_id, "success": success, "seconds": record["duration_seconds"]}), flush=True)

    headroom = summarize_headroom([row["success"] for row in records])
    summary = {
        "schema_version": 1,
        "suite": suite_name,
        "task_id": task_id,
        "trial_id": args.trial_id,
        "branch_policy_step": branch_policy_step,
        "candidate_horizon": int(candidates.shape[1]),
        "continuation_horizon": continuation_horizon,
        "terminal_policy_step": args.terminal_policy_step,
        "behavior_policy_checkpoint": str(Path(str(cfg.ckpt)).resolve()),
        "behavior_policy_sha256": sha256(Path(str(cfg.ckpt))),
        "candidate_diversity": candidate_diversity(candidates),
        "headroom": headroom,
        "records": records,
        "cache_bytes": directory_size(args.output_dir),
        "duration_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"headroom": headroom, "duration_seconds": summary["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
