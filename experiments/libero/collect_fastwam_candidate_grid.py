#!/usr/bin/env python3
"""Collect FastWAM candidates at several replayed points of one saved prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, LIBERO_EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.libero.collect_fastwam_candidates import (
    PROPRIO_KEYS,
    _model_dimensions,
    _predict,
    perturb_in_normalized_space,
)
from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_env,
    get_libero_image,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument(
        "--branch-policy-step", type=int, action="append", required=True
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument(
        "--model-seed", type=int, action="append", default=None
    )
    parser.add_argument("--perturb-std", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--perturb-base-index", type=int, default=0)
    parser.add_argument("--perturb-seed", type=int, default=10000)
    args, hydra_overrides = parser.parse_known_args()
    if args.model_seed is None:
        args.model_seed = [100, 101, 102, 103, 104]
    return args, hydra_overrides


def main() -> None:
    args, hydra_overrides = parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(set(args.branch_policy_step)) != len(args.branch_policy_step):
        raise ValueError("branch-policy-step values must be unique")
    if any(step < 0 for step in args.branch_policy_step):
        raise ValueError("branch-policy-step values cannot be negative")
    if not 0 <= args.perturb_base_index < len(args.model_seed):
        raise ValueError("perturb-base-index must select a configured model seed")

    source_metadata_path = args.source_bundle.with_suffix(".json")
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )
    dummy_steps = int(source_metadata.get("prefix_dummy_steps", 30))
    with np.load(args.source_bundle, allow_pickle=False) as source:
        initial_state = np.asarray(source["initial_state"]).copy()
        full_prefix = np.asarray(source["prefix_actions"], dtype=np.float32).copy()
        source_branch_state = np.asarray(source["branch_state"]).copy()

    required_prefix_lengths = [dummy_steps + step for step in args.branch_policy_step]
    if max(required_prefix_lengths) > len(full_prefix):
        raise ValueError(
            f"source prefix has {len(full_prefix)} actions, but requested "
            f"{max(required_prefix_lengths)}"
        )

    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="sim_libero", overrides=hydra_overrides)
    cfg.EVALUATION.output_dir = str(args.output_root.resolve())
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
    if args.horizon > action_horizon:
        raise ValueError(f"horizon {args.horizon} exceeds model horizon {action_horizon}")

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    if not 0 <= args.trial_id < len(initial_states):
        raise ValueError("trial-id is outside the benchmark initial-state range")
    expected_initial_state = np.asarray(initial_states[args.trial_id])
    initial_state_max_abs = float(np.max(np.abs(initial_state - expected_initial_state)))
    if initial_state_max_abs > 1e-7:
        raise ValueError(
            "source initial_state does not match selected suite/trial: "
            f"max_abs={initial_state_max_abs}"
        )

    env_seed = int(cfg.get("seed", 42))
    summaries: list[dict[str, Any]] = []
    for policy_step, prefix_length in zip(args.branch_policy_step, required_prefix_lengths):
        branch_started = time.time()
        np.random.seed(env_seed)
        env, task_description = get_libero_env(
            task, LIBERO_ENV_RESOLUTION, env_seed, env_num=1
        )
        try:
            env.reset()
            obs = env.set_init_state(initial_state.copy())
            for step, action in enumerate(full_prefix[:prefix_length]):
                obs, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError(
                        f"episode ended while replaying prefix at action {step}"
                    )

            branch_state = env.get_sim_state().copy()
            branch_images = get_libero_image(obs)
            branch_proprio = {
                key: np.asarray(obs[key]).copy() for key in PROPRIO_KEYS
            }
            model_candidates = [
                _predict(
                    obs=obs,
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    seed=seed,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                )[: args.horizon]
                for seed in args.model_seed
            ]
        finally:
            env.close()

        perturb_rng = np.random.default_rng(args.perturb_seed + policy_step)
        perturbed = [
            perturb_in_normalized_space(
                model_candidates[args.perturb_base_index],
                processor,
                std=std,
                rng=perturb_rng,
            )
            for std in args.perturb_std
        ]
        candidates = np.stack(model_candidates + perturbed)
        candidate_seeds = np.asarray(
            args.model_seed + [-1] * len(perturbed), dtype=np.int64
        )
        candidate_sources = [
            f"fastwam_seed:{seed}" for seed in args.model_seed
        ] + [f"normalized_perturbation:{std}" for std in args.perturb_std]

        output_dir = args.output_root / f"step{policy_step:04d}_h{args.horizon}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "candidates.npz"
        np.savez_compressed(
            output_path,
            initial_state=initial_state,
            prefix_actions=full_prefix[:prefix_length],
            candidate_actions=candidates,
            candidate_seeds=candidate_seeds,
            branch_state=branch_state,
            branch_agentview_image=branch_images["image"],
            branch_wrist_image=branch_images["wrist_image"],
            **{f"branch_{key}": value for key, value in branch_proprio.items()},
        )
        source_end_state_max_abs = (
            float(np.max(np.abs(branch_state - source_branch_state)))
            if prefix_length == len(full_prefix)
            else None
        )
        metadata = {
            "schema_version": 1,
            "suite": suite_name,
            "task_id": task_id,
            "task": task_description,
            "trial_id": args.trial_id,
            "env_seed": env_seed,
            "prefix_dummy_steps": dummy_steps,
            "prefix_policy_steps": policy_step,
            "prefix_bundle": str(args.source_bundle.resolve()),
            "prefix_bundle_sha256": sha256(args.source_bundle),
            "source_end_state_max_abs": source_end_state_max_abs,
            "horizon": args.horizon,
            "candidate_sources": candidate_sources,
            "candidate_seeds": [
                None if seed < 0 else int(seed) for seed in candidate_seeds
            ],
            "perturb_base_index": args.perturb_base_index,
            "perturb_base_seed": args.model_seed[args.perturb_base_index],
            "perturb_seed": args.perturb_seed + policy_step,
            "checkpoint": str(Path(str(cfg.ckpt)).resolve()),
            "dataset_stats": str(dataset_stats_path),
            "model_device": model_device,
            "branch_duration_seconds": time.time() - branch_started,
            "elapsed_seconds": time.time() - started,
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(
            {
                "policy_step": policy_step,
                "output": str(output_path),
                "num_candidates": len(candidates),
                "duration_seconds": metadata["branch_duration_seconds"],
            }
        )
        print(json.dumps(summaries[-1]), flush=True)

    print(
        json.dumps(
            {
                "summaries": summaries,
                "total_duration_seconds": time.time() - started,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else 0
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
