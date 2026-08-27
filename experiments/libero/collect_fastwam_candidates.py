#!/usr/bin/env python3
"""Collect one reproducible FastWAM candidate bundle for the LIBERO harness."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from libero.libero import benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, LIBERO_EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _predict_action_chunk,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed


PROPRIO_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--prefix-policy-steps", type=int, default=0)
    parser.add_argument(
        "--prefix-bundle",
        type=Path,
        default=None,
        help="Replay initial_state/prefix_actions from an existing candidate bundle.",
    )
    parser.add_argument(
        "--include-source-candidates",
        action="store_true",
        help="Append candidates from --prefix-bundle after newly generated candidates.",
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument(
        "--model-seed",
        dest="model_seed_overrides",
        type=int,
        action="append",
        default=None,
        help="Repeatable alternative to --model-seeds that is safe to mix with Hydra overrides.",
    )
    parser.add_argument("--perturb-std", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--perturb-base-index", type=int, default=0)
    parser.add_argument("--perturb-seed", type=int, default=10000)
    args, hydra_overrides = parser.parse_known_args()
    if args.model_seed_overrides is not None:
        args.model_seeds = args.model_seed_overrides
    del args.model_seed_overrides
    return args, hydra_overrides


def _env_to_raw_action(actions: np.ndarray) -> np.ndarray:
    raw = np.asarray(actions, dtype=np.float32).copy()
    raw[..., -1] = (1.0 - raw[..., -1]) / 2.0
    return raw


def _raw_to_env_action(actions: np.ndarray) -> np.ndarray:
    env = np.asarray(actions, dtype=np.float32).copy()
    env[..., -1] = np.where(env[..., -1] > 0.5, -1.0, 1.0)
    return env


def perturb_in_normalized_space(
    actions: np.ndarray,
    processor: FastWAMProcessor,
    *,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb continuous dimensions after applying the training normalizer."""
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("candidate collection expects one merged action key")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]

    raw = torch.as_tensor(_env_to_raw_action(actions)).unsqueeze(0)
    normalized = normalizer.forward(raw).squeeze(0)
    noise = torch.as_tensor(
        rng.normal(0.0, std, size=normalized[..., :-1].shape), dtype=normalized.dtype
    )
    perturbed = normalized.clone()
    perturbed[..., :-1] = torch.clamp(perturbed[..., :-1] + noise, -1.0, 1.0)
    denormalized = normalizer.backward(perturbed.unsqueeze(0)).squeeze(0).numpy()
    return _raw_to_env_action(denormalized)


def _model_dimensions(cfg: Any) -> tuple[int, int, int]:
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    video_size = cfg.data.train.video_size
    return action_horizon, int(video_size[1]), int(video_size[0])


def _predict(
    *,
    obs: dict[str, Any],
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: Any,
    seed: int,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> np.ndarray:
    previous_seed = cfg.get("seed")
    cfg.seed = int(seed)
    try:
        actions, _, _ = _predict_action_chunk(
            obs=obs,
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
    finally:
        cfg.seed = previous_seed
    return np.asarray(actions, dtype=np.float32)


def main() -> None:
    args, hydra_overrides = parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if args.prefix_policy_steps < 0:
        raise ValueError("prefix-policy-steps cannot be negative")
    if args.include_source_candidates and args.prefix_bundle is None:
        raise ValueError("--include-source-candidates requires --prefix-bundle")
    if args.perturb_base_index < 0 or args.perturb_base_index >= len(args.model_seeds):
        raise ValueError(
            "perturb-base-index must select one of the configured model seeds, "
            f"got {args.perturb_base_index} for {len(args.model_seeds)} seeds"
        )

    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="sim_libero", overrides=hydra_overrides)
    # compose() does not initialize HydraConfig, so resolve its runtime-only
    # output interpolation to this bundle's explicit artifact directory.
    cfg.EVALUATION.output_dir = str(args.output.parent.resolve())
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
        raise ValueError(f"horizon {args.horizon} exceeds model action horizon {action_horizon}")

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    if not 0 <= args.trial_id < len(initial_states):
        raise ValueError(f"trial-id {args.trial_id} is outside the benchmark initial-state range")
    benchmark_initial_state = np.asarray(initial_states[args.trial_id]).copy()
    source_candidates = None
    source_candidate_seeds = None
    source_branch_state = None
    source_metadata: dict[str, Any] = {}
    if args.prefix_bundle is None:
        initial_state = benchmark_initial_state
        replay_prefix = None
    else:
        with np.load(args.prefix_bundle, allow_pickle=False) as source:
            initial_state = np.asarray(source["initial_state"]).copy()
            replay_prefix = np.asarray(source["prefix_actions"], dtype=np.float32).copy()
            source_branch_state = np.asarray(source["branch_state"]).copy()
            if args.include_source_candidates:
                source_candidates = np.asarray(source["candidate_actions"], dtype=np.float32).copy()
                source_candidate_seeds = (
                    np.asarray(source["candidate_seeds"], dtype=np.int64).copy()
                    if "candidate_seeds" in source.files
                    else np.full(len(source_candidates), -1, dtype=np.int64)
                )
        initial_state_max_abs = float(np.max(np.abs(initial_state - benchmark_initial_state)))
        if initial_state_max_abs > 1e-7:
            raise ValueError(
                "prefix bundle initial_state does not match the selected suite/trial: "
                f"max_abs={initial_state_max_abs}"
            )
        source_metadata_path = args.prefix_bundle.with_suffix(".json")
        if source_metadata_path.exists():
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    env_seed = int(cfg.get("seed", 42))

    np.random.seed(env_seed)
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, env_seed, env_num=1)
    prefix_actions: list[np.ndarray] = []
    try:
        env.reset()
        obs = env.set_init_state(initial_state.copy())
        if replay_prefix is not None:
            for step, action in enumerate(replay_prefix):
                obs, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError(f"episode ended while replaying source prefix at step {step}")
                prefix_actions.append(action.copy())
        else:
            for _ in range(int(cfg.EVALUATION.num_steps_wait)):
                action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
                obs, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError("episode ended during the dummy-action prefix")
                prefix_actions.append(action)

            policy_steps = 0
            replan_steps = int(cfg.EVALUATION.replan_steps)
            base_seed = int(cfg.get("seed", 42))
            while policy_steps < args.prefix_policy_steps:
                chunk = _predict(
                    obs=obs,
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    seed=base_seed,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                )
                execute = min(replan_steps, args.prefix_policy_steps - policy_steps)
                for action in chunk[:execute]:
                    obs, _, done, _ = env.step(action)
                    if done:
                        raise RuntimeError("episode ended before the requested branch point")
                    prefix_actions.append(action.copy())
                    policy_steps += 1

        branch_state = env.get_sim_state().copy()
        if source_branch_state is not None:
            branch_state_max_abs = float(np.max(np.abs(branch_state - source_branch_state)))
            if branch_state_max_abs > 1e-7:
                raise RuntimeError(
                    "source prefix replay did not reconstruct the saved branch state: "
                    f"max_abs={branch_state_max_abs}"
                )
        else:
            branch_state_max_abs = None
        branch_images = get_libero_image(obs)
        branch_proprio = {key: np.asarray(obs[key]).copy() for key in PROPRIO_KEYS}

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
            for seed in args.model_seeds
        ]
    finally:
        env.close()

    perturb_rng = np.random.default_rng(args.perturb_seed)
    perturbed = [
        perturb_in_normalized_space(
            model_candidates[args.perturb_base_index], processor, std=std, rng=perturb_rng
        )
        for std in args.perturb_std
    ]
    candidate_list = model_candidates + perturbed
    candidate_seed_list = args.model_seeds + [-1] * len(perturbed)
    candidate_sources = [f"fastwam_seed:{seed}" for seed in args.model_seeds] + [
        f"normalized_perturbation:{std}" for std in args.perturb_std
    ]
    if source_candidates is not None:
        if source_candidates.ndim != 3 or source_candidates.shape[1:] != (args.horizon, 7):
            raise ValueError(
                "source candidate_actions must match the requested [H,7], got "
                f"{source_candidates.shape} for horizon={args.horizon}"
            )
        candidate_list.extend(source_candidates)
        candidate_seed_list.extend(source_candidate_seeds.tolist())
        source_names = source_metadata.get("candidate_sources", [])
        candidate_sources.extend(
            f"source_bundle:{source_names[index] if index < len(source_names) else index}"
            for index in range(len(source_candidates))
        )
    candidates = np.stack(candidate_list)
    candidate_seeds = np.asarray(candidate_seed_list, dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        initial_state=initial_state,
        prefix_actions=np.asarray(prefix_actions, dtype=np.float32),
        candidate_actions=candidates,
        candidate_seeds=candidate_seeds,
        branch_state=branch_state,
        branch_agentview_image=branch_images["image"],
        branch_wrist_image=branch_images["wrist_image"],
        **{f"branch_{key}": value for key, value in branch_proprio.items()},
    )
    metadata = {
        "schema_version": 1,
        "suite": suite_name,
        "task_id": task_id,
        "task": task_description,
        "trial_id": args.trial_id,
        "env_seed": env_seed,
        "prefix_dummy_steps": int(
            source_metadata.get("prefix_dummy_steps", cfg.EVALUATION.num_steps_wait)
        ),
        "prefix_policy_steps": int(
            source_metadata.get("prefix_policy_steps", args.prefix_policy_steps)
        ),
        "prefix_bundle": None if args.prefix_bundle is None else str(args.prefix_bundle.resolve()),
        "source_branch_state_replay_max_abs": branch_state_max_abs,
        "horizon": args.horizon,
        "candidate_sources": candidate_sources,
        "candidate_seeds": [None if seed < 0 else int(seed) for seed in candidate_seeds],
        "perturb_base_index": args.perturb_base_index,
        "perturb_base_seed": args.model_seeds[args.perturb_base_index],
        "perturb_seed": args.perturb_seed,
        "checkpoint": str(Path(str(cfg.ckpt)).resolve()),
        "dataset_stats": str(dataset_stats_path),
        "model_device": model_device,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "duration_seconds": time.time() - started,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metadata, "resolved_config": "stored in metadata"}, indent=2))


if __name__ == "__main__":
    main()
