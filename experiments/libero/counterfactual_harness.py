#!/usr/bin/env python3
"""Evaluate action candidates with isolated LIBERO prefix replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROPRIO_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)

from experiments.libero.task_progress import score_task_progress


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_actions(name: str, actions: np.ndarray, ndim: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != ndim or actions.shape[-1] != 7:
        expected = "[T, 7]" if ndim == 2 else "[K, H, 7]"
        raise ValueError(f"{name} must have shape {expected}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError(f"{name} contains non-finite values")
    return actions


def candidate_diversity(candidates: np.ndarray) -> dict[str, Any]:
    candidates = _validate_actions("candidate_actions", candidates, ndim=3)
    flat = candidates.reshape(candidates.shape[0], -1)
    pairwise = []
    for left in range(len(flat)):
        for right in range(left + 1, len(flat)):
            pairwise.append(float(np.linalg.norm(flat[left] - flat[right])))
    return {
        "num_candidates": int(candidates.shape[0]),
        "horizon": int(candidates.shape[1]),
        "mean_pairwise_l2": float(np.mean(pairwise)) if pairwise else 0.0,
        "min_pairwise_l2": float(np.min(pairwise)) if pairwise else 0.0,
        "per_action_dim_variance": np.var(candidates, axis=(0, 1)).tolist(),
    }


def summarize_headroom(successes: list[bool]) -> dict[str, Any]:
    if not successes:
        raise ValueError("at least one candidate outcome is required")
    values = np.asarray(successes, dtype=np.float64)
    first = float(values[0])
    oracle = float(np.max(values))
    return {
        "informative": bool(np.any(values == 0.0) and np.any(values == 1.0)),
        "num_successes": int(values.sum()),
        "random_candidate_success": float(values.mean()),
        "first_candidate_success": first,
        "simulator_oracle_success": oracle,
        "oracle_uplift_over_random_percentage_points": float(100.0 * (oracle - values.mean())),
        "oracle_uplift_over_first_percentage_points": float(100.0 * (oracle - first)),
    }


def summarize_progress(scores: list[float]) -> dict[str, Any]:
    if not scores:
        raise ValueError("at least one progress score is required")
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("progress scores must be finite")
    first = float(values[0])
    oracle = float(np.max(values))
    return {
        "mean_candidate_progress": float(values.mean()),
        "first_candidate_progress": first,
        "simulator_oracle_progress": oracle,
        "progress_range": float(np.ptp(values)),
        "progress_std": float(values.std()),
        "oracle_uplift_over_mean": float(oracle - values.mean()),
        "oracle_uplift_over_first": float(oracle - first),
    }


def _proprio(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(obs[key]).copy() for key in PROPRIO_KEYS}


def evaluate_candidate(
    *,
    task: Any,
    initial_state: np.ndarray,
    env_seed: int,
    resolution: int,
    prefix_actions: np.ndarray,
    candidate_actions: np.ndarray,
    suite: str,
    task_id: int,
) -> dict[str, Any]:
    """Reconstruct a branch in a fresh environment, then execute one candidate."""
    from experiments.libero.libero_utils import get_libero_env

    np.random.seed(env_seed)
    env, task_description = get_libero_env(task, resolution, env_seed, env_num=1)
    try:
        env.reset()
        obs = env.set_init_state(initial_state.copy())
        for step, action in enumerate(prefix_actions):
            obs, _, done, _ = env.step(action)
            if done:
                raise RuntimeError(f"episode ended while replaying prefix at step {step}")

        branch_state = env.get_sim_state().copy()
        branch_proprio = _proprio(obs)
        done = False
        executed = 0
        for action in candidate_actions:
            obs, _, done, _ = env.step(action)
            executed += 1
            if done:
                break

        success = bool(env.check_success())
        progress = score_task_progress(env, obs, suite=suite, task_id=task_id)
        if success:
            failure_reason = None
        elif done:
            failure_reason = "environment_done_without_success"
        else:
            failure_reason = "candidate_horizon_exhausted"
        return {
            "task_description": task_description,
            "branch_state": branch_state,
            "branch_proprio": branch_proprio,
            "terminal_state": env.get_sim_state().copy(),
            "terminal_proprio": _proprio(obs),
            "success": success,
            "done": bool(done),
            "executed_steps": executed,
            "failure_reason": failure_reason,
            "progress": progress,
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Input NPZ keys: initial_state [S], prefix_actions [T,7], "
            "candidate_actions [K,H,7], and optional candidate_seeds [K]."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--atol", type=float, default=1e-7)
    return parser.parse_args()


def main() -> None:
    from libero.libero import benchmark

    args = parse_args()
    started = time.time()
    with np.load(args.input, allow_pickle=False) as bundle:
        initial_state = np.asarray(bundle["initial_state"], dtype=np.float64)
        prefix_actions = _validate_actions("prefix_actions", bundle["prefix_actions"], ndim=2)
        candidates = _validate_actions("candidate_actions", bundle["candidate_actions"], ndim=3)
        candidate_seeds = (
            np.asarray(bundle["candidate_seeds"], dtype=np.int64)
            if "candidate_seeds" in bundle.files
            else np.full(len(candidates), -1, dtype=np.int64)
        )
    if len(candidate_seeds) != len(candidates):
        raise ValueError("candidate_seeds length must match candidate_actions")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    benchmark_initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.trial_id < len(benchmark_initial_states):
        raise ValueError(f"trial-id {args.trial_id} is outside the benchmark initial-state range")
    expected_initial_state = np.asarray(benchmark_initial_states[args.trial_id])
    initial_state_max_abs = float(np.max(np.abs(initial_state - expected_initial_state)))
    if initial_state_max_abs > args.atol:
        raise ValueError(
            f"input initial_state does not match suite trial {args.trial_id}: "
            f"max_abs={initial_state_max_abs} > atol={args.atol}"
        )

    outcomes = [
        evaluate_candidate(
            task=task,
            initial_state=initial_state,
            env_seed=args.env_seed,
            resolution=args.resolution,
            prefix_actions=prefix_actions,
            candidate_actions=candidate,
            suite=args.suite,
            task_id=args.task_id,
        )
        for candidate in candidates
    ]

    reference_state = outcomes[0]["branch_state"]
    branch_state_max_abs = [
        float(np.max(np.abs(reference_state - outcome["branch_state"]))) for outcome in outcomes
    ]
    reference_proprio = outcomes[0]["branch_proprio"]
    branch_proprio_max_abs = [
        max(
            float(np.max(np.abs(reference_proprio[key] - outcome["branch_proprio"][key])))
            for key in PROPRIO_KEYS
        )
        for outcome in outcomes
    ]
    replay_passed = (
        max(branch_state_max_abs, default=0.0) <= args.atol
        and max(branch_proprio_max_abs, default=0.0) <= args.atol
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "outcomes.npz"
    np.savez_compressed(
        artifact_path,
        initial_state=initial_state,
        prefix_actions=prefix_actions,
        candidate_actions=candidates,
        candidate_seeds=candidate_seeds,
        branch_states=np.stack([outcome["branch_state"] for outcome in outcomes]),
        terminal_states=np.stack([outcome["terminal_state"] for outcome in outcomes]),
        progress_scores=np.asarray(
            [
                np.nan if outcome["progress"] is None else outcome["progress"]["score"]
                for outcome in outcomes
            ],
            dtype=np.float64,
        ),
        **{
            f"branch_{key}": np.stack([outcome["branch_proprio"][key] for outcome in outcomes])
            for key in PROPRIO_KEYS
        },
        **{
            f"terminal_{key}": np.stack(
                [outcome["terminal_proprio"][key] for outcome in outcomes]
            )
            for key in PROPRIO_KEYS
        },
    )

    records = []
    for candidate_id, (seed, outcome) in enumerate(zip(candidate_seeds, outcomes)):
        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_seed": None if seed < 0 else int(seed),
                "success": outcome["success"],
                "done": outcome["done"],
                "executed_steps": outcome["executed_steps"],
                "failure_reason": outcome["failure_reason"],
                "progress": (
                    None if outcome["progress"] is None else outcome["progress"]["score"]
                ),
                "progress_metrics": outcome["progress"],
                "artifact_row": candidate_id,
            }
        )

    progress_scores = [
        outcome["progress"]["score"]
        for outcome in outcomes
        if outcome["progress"] is not None
    ]
    payload = {
        "schema_version": 1,
        "strategy": "isolated_prefix_replay",
        "suite": args.suite,
        "task_id": args.task_id,
        "task": outcomes[0]["task_description"],
        "trial_id": args.trial_id,
        "env_seed": args.env_seed,
        "prefix_steps": int(len(prefix_actions)),
        "candidate_horizon": int(candidates.shape[1]),
        "atol": args.atol,
        "initial_state_max_abs": initial_state_max_abs,
        "replay_check": {
            "passed": replay_passed,
            "branch_state_max_abs": branch_state_max_abs,
            "branch_proprio_max_abs": branch_proprio_max_abs,
        },
        "diversity": candidate_diversity(candidates),
        "headroom": summarize_headroom([outcome["success"] for outcome in outcomes]),
        "progress_headroom": (
            summarize_progress(progress_scores) if len(progress_scores) == len(outcomes) else None
        ),
        "candidates": records,
        "artifacts": {"outcomes": artifact_path.name},
        "source_bundle": {
            "path": str(args.input.resolve()),
            "sha256": sha256(args.input),
        },
        "duration_seconds": time.time() - started,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not replay_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
