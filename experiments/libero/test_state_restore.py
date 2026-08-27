#!/usr/bin/env python3
"""Check whether a LIBERO branch can be rebuilt by isolated prefix replay."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_env
from libero.libero import benchmark


PROPRIO_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
IMAGE_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
)


def deterministic_actions(seed: int, steps: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    actions = rng.normal(0.0, 0.05, size=(steps, 7)).astype(np.float64)
    actions[:, -1] = -1.0
    return actions


def isolated_rollout(task, initial_state: np.ndarray, args, actions: np.ndarray) -> dict:
    """Rebuild one branch without carrying hidden robosuite state across candidates."""
    # LIBERO seeds the environment after construction, but model construction can
    # consume NumPy randomness (for example visual properties). Seed both phases.
    np.random.seed(args.seed)
    env, description = get_libero_env(task, args.resolution, args.seed, env_num=1)
    try:
        env.reset()
        obs = env.set_init_state(initial_state.copy())
        for _ in range(args.warmup_steps):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                raise RuntimeError("episode ended while replaying the branch prefix")

        branch_state = env.get_sim_state().copy()
        done = False
        executed_steps = 0
        for action in actions:
            obs, _, done, _ = env.step(action)
            executed_steps += 1
            if done:
                break
        return {
            "description": description,
            "branch_state": branch_state,
            "sim_state": env.get_sim_state().copy(),
            "success": bool(env.check_success()),
            "done": bool(done),
            "steps": executed_steps,
            "proprio": {key: np.asarray(obs[key]).copy() for key in PROPRIO_KEYS},
            "images": {key: np.asarray(obs[key]).copy() for key in IMAGE_KEYS},
        }
    finally:
        env.close()


def compare(reference: dict, candidate: dict, atol: float) -> dict:
    proprio_max_abs = {
        key: float(np.max(np.abs(reference["proprio"][key] - candidate["proprio"][key])))
        for key in PROPRIO_KEYS
    }
    image_max_abs = {
        key: int(
            np.max(
                np.abs(
                    reference["images"][key].astype(np.int16)
                    - candidate["images"][key].astype(np.int16)
                )
            )
        )
        for key in IMAGE_KEYS
    }
    image_mean_abs = {
        key: float(
            np.mean(
                np.abs(
                    reference["images"][key].astype(np.int16)
                    - candidate["images"][key].astype(np.int16)
                )
            )
        )
        for key in IMAGE_KEYS
    }
    image_changed_fraction = {
        key: float(np.mean(reference["images"][key] != candidate["images"][key]))
        for key in IMAGE_KEYS
    }
    sim_state_max_abs = float(
        np.max(np.abs(reference["sim_state"] - candidate["sim_state"]))
    )
    branch_state_max_abs = float(
        np.max(np.abs(reference["branch_state"] - candidate["branch_state"]))
    )
    physics_passed = (
        reference["success"] == candidate["success"]
        and reference["done"] == candidate["done"]
        and branch_state_max_abs <= atol
        and sim_state_max_abs <= atol
        and max(proprio_max_abs.values()) <= atol
    )
    return {
        "physics_passed": physics_passed,
        "render_exact": max(image_max_abs.values()) == 0,
        "success_match": reference["success"] == candidate["success"],
        "done_match": reference["done"] == candidate["done"],
        "branch_state_max_abs": branch_state_max_abs,
        "sim_state_max_abs": sim_state_max_abs,
        "proprio_max_abs": proprio_max_abs,
        "image_max_abs": image_max_abs,
        "image_mean_abs": image_mean_abs,
        "image_changed_fraction": image_changed_fraction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.trial_id < len(initial_states):
        raise ValueError(f"trial-id {args.trial_id} outside [0, {len(initial_states)})")

    actions = deterministic_actions(args.seed + 1, args.rollout_steps)
    rollouts = [
        isolated_rollout(task, initial_states[args.trial_id], args, actions)
        for _ in range(args.repeats)
    ]
    comparisons = [compare(rollouts[0], item, args.atol) for item in rollouts[1:]]
    payload = {
        "strategy": "isolated_prefix_replay",
        "suite": args.suite,
        "task_id": args.task_id,
        "task": rollouts[0]["description"],
        "trial_id": args.trial_id,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "rollout_steps": args.rollout_steps,
        "repeats": args.repeats,
        "atol": args.atol,
        "physics_passed": all(item["physics_passed"] for item in comparisons),
        "render_exact": all(item["render_exact"] for item in comparisons),
        "comparisons": comparisons,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["physics_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
