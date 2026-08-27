#!/usr/bin/env python3
"""Probe whether LeRobot LIBERO demonstrations replay from matching init states."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from libero.libero import benchmark

from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-success", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_path = args.dataset / "data" / "chunk-000" / f"episode_{args.episode_id:06d}.parquet"
    table = pq.read_table(table_path, columns=["action", "task_index"])
    task_ids = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
    if not np.all(task_ids == args.task_id):
        raise ValueError(f"episode task ids {np.unique(task_ids)} do not match {args.task_id}")
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    actions[:, -1] = 1.0 - 2.0 * actions[:, -1]

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    trial_id = args.episode_id % len(initial_states)

    np.random.seed(args.seed)
    env, description = get_libero_env(task, 256, args.seed, env_num=1)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[trial_id].copy())
        for _ in range(args.warmup):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                break
        executed = 0
        for action in actions:
            obs, _, done, _ = env.step(action)
            executed += 1
            if done:
                break
        result = {
            "task": description,
            "episode_id": args.episode_id,
            "trial_id": trial_id,
            "warmup": args.warmup,
            "action_steps": len(actions),
            "executed": executed,
            "success": bool(env.check_success()),
            "done": bool(done),
            "mapping_assumption": "episode_id_mod_num_initial_states",
        }
        print(json.dumps(result, indent=2))
        if args.require_success and not result["success"]:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
