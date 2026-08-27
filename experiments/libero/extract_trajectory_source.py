#!/usr/bin/env python3
"""Export an exact replay prefix from a bounded trajectory checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-step", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_path = args.trajectory_dir / "trajectory.npz"
    metadata_path = args.trajectory_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        steps = np.asarray(trajectory["checkpoint_policy_steps"], dtype=np.int64)
        requested = int(steps[-1]) if args.policy_step is None else args.policy_step
        matches = np.flatnonzero(steps == requested)
        if len(matches) != 1:
            raise ValueError(
                f"policy step {requested} is not an exact checkpoint; available={steps.tolist()}"
            )
        index = int(matches[0])
        offset = int(trajectory["checkpoint_action_offsets"][index])
        initial_state = np.asarray(trajectory["initial_state"]).copy()
        prefix_actions = np.asarray(trajectory["prefix_actions"][:offset], dtype=np.float32)
        branch_state = np.asarray(trajectory["checkpoint_states"][index]).copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        initial_state=initial_state,
        prefix_actions=prefix_actions,
        branch_state=branch_state,
    )
    payload = {
        "schema_version": 1,
        "source_trajectory": str(trajectory_path.resolve()),
        "suite": metadata["suite"],
        "task_id": metadata["task_id"],
        "trial_id": metadata["trial_id"],
        "env_seed": metadata["env_seed"],
        "prefix_dummy_steps": metadata["dummy_steps"],
        "prefix_policy_steps": requested,
        "prefix_action_steps": offset,
        "checkpoint_index": index,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
