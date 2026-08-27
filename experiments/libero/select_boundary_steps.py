#!/usr/bin/env python3
"""Select semantic boundary checkpoints before inspecting candidate outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=3)
    return parser.parse_args()


def select_boundaries(steps: list[int], rows: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    if len(steps) != len(rows):
        raise ValueError("checkpoint and progress lengths differ")
    if any(row is None for row in rows):
        raise ValueError("boundary selection requires dense task progress")
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    grasped = np.asarray([bool(row["grasped"]) for row in rows])
    succeeded = np.asarray([bool(row["success"]) for row in rows])
    proposals: list[tuple[str, int]] = []

    grasp_indices = np.flatnonzero(grasped)
    if len(grasp_indices):
        first_grasp = int(grasp_indices[0])
        proposals.append(("pre_grasp", max(0, first_grasp - 1)))
        proposals.append(("first_grasp", first_grasp))
    else:
        nonsuccess_indices = np.flatnonzero(~succeeded)
        if len(nonsuccess_indices) >= 2:
            nonsuccess_scores = scores[nonsuccess_indices]
            jump_offset = int(np.argmax(np.diff(nonsuccess_scores))) + 1
            jump_index = int(nonsuccess_indices[jump_offset])
            proposals.append(("pre_major_progress", max(0, jump_index - 1)))
            proposals.append(("major_progress", jump_index))

    success_indices = np.flatnonzero(succeeded)
    if len(success_indices):
        proposals.append(("pre_completion", max(0, int(success_indices[0]) - 2)))
    elif len(grasp_indices):
        best_transport = int(grasp_indices[np.argmax(scores[grasp_indices])])
        proposals.append(("pre_best_transport", max(0, best_transport - 1)))
    else:
        best = int(np.argmax(scores))
        proposals.append(("pre_best_progress", max(0, best - 1)))
        proposals.append(("best_progress", best))

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for phase, index in proposals:
        step = int(steps[index])
        if step in seen:
            continue
        seen.add(step)
        selected.append(
            {
                "phase": phase,
                "checkpoint_index": index,
                "policy_step": step,
                "score": float(scores[index]),
                "grasped": bool(grasped[index]),
                "success": bool(succeeded[index]),
            }
        )
    return selected


def main() -> None:
    args = parse_args()
    if args.max_groups <= 0:
        raise ValueError("max-groups must be positive")
    with np.load(args.trajectory_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        steps = trajectory["checkpoint_policy_steps"].astype(int).tolist()
    rows = json.loads((args.trajectory_dir / "progress.json").read_text(encoding="utf-8"))
    selected = select_boundaries(steps, rows)[: args.max_groups]
    payload = {
        "schema_version": 1,
        "selection_rule": "semantic_boundary_v2",
        "selected_before_candidate_outcomes": True,
        "trajectory_dir": str(args.trajectory_dir.resolve()),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
