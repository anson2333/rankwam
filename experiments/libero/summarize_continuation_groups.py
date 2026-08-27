#!/usr/bin/env python3
"""Build an auditable index of completed continuation candidate groups."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LEGACY_GROUP = re.compile(r"task(?P<task>\d+)_trial(?P<trial>\d+)_step(?P<step>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-mixed-groups", type=int, default=20)
    return parser.parse_args()


def group_record(
    *,
    suite: str,
    task_id: int,
    trial_id: int,
    policy_step: int,
    successes: int,
    candidates: int,
    source: Path,
    target_protocol: str,
    cache_bytes: int | None = None,
) -> dict[str, Any]:
    mixed = 0 < successes < candidates
    return {
        "suite": suite,
        "task_id": task_id,
        "trial_id": trial_id,
        "policy_step": policy_step,
        "successes": successes,
        "candidates": candidates,
        "success_rate": successes / candidates,
        "mixed": mixed,
        "strong_contrast": 2 <= successes <= candidates - 2,
        "target_protocol": target_protocol,
        "cache_bytes": cache_bytes,
        "source": str(source.resolve()),
    }


def load_legacy(root: Path) -> list[dict[str, Any]]:
    path = root / "continuation_pilot_summary_v1.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in payload.get("groups", []):
        match = LEGACY_GROUP.fullmatch(row["group"])
        if match is None:
            raise ValueError(f"unrecognized legacy group id: {row['group']}")
        records.append(
            group_record(
                suite="libero_object",
                task_id=int(match.group("task")),
                trial_id=int(match.group("trial")),
                policy_step=int(match.group("step")),
                successes=int(row["successes"]),
                candidates=int(row["candidates"]),
                source=path,
                target_protocol="fixed_remaining_horizon_b60",
            )
        )
    return records


def load_modern(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("trajectory_task*_trial*_v*/continuation_step*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        successes = int(payload["headroom"]["num_successes"])
        terminal_policy_step = payload.get("terminal_policy_step")
        target_protocol = (
            f"fixed_terminal_policy_step_t{terminal_policy_step}"
            if terminal_policy_step is not None
            else f"fixed_remaining_horizon_b{payload['continuation_horizon']}"
        )
        records.append(
            group_record(
                suite=str(payload["suite"]),
                task_id=int(payload["task_id"]),
                trial_id=int(payload["trial_id"]),
                policy_step=int(payload["branch_policy_step"]),
                successes=successes,
                candidates=len(payload["records"]),
                source=path,
                target_protocol=target_protocol,
                cache_bytes=int(payload["cache_bytes"]),
            )
        )
    return records


def main() -> None:
    args = parse_args()
    if args.minimum_mixed_groups <= 0:
        raise ValueError("minimum-mixed-groups must be positive")
    records = load_legacy(args.root) + load_modern(args.root)
    keys = [(r["suite"], r["task_id"], r["trial_id"], r["policy_step"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate suite/task/trial/step groups found")

    mixed = [row for row in records if row["mixed"]]
    strong = [row for row in records if row["strong_contrast"]]
    primary = [
        row for row in records
        if row["target_protocol"] == "fixed_terminal_policy_step_t200"
    ]
    primary_mixed = [row for row in primary if row["mixed"]]
    primary_strong = [row for row in primary if row["strong_contrast"]]
    trials = sorted({(row["suite"], row["task_id"], row["trial_id"]) for row in records})
    payload = {
        "schema_version": 1,
        "definitions": {
            "mixed": "at least one successful and one failed candidate",
            "strong_contrast": "at least two successful and two failed candidates",
        },
        "counts": {
            "groups": len(records),
            "mixed_groups": len(mixed),
            "strong_contrast_groups": len(strong),
            "distinct_trials": len(trials),
            "candidates": sum(row["candidates"] for row in records),
            "primary_t200_groups": len(primary),
            "primary_t200_mixed_groups": len(primary_mixed),
            "primary_t200_strong_contrast_groups": len(primary_strong),
            "primary_t200_candidates": sum(row["candidates"] for row in primary),
        },
        "gate": {
            "minimum_mixed_groups": args.minimum_mixed_groups,
            "protocol": "fixed_terminal_policy_step_t200",
            "critic_training_ready": len(primary_mixed) >= args.minimum_mixed_groups,
        },
        "groups": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
