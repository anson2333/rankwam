#!/usr/bin/env python3
"""Validate and summarize one-trial LIBERO evaluation shards."""

import argparse
import json
import math
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-trial", type=int, default=0)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_ids = set(range(args.start_trial, args.start_trial + args.expected_trials))
    records = []
    for path in sorted(args.root.glob("trial_*/libero_object/gpu*_task*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        trial_start = int(payload.get("trial_start", -1))
        if trial_start not in expected_ids:
            continue
        if int(payload["total_episodes"]) != 1:
            raise ValueError(f"{path} is not a one-trial shard")
        if any(record["trial_start"] == trial_start for record in records):
            raise ValueError(f"duplicate result for trial {trial_start}")
        records.append(
            {
                "trial_start": trial_start,
                "successes": int(payload["successes"]),
                "duration_seconds": float(payload["duration"]),
                "path": str(path.resolve()),
            }
        )

    found_ids = {record["trial_start"] for record in records}
    missing = sorted(expected_ids - found_ids)
    if missing:
        raise ValueError(f"missing trial shards: {missing}")
    records.sort(key=lambda item: item["trial_start"])
    successes = sum(record["successes"] for record in records)
    total = len(records)
    payload = {
        "strategy": "one_process_per_trial",
        "start_trial": args.start_trial,
        "total_episodes": total,
        "successes": successes,
        "success_rate": successes / total,
        "wilson_95_interval": wilson_interval(successes, total),
        "summed_process_duration_seconds": sum(
            record["duration_seconds"] for record in records
        ),
        "trials": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
