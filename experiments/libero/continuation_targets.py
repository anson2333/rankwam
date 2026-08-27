"""Continuation-aware targets for candidate ranking diagnostics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def summarize_continuation(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Extract dense value and grasp-stability targets from continuation records.

    Records must contain ``continuation_step`` and a nested ``progress`` mapping
    with ``score`` and ``grasped`` fields. The function deliberately returns a
    vector of targets instead of collapsing them into a single reward.
    """
    rows = list(records)
    if not rows:
        raise ValueError("continuation records cannot be empty")
    steps = np.asarray([row["continuation_step"] for row in rows], dtype=np.float64)
    progress = [row.get("progress") for row in rows]
    if any(item is None for item in progress):
        raise ValueError("all continuation records need dense progress")
    scores = np.asarray([item["score"] for item in progress], dtype=np.float64)
    grasped = np.asarray([bool(item["grasped"]) for item in progress], dtype=bool)
    if not np.isfinite(steps).all() or not np.isfinite(scores).all():
        raise ValueError("continuation steps and scores must be finite")
    if np.any(np.diff(steps) < 0):
        raise ValueError("continuation steps must be ordered")
    duration = max(float(steps[-1] - steps[0]), 1.0)
    auc = float(np.trapz(scores, steps) / duration)
    losses = np.flatnonzero(grasped[:-1] & ~grasped[1:])
    recoveries = np.flatnonzero(~grasped[:-1] & grasped[1:])
    return {
        "max_score": float(np.max(scores)),
        "final_score": float(scores[-1]),
        "auc_score": auc,
        "score_range": float(np.ptp(scores)),
        "grasp_fraction": float(np.mean(grasped)),
        "initial_grasped": bool(grasped[0]),
        "final_grasped": bool(grasped[-1]),
        "grasp_loss_count": int(len(losses)),
        "first_grasp_loss_step": None if len(losses) == 0 else int(steps[losses[0] + 1]),
        "grasp_recovery_count": int(len(recoveries)),
        "first_grasp_recovery_step": None if len(recoveries) == 0 else int(steps[recoveries[0] + 1]),
    }
