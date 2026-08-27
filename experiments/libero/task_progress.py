"""Task-specific simulator progress metrics for RankWAM pilot labels."""

from __future__ import annotations

from typing import Any

import numpy as np


PILOT_TASKS = {
    ("libero_object", 0): ("alphabet_soup_1", "basket_1_contain_region"),
    ("libero_object", 2): ("alphabet_soup_1", "basket_1_contain_region"),
    ("libero_object", 3): ("alphabet_soup_1", "basket_1_contain_region"),
}


def shaped_pick_place_score(
    *,
    eef_to_object: float,
    object_to_target: float,
    grasped: bool,
    success: bool,
) -> dict[str, float]:
    """Return a stage-aware score in [0, 1] and its shaped components."""
    if not np.isfinite([eef_to_object, object_to_target]).all():
        raise ValueError("pick-place distances must be finite")
    if eef_to_object < 0.0 or object_to_target < 0.0:
        raise ValueError("pick-place distances cannot be negative")

    reach = float(1.0 - np.tanh(10.0 * eef_to_object))
    transport = float(1.0 - np.tanh(5.0 * object_to_target))
    if success:
        score = 1.0
    elif grasped:
        score = 0.5 + 0.49 * transport
    else:
        score = 0.49 * reach
    return {"score": float(score), "reach": reach, "transport": transport}


def _unwrap_env(env: Any) -> Any:
    current = env
    while not hasattr(current, "object_states_dict") and hasattr(current, "env"):
        current = current.env
    if not hasattr(current, "object_states_dict"):
        raise TypeError("could not locate the underlying LIBERO task environment")
    return current


def score_task_progress(
    env: Any,
    obs: dict[str, Any],
    *,
    suite: str,
    task_id: int,
) -> dict[str, Any] | None:
    """Score a supported task from simulator state; return None if unsupported."""
    task_key = (suite, int(task_id))
    if task_key not in PILOT_TASKS:
        return None

    object_name, target_region_name = PILOT_TASKS[task_key]
    inner = _unwrap_env(env)
    object_state = inner.object_states_dict[object_name]
    target_state = inner.object_states_dict[target_region_name]
    object_position = np.asarray(object_state.get_geom_state()["pos"], dtype=np.float64)
    target_position = np.asarray(target_state.get_geom_state()["pos"], dtype=np.float64)
    eef_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)

    grasped = bool(
        inner._check_grasp(inner.robots[0].gripper, inner.get_object(object_name))
    )
    success = bool(env.check_success())
    # Parsed LIBERO goal-state predicates use lowercase registry keys.
    in_target = bool(inner._eval_predicate(["in", object_name, target_region_name]))
    eef_to_object = float(np.linalg.norm(eef_position - object_position))
    object_to_target = float(np.linalg.norm(object_position - target_position))
    shaped = shaped_pick_place_score(
        eef_to_object=eef_to_object,
        object_to_target=object_to_target,
        grasped=grasped,
        success=success,
    )
    return {
        "version": "pick_place_v1",
        **shaped,
        "success": success,
        "grasped": grasped,
        "in_target": in_target,
        "eef_to_object": eef_to_object,
        "object_to_target": object_to_target,
        "eef_position": eef_position.tolist(),
        "object_position": object_position.tolist(),
        "target_position": target_position.tolist(),
        "object_name": object_name,
        "target_region_name": target_region_name,
    }
