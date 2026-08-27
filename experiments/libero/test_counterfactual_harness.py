import numpy as np

from experiments.libero.counterfactual_harness import (
    candidate_diversity,
    summarize_headroom,
    summarize_progress,
)
from experiments.libero.task_progress import shaped_pick_place_score


def test_candidate_diversity_reports_shape_and_variance():
    candidates = np.zeros((3, 2, 7), dtype=np.float64)
    candidates[1, :, 0] = 1.0
    candidates[2, :, 1] = 2.0

    result = candidate_diversity(candidates)

    assert result["num_candidates"] == 3
    assert result["horizon"] == 2
    assert result["mean_pairwise_l2"] > 0.0
    assert result["per_action_dim_variance"][0] > 0.0
    assert result["per_action_dim_variance"][1] > 0.0
    assert result["per_action_dim_variance"][2:] == [0.0] * 5


def test_headroom_detects_informative_group():
    result = summarize_headroom([False, True, False, True])

    assert result["informative"] is True
    assert result["num_successes"] == 2
    assert result["random_candidate_success"] == 0.5
    assert result["first_candidate_success"] == 0.0
    assert result["simulator_oracle_success"] == 1.0
    assert result["oracle_uplift_over_random_percentage_points"] == 50.0


def test_headroom_rejects_empty_group():
    try:
        summarize_headroom([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty candidate group should fail")


def test_progress_headroom_reports_oracle_uplift():
    result = summarize_progress([0.2, 0.5, 0.3])

    assert result["simulator_oracle_progress"] == 0.5
    assert result["progress_range"] == 0.3
    assert result["oracle_uplift_over_first"] == 0.3


def test_pick_place_progress_is_stage_ordered():
    reaching = shaped_pick_place_score(
        eef_to_object=0.01,
        object_to_target=0.5,
        grasped=False,
        success=False,
    )
    transporting = shaped_pick_place_score(
        eef_to_object=0.01,
        object_to_target=0.1,
        grasped=True,
        success=False,
    )
    succeeded = shaped_pick_place_score(
        eef_to_object=0.2,
        object_to_target=0.0,
        grasped=False,
        success=True,
    )

    assert 0.0 <= reaching["score"] < 0.5
    assert 0.5 <= transporting["score"] < 1.0
    assert succeeded["score"] == 1.0
