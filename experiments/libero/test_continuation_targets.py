import pytest

from experiments.libero.continuation_targets import summarize_continuation


def test_summarize_continuation_tracks_loss_and_recovery():
    records = [
        {"continuation_step": 0, "progress": {"score": 0.5, "grasped": True}},
        {"continuation_step": 10, "progress": {"score": 0.7, "grasped": False}},
        {"continuation_step": 20, "progress": {"score": 0.6, "grasped": True}},
    ]
    targets = summarize_continuation(records)
    assert targets["max_score"] == pytest.approx(0.7)
    assert targets["auc_score"] == pytest.approx(0.625)
    assert targets["grasp_loss_count"] == 1
    assert targets["first_grasp_loss_step"] == 10
    assert targets["grasp_recovery_count"] == 1
    assert targets["first_grasp_recovery_step"] == 20


def test_summarize_continuation_rejects_missing_progress():
    with pytest.raises(ValueError, match="dense progress"):
        summarize_continuation([{"continuation_step": 0, "progress": None}])
