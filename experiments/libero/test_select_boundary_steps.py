from experiments.libero.select_boundary_steps import select_boundaries


def row(score: float, *, grasped: bool = False, success: bool = False) -> dict:
    return {"score": score, "grasped": grasped, "success": success}


def test_no_grasp_uses_major_progress_and_completion_boundaries() -> None:
    steps = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    rows = [
        row(0.00),
        row(0.01),
        row(0.03),
        row(0.08),
        row(0.30),
        row(0.28),
        row(0.27),
        row(0.26),
        row(1.00, success=True),
    ]

    selected = select_boundaries(steps, rows)

    assert [(item["phase"], item["policy_step"]) for item in selected] == [
        ("pre_major_progress", 30),
        ("major_progress", 40),
        ("pre_completion", 60),
    ]


def test_grasp_boundaries_remain_preferred() -> None:
    steps = [0, 10, 20, 30, 40]
    rows = [row(0.0), row(0.2), row(0.5, grasped=True), row(0.7, grasped=True), row(1.0, success=True)]

    selected = select_boundaries(steps, rows)

    assert [(item["phase"], item["policy_step"]) for item in selected] == [
        ("pre_grasp", 10),
        ("first_grasp", 20),
    ]
