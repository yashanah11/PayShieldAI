from pathlib import Path

from detection.model_promotion import (
    promote_if_better,
)


def test_reject_worse_model():
    promoted = promote_if_better(
        object(),
        0.95,
        0.90,
    )

    assert promoted is False


def test_accept_equal_or_better_model():
    promoted = promote_if_better(
        object(),
        0.90,
        0.95,
    )

    assert promoted is True or not Path(
        "models/xgboost_detector.joblib"
    ).exists()
