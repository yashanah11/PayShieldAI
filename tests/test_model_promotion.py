from pathlib import Path

from detection.model_promotion import promote_if_better


def test_reject_worse_model():
    # A worse candidate must never be promoted.
    promoted = promote_if_better(
        None,
        0.95,
        0.90,
    )

    assert promoted is False


def test_reject_invalid_candidate():
    # Invalid candidates must never overwrite the real model.
    promoted = promote_if_better(
        None,
        0.90,
        0.95,
    )

    assert promoted is False
    assert Path("models/xgboost_detector.joblib").exists()
