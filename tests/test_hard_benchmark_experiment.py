import json
from pathlib import Path

from evaluation.hard_benchmark_experiment import run_experiment


def test_hard_experiment():
    run_experiment(
        n=1000,
        fraud_rate=0.05,
        seed=42,
    )

    path = Path(
        "evaluation/hard_benchmark_experiment.json"
    )

    assert path.exists()

    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    assert result["train_size"] > 0
    assert result["holdout_size"] > 0
    assert result["hard_examples_found"] >= 0

    assert 0 <= result["baseline"]["roc_auc"] <= 1
    assert 0 <= result["candidate"]["roc_auc"] <= 1

    assert result["holdout_used_for_training"] is False


def test_experiment_has_real_delta():
    run_experiment(
        n=1000,
        fraud_rate=0.05,
        seed=7,
    )

    with open(
        "evaluation/hard_benchmark_experiment.json",
        encoding="utf-8",
    ) as f:
        result = json.load(f)

    expected_delta = (
        result["candidate"]["roc_auc"]
        - result["baseline"]["roc_auc"]
    )

    assert abs(
        result["roc_auc_delta"] - expected_delta
    ) < 1e-12
