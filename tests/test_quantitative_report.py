import json
from pathlib import Path

from evaluation.quantitative_report import load


def test_quantitative_report_inputs():
    required = [
        "baseline_metrics.json",
        "adaptive_arena_results.json",
        "self_improvement.json",
        "model_promotion.json",
        "model_audit.json",
    ]

    for name in required:
        assert (Path("evaluation") / name).exists()


def test_quantitative_report_values():
    baseline = load("baseline_metrics.json")
    self_improvement = load("self_improvement.json")

    assert 0 <= baseline["roc_auc"] <= 1
    assert 0 <= baseline["precision"] <= 1
    assert 0 <= baseline["recall"] <= 1

    delta = (
        self_improvement["new_roc_auc"]
        - self_improvement["old_roc_auc"]
    )

    assert delta == 0.0
