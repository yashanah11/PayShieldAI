from pathlib import Path


def test_demo_exists():
    assert Path("demo.py").exists()


def test_demo_inputs_exist():
    required = [
        "evaluation/baseline_metrics.json",
        "evaluation/pipeline_results.json",
        "evaluation/adaptive_arena_results.json",
        "evaluation/self_improvement.json",
        "evaluation/model_promotion.json",
        "evaluation/model_audit.json",
    ]

    for path in required:
        assert Path(path).exists()
