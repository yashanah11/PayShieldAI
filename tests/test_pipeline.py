from arena.pipeline import run_pipeline


def test_pipeline():
    run_pipeline(n=100, seed=42)

    import json

    with open(
        "evaluation/pipeline_results.json",
        encoding="utf-8",
    ) as f:
        result = json.load(f)

    assert result["transactions_analyzed"] == 100
    assert result["fraud_predictions"] >= 0
    assert result["results_with_explanations"] >= 0
