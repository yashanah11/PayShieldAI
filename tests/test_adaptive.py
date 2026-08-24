from arena.adaptive import run_adaptive_arena


def test_adaptive_arena():
    run_adaptive_arena(
        rounds=3,
        n=500,
        seed=42,
    )

    import json

    with open(
        "evaluation/adaptive_arena_results.json",
        encoding="utf-8",
    ) as f:
        result = json.load(f)

    assert result["rounds"] == 3
    assert len(result["history"]) == 3
    assert 0.0 <= result["summary"]["best_detection_rate"] <= 1.0
