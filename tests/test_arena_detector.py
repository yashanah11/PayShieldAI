from arena.detector_runner import run_arena_round


def test_arena_detector():
    result = run_arena_round(
        round_number=1,
        n=1000,
        seed=42,
    )

    assert result.attack_count == 50
    assert 0 <= result.detected_count <= result.attack_count
    assert 0.0 <= result.detection_rate <= 1.0
