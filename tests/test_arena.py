from arena.engine import FraudArena


def test_arena():
    arena = FraudArena()

    r1 = arena.record_round(1, 1000, 850)
    r2 = arena.record_round(2, 1000, 720)
    r3 = arena.record_round(3, 1000, 940)

    assert r1.detection_rate == 0.85
    assert r2.detection_rate == 0.72
    assert r3.detection_rate == 0.94

    summary = arena.summary()

    assert summary["rounds"] == 3
    assert summary["best_detection_rate"] == 0.94
    assert summary["latest_detection_rate"] == 0.94
