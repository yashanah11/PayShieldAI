from evaluation.hard_benchmark import (
    generate_hard_benchmark,
    validate_hard_benchmark,
)


def test_hard_benchmark_generation():
    df = generate_hard_benchmark(
        n=1000,
        fraud_rate=0.05,
        seed=42,
    )

    assert len(df) == 1000
    assert validate_hard_benchmark(df)


def test_hard_benchmark_reproducible():
    first = generate_hard_benchmark(
        n=500,
        fraud_rate=0.05,
        seed=42,
    )

    second = generate_hard_benchmark(
        n=500,
        fraud_rate=0.05,
        seed=42,
    )

    assert first.equals(second)


def test_hard_benchmark_contains_both_classes():
    df = generate_hard_benchmark(
        n=1000,
        fraud_rate=0.05,
        seed=7,
    )

    assert df["is_fraud"].nunique() == 2
