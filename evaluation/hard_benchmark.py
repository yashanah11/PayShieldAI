import numpy as np
import pandas as pd

from generation.generator import FEATURES


def generate_hard_benchmark(
    n=10000,
    fraud_rate=0.05,
    seed=42,
):
    rng = np.random.default_rng(seed)

    n_fraud = int(n * fraud_rate)
    n_normal = n - n_fraud

    normal = pd.DataFrame({
        "amount": np.round(
            rng.lognormal(5.0, 1.0, n_normal), 2
        ),
        "hour": rng.integers(0, 24, n_normal),
        "velocity_1h": rng.poisson(1.5, n_normal),
        "velocity_24h": rng.poisson(5, n_normal),
        "device_age_days": rng.integers(1, 1500, n_normal),
        "distance_km": np.round(
            rng.exponential(25, n_normal), 2
        ),
        "merchant_risk": np.round(
            rng.beta(2, 5, n_normal), 4
        ),
        "is_fraud": 0,
    })

    fraud = pd.DataFrame({
        "amount": np.round(
            rng.lognormal(5.2, 1.0, n_fraud), 2
        ),
        "hour": rng.integers(0, 24, n_fraud),
        "velocity_1h": rng.poisson(2.2, n_fraud),
        "velocity_24h": rng.poisson(7, n_fraud),
        "device_age_days": rng.integers(5, 500, n_fraud),
        "distance_km": np.round(
            rng.exponential(30, n_fraud), 2
        ),
        "merchant_risk": np.round(
            rng.beta(3, 4, n_fraud), 4
        ),
        "is_fraud": 1,
    })

    df = pd.concat(
        [normal, fraud],
        ignore_index=True,
    )

    df = df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return df


def validate_hard_benchmark(df):
    required = FEATURES + ["is_fraud"]

    assert all(
        column in df.columns
        for column in required
    )

    assert len(df) > 0

    assert set(df["is_fraud"].unique()).issubset({0, 1})

    assert df["is_fraud"].sum() > 0

    assert (df["is_fraud"] == 0).sum() > 0

    return True


if __name__ == "__main__":
    df = generate_hard_benchmark()

    validate_hard_benchmark(df)

    print("HARD BENCHMARK: OK")
    print(f"TRANSACTIONS: {len(df)}")
    print(f"FRAUD: {int(df['is_fraud'].sum())}")
    print(
        f"NON-FRAUD: {int((df['is_fraud'] == 0).sum())}"
    )
