import numpy as np
import pandas as pd

from generation.generator import FEATURES


def generate_stage2_repaired_benchmark(
    n=10000,
    fraud_rate=0.05,
    seed=42,
):
    rng = np.random.default_rng(seed)

    n_fraud = int(n * fraud_rate)
    n_normal = n - n_fraud

    # ============================================================
    # BENIGN TRANSACTIONS
    # ============================================================

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

    # ============================================================
    # FRAUD
    #
    # Multiple fraud modes.
    # Important: fraud is NOT always a young device,
    # high merchant risk, or extreme geography.
    # ============================================================

    fraud = pd.DataFrame({
        "amount": np.round(
            rng.lognormal(5.2, 1.0, n_fraud), 2
        ),
        "hour": rng.integers(0, 24, n_fraud),
        "velocity_1h": rng.poisson(2.2, n_fraud),
        "velocity_24h": rng.poisson(7, n_fraud),

        # Broad device-age distribution.
        "device_age_days": rng.integers(
            1,
            1500,
            n_fraud,
        ),

        # Broader geographic distribution.
        "distance_km": np.round(
            rng.exponential(35, n_fraud),
            2,
        ),

        # Fraud should include ordinary merchant-risk values.
        "merchant_risk": np.round(
            rng.beta(2.5, 5.0, n_fraud),
            4,
        ),

        "is_fraud": 1,
    })

    # ============================================================
    # FRAUD SUB-FAMILIES
    #
    # Each family modifies only part of the feature space.
    # This prevents the model from relying on one shortcut.
    # ============================================================

    indices = rng.permutation(n_fraud)

    # ------------------------------------------------------------
    # Family 1: velocity fraud
    # ------------------------------------------------------------

    n1 = n_fraud // 3
    idx = indices[:n1]

    fraud.loc[idx, "velocity_1h"] += rng.integers(
        2,
        5,
        len(idx),
    )

    fraud.loc[idx, "velocity_24h"] += rng.integers(
        3,
        10,
        len(idx),
    )

    # Keep device and merchant signals ordinary.

    # ------------------------------------------------------------
    # Family 2: geographic fraud
    # ------------------------------------------------------------

    n2 = n_fraud // 3
    idx = indices[n1:n1 + n2]

    fraud.loc[idx, "distance_km"] += rng.uniform(
        30,
        120,
        len(idx),
    )

    # Deliberately normal device age.
    fraud.loc[idx, "device_age_days"] = rng.integers(
        100,
        1200,
        len(idx),
    )

    # Deliberately ordinary merchant risk.
    fraud.loc[idx, "merchant_risk"] = rng.uniform(
        0.2,
        0.6,
        len(idx),
    )

    # ------------------------------------------------------------
    # Family 3: coordinated stealth fraud
    # ------------------------------------------------------------

    idx = indices[n1 + n2:]

    fraud.loc[idx, "amount"] *= rng.uniform(
        1.5,
        3.0,
        len(idx),
    )

    fraud.loc[idx, "velocity_24h"] += rng.integers(
        2,
        8,
        len(idx),
    )

    fraud.loc[idx, "distance_km"] += rng.uniform(
        20,
        100,
        len(idx),
    )

    # Important:
    # Do NOT force young devices or high merchant risk.

    # ============================================================
    # COMBINE
    # ============================================================

    df = pd.concat(
        [normal, fraud],
        ignore_index=True,
    )

    df = df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return df


def validate_stage2_benchmark(df):
    required = FEATURES + ["is_fraud"]

    assert all(
        column in df.columns
        for column in required
    )

    assert len(df) > 0

    assert set(
        df["is_fraud"].unique()
    ).issubset({0, 1})

    assert df["is_fraud"].sum() > 0

    assert (df["is_fraud"] == 0).sum() > 0

    return True


if __name__ == "__main__":
    df = generate_stage2_repaired_benchmark()

    validate_stage2_benchmark(df)

    print("STAGE 2 REPAIRED BENCHMARK: OK")
    print(f"TRANSACTIONS: {len(df)}")
    print(f"FRAUD: {int(df['is_fraud'].sum())}")
    print(
        f"NON-FRAUD: "
        f"{int((df['is_fraud'] == 0).sum())}"
    )