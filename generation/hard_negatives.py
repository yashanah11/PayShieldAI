import numpy as np

from generation.generator import generate_transactions


def generate_legitimate_travelers(
    n=10000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    df = generate_transactions(
        n,
        seed,
    )

    traveler_count = int(n * 0.20)

    indices = rng.choice(
        df.index,
        size=traveler_count,
        replace=False,
    )

    # Legitimate long-distance travel.
    df.loc[
        indices,
        "distance_km"
    ] += rng.uniform(
        50,
        250,
        traveler_count,
    )

    # Legitimate large purchases.
    df.loc[
        indices,
        "amount"
    ] *= rng.uniform(
        1.5,
        5.0,
        traveler_count,
    )

    # Some newer devices.
    df.loc[
        indices,
        "device_age_days"
    ] = rng.integers(
        10,
        180,
        traveler_count,
    )

    # Unusual but legitimate hours.
    df.loc[
        indices,
        "hour"
    ] = rng.choice(
        [0, 1, 2, 3, 4, 23],
        size=traveler_count,
    )

    # Moderate merchant risk.
    df.loc[
        indices,
        "merchant_risk"
    ] = rng.uniform(
        0.15,
        0.55,
        traveler_count,
    )

    # Keep them legitimate.
    df["is_fraud"] = 0

    return df
