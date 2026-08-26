import numpy as np
import pandas as pd

from generation.generator import FEATURES
from evaluation.hard_benchmark import generate_hard_benchmark


def make_unseen_attacks(df, seed=42):
    rng = np.random.default_rng(seed)

    result = df.copy()

    fraud_indices = rng.choice(
        result.index,
        size=max(1, int(len(result) * 0.05)),
        replace=False,
    )

    # Velocity attacks
    idx = fraud_indices[:len(fraud_indices)//3]

    result.loc[idx, "velocity_1h"] += rng.integers(
        2, 5, len(idx)
    )
    result.loc[idx, "velocity_24h"] += rng.integers(
        3, 10, len(idx)
    )

    # Geographic attacks
    start = len(fraud_indices)//3
    end = 2 * len(fraud_indices)//3

    idx = fraud_indices[start:end]

    result.loc[idx, "distance_km"] += rng.uniform(
        30, 120, len(idx)
    )

    # Coordinated attacks
    idx = fraud_indices[end:]

    result.loc[idx, "amount"] *= rng.uniform(
        1.5, 3.0, len(idx)
    )

    result.loc[idx, "velocity_24h"] += rng.integers(
        2, 8, len(idx)
    )

    result.loc[idx, "distance_km"] += rng.uniform(
        20, 100, len(idx)
    )

    result["is_fraud"] = 0
    result.loc[fraud_indices, "is_fraud"] = 1

    return result


print("=== STAGE 2A: DATA & LABEL AUDIT ===")

train = generate_hard_benchmark(
    n=10000,
    fraud_rate=0.05,
    seed=42,
)

unseen_base = generate_hard_benchmark(
    n=5000,
    fraud_rate=0.05,
    seed=999,
)

unseen = make_unseen_attacks(
    unseen_base,
    seed=12345,
)

print()
print("=== LABEL COUNTS ===")

print("TRAIN")
print(train["is_fraud"].value_counts().sort_index())

print()
print("UNSEEN")
print(unseen["is_fraud"].value_counts().sort_index())

print()
print("=== TRAIN FRAUD MEANS ===")

print(
    train.loc[
        train["is_fraud"] == 1,
        FEATURES
    ].mean().round(4)
)

print()
print("=== UNSEEN FRAUD MEANS ===")

print(
    unseen.loc[
        unseen["is_fraud"] == 1,
        FEATURES
    ].mean().round(4)
)

print()
print("=== TRAIN BENIGN MEANS ===")

print(
    train.loc[
        train["is_fraud"] == 0,
        FEATURES
    ].mean().round(4)
)

print()
print("=== UNSEEN BENIGN MEANS ===")

print(
    unseen.loc[
        unseen["is_fraud"] == 0,
        FEATURES
    ].mean().round(4)
)

print()
print("=== ATTACK FAMILY SIZES ===")

fraud_count = int(unseen["is_fraud"].sum())

print("Total fraud :", fraud_count)
print("Velocity    :", fraud_count // 3)
print("Geographic  :", fraud_count // 3)
print(
    "Coordinated :",
    fraud_count - 2 * (fraud_count // 3)
)

print()
print("=== TRAIN FRAUD vs UNSEEN FRAUD ===")

train_fraud = train.loc[
    train["is_fraud"] == 1,
    FEATURES
].mean()

unseen_fraud = unseen.loc[
    unseen["is_fraud"] == 1,
    FEATURES
].mean()

difference = (
    unseen_fraud - train_fraud
).sort_values(
    key=lambda x: abs(x),
    ascending=False
)

print(difference.round(4))

print()
print("=== RANGE CHECK ===")

for feature in FEATURES:

    train_min = train[feature].min()
    train_max = train[feature].max()

    unseen_min = unseen.loc[
        unseen["is_fraud"] == 1,
        feature
    ].min()

    unseen_max = unseen.loc[
        unseen["is_fraud"] == 1,
        feature
    ].max()

    print(
        f"{feature:18s} "
        f"TRAIN [{train_min:.3f}, {train_max:.3f}] "
        f"UNSEEN [{unseen_min:.3f}, {unseen_max:.3f}]"
    )

print()
print("=== STAGE 2A COMPLETE ===")
