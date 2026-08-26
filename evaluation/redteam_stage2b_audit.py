import pandas as pd

from evaluation.hard_benchmark_stage2_repaired import (
    generate_stage2_repaired_benchmark,
)

from evaluation.redteam_stage2a_audit import make_unseen_attacks


print("=== STAGE 2B: REPAIRED DISTRIBUTION AUDIT ===")

train = generate_stage2_repaired_benchmark(
    n=10000,
    fraud_rate=0.05,
    seed=42,
)

unseen_base = generate_stage2_repaired_benchmark(
    n=5000,
    fraud_rate=0.05,
    seed=999,
)

unseen = make_unseen_attacks(
    unseen_base,
    seed=12345,
)

train_fraud = train[
    train["is_fraud"] == 1
]

unseen_fraud = unseen[
    unseen["is_fraud"] == 1
]

train_benign = train[
    train["is_fraud"] == 0
]

unseen_benign = unseen[
    unseen["is_fraud"] == 0
]

features = [
    "amount",
    "hour",
    "velocity_1h",
    "velocity_24h",
    "device_age_days",
    "distance_km",
    "merchant_risk",
]

print()
print("=== TRAIN FRAUD MEANS ===")
print(train_fraud[features].mean())

print()
print("=== UNSEEN FRAUD MEANS ===")
print(unseen_fraud[features].mean())

print()
print("=== TRAIN BENIGN MEANS ===")
print(train_benign[features].mean())

print()
print("=== UNSEEN BENIGN MEANS ===")
print(unseen_benign[features].mean())

print()
print("=== TRAIN FRAUD vs UNSEEN FRAUD ===")

difference = (
    unseen_fraud[features].mean()
    - train_fraud[features].mean()
)

print(
    difference.sort_values(
        key=lambda x: abs(x),
        ascending=False,
    )
)

print()
print("=== STAGE 2B COMPLETE ===")