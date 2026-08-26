import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from generation.generator import FEATURES
from evaluation.hard_benchmark_stage2_repaired import (
    generate_stage2_repaired_benchmark,
)


def make_unseen_attacks(df, seed=12345):
    rng = np.random.default_rng(seed)

    result = df.copy()

    fraud_indices = rng.choice(
        result.index,
        size=max(1, int(len(result) * 0.05)),
        replace=False,
    )

    # Velocity attacks
    idx = fraud_indices[:len(fraud_indices) // 3]

    result.loc[idx, "velocity_1h"] += rng.integers(
        2, 5, len(idx)
    )

    result.loc[idx, "velocity_24h"] += rng.integers(
        3, 10, len(idx)
    )

    # Geographic attacks
    start = len(fraud_indices) // 3
    end = 2 * len(fraud_indices) // 3

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


print("=== STAGE 2C DIAGNOSTIC: SCORE DISTRIBUTION ===")

model = joblib.load(
    "models/xgboost_detector.joblib"
)

base = generate_stage2_repaired_benchmark(
    n=5000,
    fraud_rate=0.05,
    seed=999,
)

unseen = make_unseen_attacks(
    base,
    seed=12345,
)

X = unseen[FEATURES]
y = unseen["is_fraud"]

scores = model.predict_proba(X)[:, 1]

fraud_scores = scores[y == 1]
benign_scores = scores[y == 0]


print("\n=== FRAUD SCORES ===")

print(
    pd.Series(fraud_scores).describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


print("\n=== BENIGN SCORES ===")

print(
    pd.Series(benign_scores).describe(
        percentiles=[
            0.90,
            0.95,
            0.99,
            0.995,
            0.999,
        ]
    )
)


print("\n=== THRESHOLD ANALYSIS ===")

for threshold in np.arange(0.05, 0.51, 0.05):

    predictions = (
        scores >= threshold
    ).astype(int)

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    benign_fpr = predictions[y == 0].mean()

    print(
        f"{threshold:0.2f} | "
        f"FPR {benign_fpr:.4f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )


print("\n=== SCORE QUANTILES ===")

print(
    "Fraud median :",
    np.median(fraud_scores)
)

print(
    "Benign median:",
    np.median(benign_scores)
)

print(
    "Fraud >= 0.50:",
    np.mean(fraud_scores >= 0.50)
)

print(
    "Fraud >= 0.30:",
    np.mean(fraud_scores >= 0.30)
)

print(
    "Fraud >= 0.10:",
    np.mean(fraud_scores >= 0.10)
)

print("\n=== DIAGNOSIS COMPLETE ===")
