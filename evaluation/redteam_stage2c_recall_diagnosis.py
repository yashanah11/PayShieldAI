import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

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

    idx = fraud_indices[:len(fraud_indices) // 3]

    result.loc[idx, "velocity_1h"] += rng.integers(
        2, 5, len(idx)
    )
    result.loc[idx, "velocity_24h"] += rng.integers(
        3, 10, len(idx)
    )

    start = len(fraud_indices) // 3
    end = 2 * len(fraud_indices) // 3

    idx = fraud_indices[start:end]

    result.loc[idx, "distance_km"] += rng.uniform(
        30, 120, len(idx)
    )

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


print("=== STAGE 2C RECALL DIAGNOSIS ===")

training_df = generate_stage2_repaired_benchmark(
    n=10000,
    fraud_rate=0.05,
    seed=42,
)

X_train, _, y_train, _ = train_test_split(
    training_df[FEATURES],
    training_df["is_fraud"],
    test_size=0.30,
    random_state=42,
    stratify=training_df["is_fraud"],
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_train, y_train)

unseen_base = generate_stage2_repaired_benchmark(
    n=5000,
    fraud_rate=0.05,
    seed=999,
)

unseen = make_unseen_attacks(
    unseen_base,
    seed=12345,
)

X = unseen[FEATURES]
y = unseen["is_fraud"]

scores = model.predict_proba(X)[:, 1]

fraud_scores = scores[y == 1]
benign_scores = scores[y == 0]

print()
print("=== GLOBAL ===")
print(f"AUC: {roc_auc_score(y, scores):.4f}")
print(f"Fraud cases: {len(fraud_scores)}")
print(f"Benign cases: {len(benign_scores)}")

print()
print("=== FRAUD SCORE DISTRIBUTION ===")
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

print()
print("=== BENIGN SCORE DISTRIBUTION ===")
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

print()
print("=== FRAUD ABOVE THRESHOLDS ===")

for threshold in [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]:
    recall = np.mean(fraud_scores >= threshold)
    fpr = np.mean(benign_scores >= threshold)

    print(
        f"{threshold:0.2f} | "
        f"Recall {recall:.4f} | "
        f"FPR {fpr:.4f}"
    )

print()
print("=== ATTACK FAMILY SCORES ===")

fraud_positions = np.flatnonzero(y.to_numpy() == 1)

n = len(fraud_positions)

groups = {
    "velocity": fraud_positions[:n // 3],
    "geographic": fraud_positions[n // 3:2 * n // 3],
    "coordinated": fraud_positions[2 * n // 3:],
}

for name, positions in groups.items():
    family_scores = scores[positions]

    print()
    print(name.upper())
    print(f"count : {len(family_scores)}")
    print(f"mean  : {family_scores.mean():.6f}")
    print(f"median: {np.median(family_scores):.6f}")
    print(f"p90   : {np.quantile(family_scores, 0.90):.6f}")
    print(f">=0.50: {np.mean(family_scores >= 0.50):.4f}")

print()
print("=== FEATURE IMPORTANCE ===")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print(importance)

print()
print("=== DIAGNOSIS COMPLETE ===")
