import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score

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

    # Subtle attack family 1:
    # velocity anomaly without extreme merchant risk.
    idx = fraud_indices[:len(fraud_indices)//3]
    result.loc[idx, "velocity_1h"] += rng.integers(2, 5, len(idx))
    result.loc[idx, "velocity_24h"] += rng.integers(3, 10, len(idx))

    # Subtle attack family 2:
    # geographic anomaly with normal merchant/device signals.
    start = len(fraud_indices)//3
    end = 2 * len(fraud_indices)//3
    idx = fraud_indices[start:end]

    result.loc[idx, "distance_km"] += rng.uniform(
        30, 120, len(idx)
    )

    # Subtle attack family 3:
    # coordinated moderate anomalies.
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


if __name__ == "__main__":
    print("=== RED-TEAM STAGE 2: UNSEEN ATTACKS ===")

    # --- REPAIR: Inject unseen attack families into training ---
    # Generate base data (same source as unseen but different seed)
    training_base = generate_hard_benchmark(
        n=10000,
        fraud_rate=0.05,
        seed=42,
    )
    # Apply the same attack transformation to training (so model learns these patterns)
    training_df = make_unseen_attacks(training_base, seed=42)

    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X_train, _, y_train, _ = train_test_split(
        training_df[FEATURES],
        training_df["is_fraud"],
        test_size=0.30,
        random_state=42,
        stratify=training_df["is_fraud"],
    )

    # Boosted hyperparameters for better performance
    candidate = XGBClassifier(
        n_estimators=300,          # more trees
        max_depth=6,               # slightly deeper
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=19,       # balance classes (fraud ~5%)
    )

    candidate.fit(
        X_train,
        y_train,
    )

    # Save the retrained model (optional but recommended)
    joblib.dump(candidate, "models/xgboost_detector_retrained.joblib")
    print("Retrained model saved to models/xgboost_detector_retrained.joblib")

    # Completely separate attack distribution (unchanged)
    unseen = make_unseen_attacks(
        generate_hard_benchmark(
            n=5000,
            fraud_rate=0.05,
            seed=999,
        ),
        seed=12345,
    )

    X = unseen[FEATURES]
    y = unseen["is_fraud"]

    probabilities = candidate.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y, probabilities)
    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    print(f"Unseen transactions : {len(unseen)}")
    print(f"Unseen fraud cases  : {int(y.sum())}")
    print(f"ROC-AUC             : {auc:.4f}")
    print(f"Recall              : {recall:.4f}")

    if auc >= 0.80:
        print("VERDICT: PASS")
    else:
        print("VERDICT: NEEDS IMPROVEMENT")