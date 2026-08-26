import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score

from xgboost import XGBClassifier

from generation.generator import generate_transactions, FEATURES
from evaluation.redteam_stealth_curriculum import (
    build_stealth_curriculum,
    make_stealth_dataset,
)


def generate_stealth_hard_negatives(
    n=20000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    df = generate_transactions(n, seed)

    # Legitimate transactions intentionally resembling
    # stealth fraud without being labelled fraud.
    df["amount"] *= rng.uniform(
        1.2, 2.0, n
    )

    df["velocity_1h"] += rng.integers(
        1, 3, n
    )

    df["velocity_24h"] += rng.integers(
        1, 5, n
    )

    df["distance_km"] += rng.uniform(
        20, 100, n
    )

    df["device_age_days"] = rng.integers(
        50, 1000, n
    )

    df["merchant_risk"] = rng.uniform(
        0.10, 0.60, n
    )

    df["is_fraud"] = 0

    return df


def train_balanced_model(
    seed=42,
):
    # Original adversarial curriculum.
    fraud_training = pd.concat(
        build_stealth_curriculum(
            seed=seed
        ),
        ignore_index=True,
    )

    # New hard-negative benign examples.
    benign_training = (
        generate_stealth_hard_negatives(
            n=20000,
            seed=seed + 5000,
        )
    )

    training = pd.concat(
        [
            fraud_training,
            benign_training,
        ],
        ignore_index=True,
    )

    X = training[FEATURES]
    y = training["is_fraud"]

    model = XGBClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.035,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.9,
        gamma=0.08,
        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(X, y)

    return model, training


def main():

    print(
        "=== STAGE 15: STEALTH HARD-NEGATIVE FIX ==="
    )

    model, training = train_balanced_model(
        seed=42
    )

    print(
        f"Training transactions : "
        f"{len(training)}"
    )

    print(
        f"Fraud transactions    : "
        f"{int(training.is_fraud.sum())}"
    )

    print(
        f"Benign transactions   : "
        f"{int((training.is_fraud == 0).sum())}"
    )

    # Completely unseen stealth fraud.
    fraud = make_stealth_dataset(
        n=12000,
        fraud_rate=0.05,
        seed=8888,
        level=4,
    )

    fraud_scores = model.predict_proba(
        fraud[FEATURES]
    )[:, 1]

    threshold = 0.10

    fraud_predictions = (
        fraud_scores >= threshold
    ).astype(int)

    auc = roc_auc_score(
        fraud["is_fraud"],
        fraud_scores,
    )

    recall = recall_score(
        fraud["is_fraud"],
        fraud_predictions,
        zero_division=0,
    )

    # Completely unseen benign hard negatives.
    benign = generate_stealth_hard_negatives(
        n=12000,
        seed=9999,
    )

    benign_scores = model.predict_proba(
        benign[FEATURES]
    )[:, 1]

    fpr = float(
        (
            benign_scores >= threshold
        ).mean()
    )

    print()
    print("=== UNSEEN STEALTH FRAUD ===")

    print(
        f"ROC-AUC       : {auc:.4f}"
    )

    print(
        f"Recall @ 0.10 : {recall:.4f}"
    )

    print()
    print("=== UNSEEN HARD BENIGN ===")

    print(
        f"Transactions : {len(benign)}"
    )

    print(
        f"False positives : "
        f"{int((benign_scores >= threshold).sum())}"
    )

    print(
        f"FPR             : {fpr:.4f}"
    )

    print()
    print("=== HARD GATE ===")

    print(
        f"AUC >= 0.90 : {auc >= 0.90}"
    )

    print(
        f"Recall >= 0.70 : "
        f"{recall >= 0.70}"
    )

    print(
        f"FPR <= 0.05 : "
        f"{fpr <= 0.05}"
    )

    print()

    if (
        auc >= 0.90
        and recall >= 0.70
        and fpr <= 0.05
    ):
        print(
            "VERDICT: HARD-NEGATIVE FIX PASS"
        )
    else:
        print(
            "VERDICT: HARD-NEGATIVE FIX FAILED"
        )


if __name__ == "__main__":
    main()
