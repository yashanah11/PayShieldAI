import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier

from generation.generator import generate_transactions
from generation.generator import FEATURES


def make_stealth_dataset(
    n=10000,
    fraud_rate=0.05,
    seed=42,
    level=1,
):
    rng = np.random.default_rng(seed)

    df = generate_transactions(n, seed)
    df["is_fraud"] = 0

    fraud_count = int(n * fraud_rate)

    indices = rng.choice(
        df.index,
        size=fraud_count,
        replace=False,
    )

    # Progressive adversarial curriculum.
    #
    # The important point is that no single feature
    # should be an extreme outlier.

    if level == 1:
        # Moderate anomalies.
        df.loc[indices, "amount"] *= rng.uniform(
            1.5, 2.5, fraud_count
        )
        df.loc[indices, "velocity_24h"] += rng.integers(
            2, 6, fraud_count
        )
        df.loc[indices, "distance_km"] += rng.uniform(
            20, 80, fraud_count
        )

    elif level == 2:
        # Subtle coordinated anomalies.
        df.loc[indices, "amount"] *= rng.uniform(
            1.3, 2.0, fraud_count
        )
        df.loc[indices, "velocity_1h"] += rng.integers(
            1, 3, fraud_count
        )
        df.loc[indices, "velocity_24h"] += rng.integers(
            2, 5, fraud_count
        )
        df.loc[indices, "distance_km"] += rng.uniform(
            20, 70, fraud_count
        )

    elif level == 3:
        # Strongly stealthy attacks:
        # individually normal, jointly suspicious.
        df.loc[indices, "amount"] *= rng.uniform(
            1.2, 1.8, fraud_count
        )
        df.loc[indices, "velocity_1h"] += rng.integers(
            1, 3, fraud_count
        )
        df.loc[indices, "velocity_24h"] += rng.integers(
            1, 4, fraud_count
        )
        df.loc[indices, "distance_km"] += rng.uniform(
            15, 60, fraud_count
        )

        df.loc[indices, "merchant_risk"] = rng.uniform(
            0.15, 0.55, fraud_count
        )

        df.loc[indices, "device_age_days"] = rng.integers(
            100, 1200, fraud_count
        )

    else:
        # Level 4: multi-factor stealth.
        #
        # Keep every individual feature relatively ordinary.
        # The model must learn the interaction.
        df.loc[indices, "amount"] *= rng.uniform(
            1.15, 1.7, fraud_count
        )

        df.loc[indices, "velocity_1h"] += rng.integers(
            1, 3, fraud_count
        )

        df.loc[indices, "velocity_24h"] += rng.integers(
            1, 4, fraud_count
        )

        df.loc[indices, "distance_km"] += rng.uniform(
            15, 55, fraud_count
        )

        df.loc[indices, "merchant_risk"] = rng.uniform(
            0.15, 0.55, fraud_count
        )

        df.loc[indices, "device_age_days"] = rng.integers(
            150, 1200, fraud_count
        )

    df.loc[indices, "is_fraud"] = 1

    return df


def build_stealth_curriculum(seed=42):
    datasets = []

    for level in range(1, 5):
        datasets.append(
            make_stealth_dataset(
                n=10000,
                fraud_rate=0.05,
                seed=seed + level,
                level=level,
            )
        )

    return datasets


def train_model(df, seed=42):
    X = df[FEATURES]
    y = df["is_fraud"]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.9,
        gamma=0.05,
        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(X, y)

    return model


def main():
    print("=== STAGE 14: STEALTH ADVERSARIAL CURRICULUM ===")

    curriculum = build_stealth_curriculum()

    training = pd.concat(
        curriculum,
        ignore_index=True,
    )

    print(
        f"Training transactions: {len(training)}"
    )

    print(
        f"Fraud transactions: "
        f"{int(training['is_fraud'].sum())}"
    )

    print()

    model = train_model(training)

    # Completely separate stealth holdout.
    holdout = make_stealth_dataset(
        n=12000,
        fraud_rate=0.05,
        seed=7777,
        level=4,
    )

    probabilities = model.predict_proba(
        holdout[FEATURES]
    )[:, 1]

    threshold = 0.10

    predictions = (
        probabilities >= threshold
    ).astype(int)

    auc = roc_auc_score(
        holdout["is_fraud"],
        probabilities,
    )

    recall = recall_score(
        holdout["is_fraud"],
        predictions,
        zero_division=0,
    )

    print("=== UNSEEN STEALTH HOLDOUT ===")
    print(
        f"Holdout transactions : {len(holdout)}"
    )
    print(
        f"Fraud cases          : "
        f"{int(holdout['is_fraud'].sum())}"
    )
    print(
        f"ROC-AUC              : {auc:.4f}"
    )
    print(
        f"Recall @ {threshold:.2f}       : "
        f"{recall:.4f}"
    )

    print()

    if auc >= 0.90 and recall >= 0.70:
        print(
            "VERDICT: STEALTH ROBUSTNESS PASS"
        )
    else:
        print(
            "VERDICT: STEALTH ROBUSTNESS FAILED"
        )


if __name__ == "__main__":
    main()
