import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score

from xgboost import XGBClassifier

from generation.generator import generate_transactions, FEATURES
from generation.hard_negatives import generate_legitimate_travelers
from evaluation.redteam_geo_curriculum import make_geo_dataset


SEED = 42
THRESHOLD = 0.10


def make_stealth_fraud(n=12000, fraud_rate=0.05, seed=42):
    rng = np.random.default_rng(seed)

    df = generate_transactions(n, seed)
    fraud_count = int(n * fraud_rate)

    indices = rng.choice(
        df.index,
        size=fraud_count,
        replace=False,
    )

    df["is_fraud"] = 0

    # Multi-factor stealth fraud:
    # moderate signals rather than extreme single-feature anomalies.
    df.loc[indices, "amount"] *= rng.uniform(
        1.5, 3.0, fraud_count
    )

    df.loc[indices, "velocity_1h"] += rng.integers(
        1, 4, fraud_count
    )

    df.loc[indices, "velocity_24h"] += rng.integers(
        2, 7, fraud_count
    )

    df.loc[indices, "distance_km"] += rng.uniform(
        20, 100, fraud_count
    )

    df.loc[indices, "device_age_days"] = rng.integers(
        30, 500, fraud_count
    )

    df.loc[indices, "merchant_risk"] = rng.uniform(
        0.15, 0.55, fraud_count
    )

    df.loc[indices, "is_fraud"] = 1

    return df


def make_hard_benign(n=12000, seed=42):
    """
    Legitimate transactions deliberately constructed to resemble
    the feature combinations that previously caused false positives.
    """

    rng = np.random.default_rng(seed)

    df = generate_transactions(n, seed)
    df["is_fraud"] = 0

    # Legitimate travelers
    traveler_count = int(n * 0.35)
    traveler_idx = rng.choice(
        df.index,
        size=traveler_count,
        replace=False,
    )

    df.loc[traveler_idx, "distance_km"] = rng.uniform(
        60, 300, traveler_count
    )

    df.loc[traveler_idx, "amount"] *= rng.uniform(
        1.0, 3.5, traveler_count
    )

    # Legitimate high-value purchases
    remaining = df.index.difference(traveler_idx)

    value_count = int(n * 0.25)
    value_idx = rng.choice(
        remaining,
        size=value_count,
        replace=False,
    )

    df.loc[value_idx, "amount"] *= rng.uniform(
        2.0, 8.0, value_count
    )

    # Legitimate high-velocity users
    remaining = remaining.difference(value_idx)

    velocity_count = int(n * 0.20)
    velocity_idx = rng.choice(
        remaining,
        size=velocity_count,
        replace=False,
    )

    df.loc[velocity_idx, "velocity_1h"] += rng.integers(
        1, 5, velocity_count
    )

    df.loc[velocity_idx, "velocity_24h"] += rng.integers(
        2, 10, velocity_count
    )

    # Hard combined benign examples.
    # These intentionally combine the signals that the previous
    # detector over-relied upon.
    remaining = remaining.difference(velocity_idx)

    combined_idx = remaining

    df.loc[combined_idx, "distance_km"] = rng.uniform(
        40, 250, len(combined_idx)
    )

    df.loc[combined_idx, "amount"] *= rng.uniform(
        1.2, 4.0, len(combined_idx)
    )

    df.loc[combined_idx, "velocity_1h"] += rng.integers(
        1, 4, len(combined_idx)
    )

    df.loc[combined_idx, "velocity_24h"] += rng.integers(
        2, 8, len(combined_idx)
    )

    # Crucially, these are legitimate.
    df["is_fraud"] = 0

    return df


def build_training_set(seed=42):
    datasets = []

    # Normal baseline transactions
    datasets.append(
        generate_transactions(
            20000,
            seed,
        )
    )

    # Geographic fraud curriculum
    for level in range(1, 6):
        datasets.append(
            make_geo_dataset(
                n=5000,
                fraud_rate=0.05,
                seed=seed + level,
                level=level,
            )
        )

    # Stealth fraud
    datasets.append(
        make_stealth_fraud(
            n=20000,
            fraud_rate=0.05,
            seed=seed + 100,
        )
    )

    # Hard benign data
    datasets.append(
        make_hard_benign(
            n=20000,
            seed=seed + 200,
        )
    )

    return pd.concat(
        datasets,
        ignore_index=True,
    )


def train_model(df, seed=42):
    X = df[FEATURES]
    y = df["is_fraud"]

    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(X, y)

    return model


def evaluate(model, fraud_df, benign_df):
    fraud_scores = model.predict_proba(
        fraud_df[FEATURES]
    )[:, 1]

    benign_scores = model.predict_proba(
        benign_df[FEATURES]
    )[:, 1]

    y_true = np.concatenate([
        np.ones(len(fraud_df)),
        np.zeros(len(benign_df)),
    ])

    scores = np.concatenate([
        fraud_scores,
        benign_scores,
    ])

    predictions = (
        scores >= THRESHOLD
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        scores,
    )

    recall = recall_score(
        np.ones(len(fraud_df)),
        (fraud_scores >= THRESHOLD).astype(int),
        zero_division=0,
    )

    fpr = float(
        (benign_scores >= THRESHOLD).mean()
    )

    return auc, recall, fpr


if __name__ == "__main__":

    print("=== STAGE 15: CONTEXT-AWARE HARD-NEGATIVE REPAIR ===")
    print()

    training = build_training_set(SEED)

    print(
        f"Training transactions : {len(training)}"
    )

    print(
        f"Fraud transactions    : "
        f"{int(training['is_fraud'].sum())}"
    )

    print(
        f"Benign transactions   : "
        f"{int((training['is_fraud'] == 0).sum())}"
    )

    print()

    model = train_model(
        training,
        SEED,
    )

    # ---------------------------------------------------------
    # COMPLETELY UNTOUCHED STEALTH FRAUD
    # ---------------------------------------------------------

    stealth_holdout = make_stealth_fraud(
        n=12000,
        fraud_rate=0.05,
        seed=9001,
    )

    # ---------------------------------------------------------
    # COMPLETELY UNTOUCHED HARD BENIGN
    # ---------------------------------------------------------

    benign_holdout = make_hard_benign(
        n=12000,
        seed=9002,
    )

    auc, recall, fpr = evaluate(
        model,
        stealth_holdout,
        benign_holdout,
    )

    print("=== UNTOUCHED STEALTH FRAUD ===")
    print(
        f"Transactions : {len(stealth_holdout)}"
    )
    print(
        f"Fraud cases  : "
        f"{int(stealth_holdout['is_fraud'].sum())}"
    )
    print(
        f"ROC-AUC      : {auc:.4f}"
    )
    print(
        f"Recall @ {THRESHOLD:.2f} : {recall:.4f}"
    )

    print()

    print("=== UNTOUCHED HARD BENIGN ===")
    print(
        f"Transactions     : {len(benign_holdout)}"
    )
    print(
        f"False positives  : "
        f"{int((model.predict_proba(benign_holdout[FEATURES])[:,1] >= THRESHOLD).sum())}"
    )
    print(
        f"FPR              : {fpr:.4f}"
    )

    print()

    print("=== HARD GATE ===")
    print(
        f"AUC >= 0.90    : {auc >= 0.90}"
    )
    print(
        f"Recall >= 0.70 : {recall >= 0.70}"
    )
    print(
        f"FPR <= 0.05    : {fpr <= 0.05}"
    )

    print()

    passed = (
        auc >= 0.90
        and recall >= 0.70
        and fpr <= 0.05
    )

    if passed:
        print(
            "VERDICT: CONTEXT-AWARE HARD-NEGATIVE FIX PASS"
        )
    else:
        print(
            "VERDICT: CONTEXT-AWARE HARD-NEGATIVE FIX FAILED"
        )

    print()
    print("NO MODEL PROMOTION PERFORMED.")
