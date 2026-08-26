import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier

from generation.generator import generate_transactions
from generation.generator import FEATURES
from generation.hard_negatives import (
    generate_legitimate_travelers,
)


def make_geo_dataset(
    n=10000,
    fraud_rate=0.05,
    seed=42,
    level=3,
):
    rng = np.random.default_rng(seed)

    df = generate_transactions(
        n,
        seed,
    )

    fraud_count = int(
        n * fraud_rate
    )

    indices = rng.choice(
        df.index,
        size=fraud_count,
        replace=False,
    )

    df["is_fraud"] = 0

    if level == 1:
        distance = rng.uniform(
            200,
            800,
            fraud_count,
        )

    elif level == 2:
        distance = rng.uniform(
            100,
            300,
            fraud_count,
        )

    elif level == 3:
        distance = rng.uniform(
            50,
            150,
            fraud_count,
        )

    elif level == 4:
        distance = rng.uniform(
            30,
            100,
            fraud_count,
        )

    else:
        distance = rng.uniform(
            20,
            120,
            fraud_count,
        )

    df.loc[
        indices,
        "distance_km"
    ] += distance

    # Geographic fraud deliberately keeps
    # several other signals ordinary.
    df.loc[
        indices,
        "merchant_risk"
    ] = rng.uniform(
        0.2,
        0.6,
        fraud_count,
    )

    df.loc[
        indices,
        "device_age_days"
    ] = rng.integers(
        100,
        1200,
        fraud_count,
    )

    df.loc[
        indices,
        "velocity_1h"
    ] = rng.poisson(
        1.5,
        fraud_count,
    )

    df.loc[
        indices,
        "velocity_24h"
    ] = rng.poisson(
        5,
        fraud_count,
    )

    df.loc[
        indices,
        "is_fraud"
    ] = 1

    return df


def build_curriculum(seed=42):
    datasets = []

    # ---------------------------------
    # Geographic fraud curriculum
    # ---------------------------------

    for level in range(1, 6):

        datasets.append(
            make_geo_dataset(
                n=4000,
                fraud_rate=0.05,
                seed=seed + level,
                level=level,
            )
        )

    # ---------------------------------
    # HARD NEGATIVES
    #
    # Legitimate transactions that
    # resemble geographic fraud.
    # ---------------------------------

    datasets.append(
        generate_legitimate_travelers(
            n=10000,
            seed=seed + 100,
        )
    )

    return datasets


def train_model(
    df,
    seed=42,
):
    X = df[FEATURES]
    y = df["is_fraud"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(
        X,
        y,
    )

    return model


if __name__ == "__main__":

    print(
        "=== STAGE 10: "
        "CONTEXT-AWARE GEOGRAPHIC CURRICULUM ==="
    )

    curriculum = build_curriculum()

    training = pd.concat(
        curriculum,
        ignore_index=True,
    )

    print(
        f"Training transactions: "
        f"{len(training)}"
    )

    print(
        f"Fraud transactions: "
        f"{int(training['is_fraud'].sum())}"
    )

    print(
        f"Legitimate transactions: "
        f"{int((training['is_fraud'] == 0).sum())}"
    )

    model = train_model(
        training
    )

    # ---------------------------------
    # UNSEEN GEOGRAPHIC HOLDOUT
    # ---------------------------------

    print()
    print(
        "=== UNSEEN GEOGRAPHIC HOLDOUT ==="
    )

    holdout = make_geo_dataset(
        n=10000,
        fraud_rate=0.05,
        seed=9999,
        level=5,
    )

    probabilities = model.predict_proba(
        holdout[FEATURES]
    )[:, 1]

    predictions = (
        probabilities >= 0.5
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

    print(
        f"Holdout transactions : "
        f"{len(holdout)}"
    )

    print(
        f"Fraud cases          : "
        f"{int(holdout['is_fraud'].sum())}"
    )

    print(
        f"ROC-AUC              : "
        f"{auc:.4f}"
    )

    print(
        f"Recall               : "
        f"{recall:.4f}"
    )

    # ---------------------------------
    # BENIGN TRAVELER HOLDOUT
    # ---------------------------------

    print()
    print(
        "=== BENIGN TRAVELER HOLDOUT ==="
    )

    benign = generate_legitimate_travelers(
        n=10000,
        seed=8888,
    )

    benign_probabilities = model.predict_proba(
        benign[FEATURES]
    )[:, 1]

    benign_predictions = (
        benign_probabilities >= 0.28
    ).astype(int)

    false_positive_rate = (
        benign_predictions.mean()
    )

    print(
        f"Transactions       : "
        f"{len(benign)}"
    )

    print(
        f"False positives    : "
        f"{int(benign_predictions.sum())}"
    )

    print(
        f"False-positive rate: "
        f"{false_positive_rate:.4f}"
    )

    print()

    if (
        auc >= 0.80
        and recall >= 0.50
        and false_positive_rate <= 0.10
    ):
        print(
            "VERDICT: "
            "CONTEXT-AWARE FIX PASSED"
        )
    else:
        print(
            "VERDICT: "
            "CONTEXT-AWARE FIX FAILED"
        )
