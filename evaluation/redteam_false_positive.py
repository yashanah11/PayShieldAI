import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from generation.generator import generate_transactions
from generation.generator import FEATURES

from evaluation.redteam_geo_curriculum import (
    build_curriculum,
    train_model,
)


def make_benign_stress_set(
    n=10000,
    seed=4242,
):
    rng = np.random.default_rng(seed)

    df = generate_transactions(
        n,
        seed,
    )

    # Legitimate travelers:
    # suspicious-looking, but NOT fraud.

    traveler_count = int(n * 0.20)

    indices = rng.choice(
        df.index,
        size=traveler_count,
        replace=False,
    )

    # Large geographic movement
    df.loc[indices, "distance_km"] += rng.uniform(
        50,
        250,
        traveler_count,
    )

    # Some large purchases
    df.loc[indices, "amount"] *= rng.uniform(
        1.5,
        5.0,
        traveler_count,
    )

    # New-ish devices
    df.loc[indices, "device_age_days"] = rng.integers(
        10,
        180,
        traveler_count,
    )

    # Some unusual transaction times
    df.loc[indices, "hour"] = rng.choice(
        [0, 1, 2, 3, 4, 23],
        size=traveler_count,
    )

    # Moderate merchant risk,
    # but not obviously fraudulent.
    df.loc[indices, "merchant_risk"] = rng.uniform(
        0.15,
        0.55,
        traveler_count,
    )

    # Keep all of them legitimate.
    df["is_fraud"] = 0

    return df


def make_true_fraud_set(
    n=10000,
    seed=5151,
):
    rng = np.random.default_rng(seed)

    df = generate_transactions(
        n,
        seed,
    )

    fraud_count = int(n * 0.05)

    indices = rng.choice(
        df.index,
        size=fraud_count,
        replace=False,
    )

    df.loc[indices, "velocity_1h"] += rng.integers(
        3,
        10,
        fraud_count,
    )

    df.loc[indices, "velocity_24h"] += rng.integers(
        5,
        20,
        fraud_count,
    )

    df.loc[indices, "device_age_days"] = rng.integers(
        1,
        30,
        fraud_count,
    )

    df.loc[indices, "distance_km"] += rng.uniform(
        50,
        500,
        fraud_count,
    )

    df.loc[indices, "merchant_risk"] = rng.uniform(
        0.7,
        1.0,
        fraud_count,
    )

    df["is_fraud"] = 0
    df.loc[indices, "is_fraud"] = 1

    return df


if __name__ == "__main__":
    print("=== STAGE 7: FALSE-POSITIVE STRESS TEST ===")

    # Train only on the existing curriculum.
    curriculum = build_curriculum(
        seed=42,
    )

    training = pd.concat(
        curriculum,
        ignore_index=True,
    )

    model = train_model(
        training,
    )

    threshold = 0.28

    benign = make_benign_stress_set()

    probabilities = model.predict_proba(
        benign[FEATURES]
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    false_positive_rate = predictions.mean()

    print()
    print("BENIGN ADVERSARIAL TRANSACTIONS")
    print(f"Transactions       : {len(benign)}")
    print(f"False positives    : {int(predictions.sum())}")
    print(
        f"False-positive rate: "
        f"{false_positive_rate:.4f}"
    )

    # Now evaluate a mixed dataset.
    fraud = make_true_fraud_set()

    mixed = pd.concat(
        [benign, fraud],
        ignore_index=True,
    )

    probabilities = model.predict_proba(
        mixed[FEATURES]
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    y = mixed["is_fraud"]

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        y,
        probabilities,
    )

    print()
    print("MIXED BENIGN + FRAUD")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    print()
    print("CONFUSION MATRIX")
    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    print()

    if false_positive_rate <= 0.10:
        print("VERDICT: FALSE-POSITIVE CONTROL PASS")
    else:
        print(
            "VERDICT: FALSE-POSITIVE CONTROL FAILED"
        )
