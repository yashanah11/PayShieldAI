import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier

from generation.generator import generate_transactions, FEATURES
from evaluation.redteam_geo_curriculum import make_geo_dataset
from evaluation.redteam_context_hard_negative_repair import (
    make_stealth_fraud,
    make_hard_benign,
)

SEED = 42
THRESHOLD = 0.10


def make_balanced_training(seed=42):
    """
    Controlled curriculum.

    Each stage contributes a controlled amount of fraud and benign data.
    Hard negatives are introduced only after the fraud representation
    has been established.
    """

    rng = np.random.default_rng(seed)

    datasets = []

    # ---------------------------------------------------------
    # PHASE 1 — BASELINE
    # ---------------------------------------------------------

    normal = generate_transactions(
        20000,
        seed=seed,
    )

    datasets.append(normal)

    # ---------------------------------------------------------
    # PHASE 2 — GEOGRAPHIC FRAUD
    # ---------------------------------------------------------

    geo = make_geo_dataset(
        n=12000,
        fraud_rate=0.10,
        seed=seed + 10,
        level=5,
    )

    datasets.append(geo)

    # ---------------------------------------------------------
    # PHASE 3 — STEALTH FRAUD
    # ---------------------------------------------------------

    stealth = make_stealth_fraud(
        n=12000,
        fraud_rate=0.10,
        seed=seed + 20,
    )

    datasets.append(stealth)

    # ---------------------------------------------------------
    # PHASE 4 — HARD BENIGN
    # ---------------------------------------------------------

    hard_benign = make_hard_benign(
        n=12000,
        seed=seed + 30,
    )

    # Down-weight the hard-negative volume by sampling.
    # We want hard negatives to challenge the model, not dominate it.

    hard_benign = hard_benign.sample(
        n=8000,
        random_state=seed,
    )

    datasets.append(hard_benign)

    training = pd.concat(
        datasets,
        ignore_index=True,
    )

    return training


def train_controlled_model(df, seed=42):

    X = df[FEATURES]
    y = df["is_fraud"]

    # Explicit class balancing.
    positives = int(y.sum())
    negatives = int((y == 0).sum())

    scale_pos_weight = negatives / max(
        positives,
        1,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,

        min_child_weight=5,

        reg_lambda=2.0,
        reg_alpha=0.1,

        scale_pos_weight=scale_pos_weight,

        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(
        X,
        y,
    )

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

    auc = roc_auc_score(
        y_true,
        scores,
    )

    fraud_predictions = (
        fraud_scores >= THRESHOLD
    ).astype(int)

    recall = recall_score(
        np.ones(len(fraud_df)),
        fraud_predictions,
        zero_division=0,
    )

    fpr = float(
        (
            benign_scores >= THRESHOLD
        ).mean()
    )

    return auc, recall, fpr


if __name__ == "__main__":

    print(
        "=== STAGE 15B: CONTROLLED CURRICULUM REPAIR ==="
    )

    print()

    training = make_balanced_training(
        SEED
    )

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

    model = train_controlled_model(
        training,
        SEED,
    )

    # =========================================================
    # COMPLETELY UNSEEN STEALTH FRAUD
    # =========================================================

    stealth_holdout = make_stealth_fraud(
        n=12000,
        fraud_rate=0.05,
        seed=7001,
    )

    # =========================================================
    # COMPLETELY UNSEEN HARD BENIGN
    # =========================================================

    benign_holdout = make_hard_benign(
        n=12000,
        seed=7002,
    )

    auc, recall, fpr = evaluate(
        model,
        stealth_holdout,
        benign_holdout,
    )

    print(
        "=== UNSEEN STEALTH FRAUD ==="
    )

    print(
        f"ROC-AUC      : {auc:.4f}"
    )

    print(
        f"Recall @ {THRESHOLD:.2f} : "
        f"{recall:.4f}"
    )

    print()

    print(
        "=== UNSEEN HARD BENIGN ==="
    )

    benign_scores = model.predict_proba(
        benign_holdout[FEATURES]
    )[:, 1]

    print(
        f"Transactions : {len(benign_holdout)}"
    )

    print(
        f"False positives : "
        f"{int((benign_scores >= THRESHOLD).sum())}"
    )

    print(
        f"FPR          : {fpr:.4f}"
    )

    print()

    print(
        "=== HARD GATE ==="
    )

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
            "VERDICT: CONTROLLED CURRICULUM REPAIR PASS"
        )

    else:

        print(
            "VERDICT: CONTROLLED CURRICULUM REPAIR FAILED"
        )

    print()
    print(
        "NO MODEL PROMOTION PERFORMED."
    )
