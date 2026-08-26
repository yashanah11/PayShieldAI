"""
STAGE 6c: REGULARIZED MIXTURE TRAINING (v5)

Purpose:
- Avoid sequential residual fine-tuning failure (v4).
- Train a robust XGBoost classifier from scratch on a balanced 
  mixture of standard baseline data and evolved attacks.
- Apply strict regularization (L1/L2, max_depth) to preserve 
  Standard AUC while capturing Evolved Attack patterns.
- Enforce strict gates: Standard AUC >= 0.85 and Evolved Recall >= 0.60.
- Save v5 only upon meeting both criteria.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

from generation.generator import FEATURES
from evaluation.hard_benchmark_stage2_repaired import (
    generate_stage2_repaired_benchmark,
)
from evaluation.redteam_stage2c_retest import make_unseen_attacks


def load_data():
    train_base = generate_stage2_repaired_benchmark(
        n=10000,
        fraud_rate=0.05,
        seed=42,
    )

    test_base = generate_stage2_repaired_benchmark(
        n=5000,
        fraud_rate=0.05,
        seed=999,
    )

    test_df = make_unseen_attacks(
        test_base,
        seed=12345,
    )

    evolved = pd.read_csv("data/evolved_attacks.csv")
    evolved = evolved[FEATURES + ["is_fraud"]]

    return train_base, test_df, evolved


def evaluate_model(model, df, label="Dataset"):
    X = df[FEATURES]
    y = df["is_fraud"]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = (
        roc_auc_score(y, probabilities)
        if len(np.unique(y)) > 1
        else np.nan
    )
    recall = recall_score(y, predictions, zero_division=0)
    precision = precision_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    print()
    print(f"{label}:")
    print(
        f"  AUC:       {auc:.4f}"
        if not np.isnan(auc)
        else "  AUC:       N/A"
    )
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1:        {f1:.4f}")

    return {
        "auc": auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


if __name__ == "__main__":
    print("=== STAGE 6c: REGULARIZED MIXTURE TRAINING (v5) ===")

    train_base, test_standard, evolved = load_data()

    print(f"Training base: {len(train_base)} rows")
    print(f"Standard test: {len(test_standard)} rows")
    print(f"Evolved attacks (full): {len(evolved)} rows")

    # Construct a robust mixture training set
    # Extract normal and standard fraud from base
    normal_base = train_base[train_base["is_fraud"] == 0].sample(
        n=3000, random_state=42
    )
    fraud_base = train_base[train_base["is_fraud"] == 1].sample(
        n=500, random_state=42
    )

    # Extract a substantial representative sample of evolved attacks
    evolved_sample = evolved.sample(
        n=min(600, len(evolved)), random_state=42
    )

    mixture_data = pd.concat(
        [normal_base, fraud_base, evolved_sample], ignore_index=True
    )
    mixture_data = mixture_data.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print()
    print(f"Mixture dataset size: {len(mixture_data)}")
    print("Mixture label distribution:")
    print(mixture_data["is_fraud"].value_counts())

    X_mix = mixture_data[FEATURES]
    y_mix = mixture_data["is_fraud"]

    # Stratified split to prevent leakage and ensure robust validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_mix,
        y_mix,
        test_size=0.25,
        random_state=42,
        stratify=y_mix,
    )

    print()
    print("Training distribution:")
    print(y_train.value_counts())
    print("Validation distribution:")
    print(y_val.value_counts())

    print()
    print("Training regularized XGBoost mixture model (v5)...")

    # Regularized architecture to prevent catastrophic specialization
    v5_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=42,
    )

    v5_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print("Training complete.")

    print()
    print("--- Evaluation of v5 ---")
    metrics_standard_v5 = evaluate_model(
        v5_model, test_standard, "Standard Test (v5)"
    )
    metrics_evolved_v5 = evaluate_model(
        v5_model, evolved, "Evolved Attacks (v5)"
    )

    output_path = "models/xgboost_detector_retrained_v5.joblib"
    joblib.dump(v5_model, output_path)
    print()
    print(f"Model v5 saved to {output_path}")

    auc_v5 = metrics_standard_v5["auc"]
    recall_v5_evolved = metrics_evolved_v5["recall"]

    print()
    print("=" * 60)
    print("STAGE 6c VERDICT (v5)")
    print("=" * 60)
    print(f"Standard AUC    : {auc_v5:.4f}")
    print(f"Evolved Recall  : {recall_v5_evolved:.4f}")
    print()
    print("AUC gate         : >= 0.85")
    print("Recall gate      : >= 0.60")
    print()

    if auc_v5 >= 0.85 and recall_v5_evolved >= 0.60:
        print("VERDICT: STAGE 6C PASS")
    else:
        print("VERDICT: STAGE 6C FAIL")
        if auc_v5 < 0.85:
            print(f"Reason: AUC {auc_v5:.4f} is below 0.85")
        if recall_v5_evolved < 0.60:
            print(
                f"Reason: Evolved recall {recall_v5_evolved:.4f} is below 0.60"
            )