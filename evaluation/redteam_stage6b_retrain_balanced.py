"""
STAGE 6b: BALANCED RETRAINING
- Use a smaller sample of evolved attacks (200) to reduce over-specialisation.
- Apply class weighting (scale_pos_weight) to balance standard fraud vs evolved attacks.
- Retrain with slightly more estimators and lower learning rate.
- Compare against v1 and v2 models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
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
    train_base = generate_stage2_repaired_benchmark(n=10000, fraud_rate=0.05, seed=42)
    test_base = generate_stage2_repaired_benchmark(n=5000, fraud_rate=0.05, seed=999)
    test_df = make_unseen_attacks(test_base, seed=12345)
    evolved = pd.read_csv("data/evolved_attacks.csv")
    evolved = evolved[FEATURES + ["is_fraud"]]
    return train_base, test_df, evolved


def train_model(train_df, seed=42, scale_pos_weight=None):
    X_train, _, y_train, _ = train_test_split(
        train_df[FEATURES],
        train_df["is_fraud"],
        test_size=0.30,
        random_state=seed,
        stratify=train_df["is_fraud"],
    )
    if scale_pos_weight is None:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, df, label="Dataset"):
    X = df[FEATURES]
    y = df["is_fraud"]
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else np.nan
    rec = recall_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    print(f"\n{label}:")
    print(f"  AUC:       {auc:.4f}" if not np.isnan(auc) else "  AUC:       N/A")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1:        {f1:.4f}")
    return {"auc": auc, "recall": rec, "precision": prec, "f1": f1}


if __name__ == "__main__":
    print("=== STAGE 6b: BALANCED RETRAINING ===")

    # Load data
    train_base, test_standard, evolved = load_data()
    print(f"Training base: {len(train_base)} rows")
    print(f"Standard test: {len(test_standard)} rows")
    print(f"Evolved attacks (full): {len(evolved)} rows")

    # Use a subset of evolved attacks (200) to avoid dominance
    n_evolved_sample = 200
    evolved_sample = evolved.sample(n=n_evolved_sample, random_state=42)
    print(f"Using {n_evolved_sample} evolved attacks for training (sampled).")

    # Combine
    train_augmented = pd.concat([train_base, evolved_sample], ignore_index=True)
    train_augmented = train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Augmented training size: {len(train_augmented)} rows")

    # Compute class ratio for scale_pos_weight
    neg = (train_augmented["is_fraud"] == 0).sum()
    pos = (train_augmented["is_fraud"] == 1).sum()
    scale = neg / pos if pos > 0 else 1.0
    print(f"Class ratio (neg/pos): {scale:.2f}")

    # Train new model (v3)
    print("\nTraining balanced model (v3)...")
    model_v3 = train_model(train_augmented, seed=42, scale_pos_weight=scale)
    print("Training complete.")

    # Evaluate
    print("\n--- Evaluation of v3 ---")
    metrics_standard_v3 = evaluate_model(model_v3, test_standard, "Standard Test (v3)")
    metrics_evolved_v3 = evaluate_model(model_v3, evolved, "Evolved Attacks (v3)")

    # Load v1 and v2 if available
    models = {}
    try:
        model_v1 = joblib.load("models/xgboost_detector_retrained.joblib")
        models["v1"] = model_v1
        print("Loaded v1 model.")
    except:
        print("v1 model not found.")
    try:
        model_v2 = joblib.load("models/xgboost_detector_retrained_v2.joblib")
        models["v2"] = model_v2
        print("Loaded v2 model.")
    except:
        print("v2 model not found.")

    if models:
        print("\n--- Comparison on Evolved Attacks ---")
        for name, m in models.items():
            evaluate_model(m, evolved, f"Model {name.upper()} on Evolved")
        print("\n--- Comparison on Standard Test ---")
        for name, m in models.items():
            evaluate_model(m, test_standard, f"Model {name.upper()} on Standard")

    # Save v3
    joblib.dump(model_v3, "models/xgboost_detector_retrained_v3.joblib")
    print("\nModel v3 saved to models/xgboost_detector_retrained_v3.joblib")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    rec_v3_evolved = metrics_evolved_v3["recall"]
    auc_v3_standard = metrics_standard_v3["auc"]

    if auc_v3_standard >= 0.85 and rec_v3_evolved >= 0.6:
        print("PASS: Standard AUC ≥ 0.85 and evolved recall ≥ 0.60.")
        print("       Balanced retraining succeeded.")
    else:
        if auc_v3_standard < 0.85:
            print(f"FAIL: Standard AUC = {auc_v3_standard:.4f} < 0.85")
        if rec_v3_evolved < 0.6:
            print(f"FAIL: Evolved recall = {rec_v3_evolved:.4f} < 0.60")
        print("       Consider further tuning or more evolved examples.")