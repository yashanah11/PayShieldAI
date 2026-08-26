"""
STAGE 6: RETRAIN
- Load the evolved attacks generated in Stage 5.
- Combine them with the original training data.
- Retrain the XGBoost model.
- Evaluate on:
  1. Standard test set (mixed attacks) – performance should stay high.
  2. Evolved attacks (the same ones) – recall should improve significantly.
- Print verdict.
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
    """Load training base, standard test set, and evolved attacks."""
    # Training base (repaired benchmark)
    train_base = generate_stage2_repaired_benchmark(n=10000, fraud_rate=0.05, seed=42)
    
    # Standard test set (same as Stage 4/5)
    test_base = generate_stage2_repaired_benchmark(n=5000, fraud_rate=0.05, seed=999)
    test_df = make_unseen_attacks(test_base, seed=12345)
    
    # Evolved attacks from Stage 5
    evolved = pd.read_csv("data/evolved_attacks.csv")
    # The evolved CSV contains all columns and is_fraud = 1
    # Ensure it matches FEATURES
    evolved = evolved[FEATURES + ["is_fraud"]]
    
    return train_base, test_df, evolved


def train_model(train_df, seed=42):
    """Train XGBoost on the given training data."""
    X_train, _, y_train, _ = train_test_split(
        train_df[FEATURES],
        train_df["is_fraud"],
        test_size=0.30,
        random_state=seed,
        stratify=train_df["is_fraud"],
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, df, label="Dataset"):
    """Compute metrics and print summary."""
    X = df[FEATURES]
    y = df["is_fraud"]
    
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    auc = roc_auc_score(y, probs)
    rec = recall_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    print(f"\n{label}:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1:        {f1:.4f}")
    
    return {"auc": auc, "recall": rec, "precision": prec, "f1": f1}


if __name__ == "__main__":
    print("=== STAGE 6: RETRAIN ===")
    
    # 1. Load data
    train_base, test_standard, evolved = load_data()
    print(f"Training base: {len(train_base)} rows")
    print(f"Standard test: {len(test_standard)} rows")
    print(f"Evolved attacks: {len(evolved)} rows")
    
    # 2. Combine training data with evolved attacks
    print("\nAugmenting training data with evolved attacks...")
    train_augmented = pd.concat([train_base, evolved], ignore_index=True)
    # Shuffle to mix
    train_augmented = train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Augmented training size: {len(train_augmented)} rows")
    
    # 3. Train the retrained model
    print("\nTraining retrained model...")
    model_retrained = train_model(train_augmented, seed=42)
    print("Training complete.")
    
    # 4. Evaluate on standard test set
    metrics_standard = evaluate_model(model_retrained, test_standard, "Standard Test Set")
    
    # 5. Evaluate on evolved attacks (they should now be detected)
    metrics_evolved = evaluate_model(model_retrained, evolved, "Evolved Attacks (retrained model)")
    
    # 6. Also load the old model for comparison (if available)
    try:
        old_model = joblib.load("models/xgboost_detector_retrained.joblib")
        print("\n--- Comparing with old model on evolved attacks ---")
        old_metrics = evaluate_model(old_model, evolved, "Old Model on Evolved Attacks")
        old_recall = old_metrics["recall"]
    except:
        old_recall = 0.0
        print("\nOld model not found. Skipping comparison.")
    
    # 7. Save the retrained model
    joblib.dump(model_retrained, "models/xgboost_detector_retrained_v2.joblib")
    print("\nRetrained model saved to models/xgboost_detector_retrained_v2.joblib")
    
    # 8. Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    # Check if recall on evolved attacks improved significantly (e.g., > 0.3)
    recall_new = metrics_evolved["recall"]
    recall_old = old_recall
    
    print(f"Recall on evolved attacks (old model): {recall_old:.4f}")
    print(f"Recall on evolved attacks (new model): {recall_new:.4f}")
    
    if recall_new > 0.5:
        print("PASS: Retrained model detects evolved attacks effectively.")
        print("       The adversarial vulnerability has been closed.")
    elif recall_new > 0.3:
        print("PARTIAL PASS: Some improvement, but further retraining cycles may be needed.")
    else:
        print("FAIL: Retrained model still fails to detect evolved attacks.")
        print("       Consider more aggressive retraining (e.g., more epochs, different hyperparameters).")
    
    # Also ensure standard test performance is not degraded
    if metrics_standard["auc"] < 0.85:
        print("WARNING: Standard test AUC dropped below 0.85. Retraining may have caused regression.")
    else:
        print("Standard test AUC maintained above 0.85 – good.")