"""
STAGE 4: FULL EVALUATION
- Train a model on the repaired benchmark (includes all attack families).
- Evaluate on a large test set containing all attack types mixed.
- Compute AUC, precision, recall, F1 at 0.5 threshold and optimal threshold.
- Print verdict.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
)
from xgboost import XGBClassifier

from generation.generator import FEATURES
from evaluation.hard_benchmark_stage2_repaired import (
    generate_stage2_repaired_benchmark,
)
from evaluation.redteam_stage2c_retest import make_unseen_attacks


def generate_test_set(n=10000, fraud_rate=0.05, seed=999):
    """Create a test set that includes all attack families mixed."""
    base = generate_stage2_repaired_benchmark(n=n, fraud_rate=fraud_rate, seed=seed)
    # Apply the same unseen attack transformation (mixes all three families)
    test_df = make_unseen_attacks(base, seed=seed+1)
    return test_df


def evaluate_model(model, df):
    """Compute all metrics and return as dict."""
    X = df[FEATURES]
    y = df["is_fraud"]
    
    probs = model.predict_proba(X)[:, 1]
    preds_05 = (probs >= 0.5).astype(int)
    
    # Optimal threshold (maximising F1)
    prec, rec, thresh = precision_recall_curve(y, probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresh[best_idx] if best_idx < len(thresh) else 0.5
    preds_opt = (probs >= best_thresh).astype(int)
    
    metrics = {
        "auc": roc_auc_score(y, probs),
        "threshold_05": {
            "precision": precision_score(y, preds_05, zero_division=0),
            "recall": recall_score(y, preds_05, zero_division=0),
            "f1": f1_score(y, preds_05, zero_division=0),
            "confusion": confusion_matrix(y, preds_05).tolist(),
        },
        "optimal_threshold": {
            "threshold": best_thresh,
            "precision": precision_score(y, preds_opt, zero_division=0),
            "recall": recall_score(y, preds_opt, zero_division=0),
            "f1": f1_score(y, preds_opt, zero_division=0),
            "confusion": confusion_matrix(y, preds_opt).tolist(),
        },
    }
    return metrics


def train_model(seed=42):
    """Train XGBoost on repaired benchmark (same as Stage 2C)."""
    train_df = generate_stage2_repaired_benchmark(n=10000, fraud_rate=0.05, seed=seed)
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


if __name__ == "__main__":
    print("=== STAGE 4: FULL EVALUATION ===")
    
    # Train model (you can also load a saved one if preferred)
    print("\nTraining model on repaired benchmark...")
    model = train_model(seed=42)
    print("Training complete.\n")
    
    # Generate test set (larger and mixed)
    print("Generating test set...")
    test_df = generate_test_set(n=10000, fraud_rate=0.05, seed=999)
    print(f"Test set size: {len(test_df)} rows")
    print(f"Fraud cases: {int(test_df['is_fraud'].sum())}\n")
    
    # Evaluate
    metrics = evaluate_model(model, test_df)
    
    # Print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"ROC-AUC: {metrics['auc']:.4f}")
    print()
    print("--- At Threshold 0.5 ---")
    t05 = metrics["threshold_05"]
    print(f"Precision: {t05['precision']:.4f}")
    print(f"Recall:    {t05['recall']:.4f}")
    print(f"F1:        {t05['f1']:.4f}")
    print(f"Confusion Matrix: {t05['confusion']}")
    print()
    print("--- At Optimal Threshold (max F1) ---")
    opt = metrics["optimal_threshold"]
    print(f"Threshold: {opt['threshold']:.4f}")
    print(f"Precision: {opt['precision']:.4f}")
    print(f"Recall:    {opt['recall']:.4f}")
    print(f"F1:        {opt['f1']:.4f}")
    print(f"Confusion Matrix: {opt['confusion']}")
    print("=" * 60)
    
    # Verdict
    print("\nVERDICT:")
    auc_gate = 0.85   # stricter than Stage 2 (which was 0.80)
    recall_gate = 0.60 # at optimal threshold, we expect at least 60% recall
    
    if metrics["auc"] >= auc_gate and opt["recall"] >= recall_gate:
        print(f"PASS (AUC={metrics['auc']:.4f} ≥ {auc_gate}, Recall={opt['recall']:.4f} ≥ {recall_gate})")
    else:
        print(f"FAIL (AUC={metrics['auc']:.4f} < {auc_gate} or Recall={opt['recall']:.4f} < {recall_gate})")