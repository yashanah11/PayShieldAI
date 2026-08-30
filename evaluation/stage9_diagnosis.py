"""
Stage 9: Diagnostic & Root Cause Analysis
PayShieldAI System

Investigates why specific red-team attacks fail by analyzing feature distributions,
model probability shifts, and XGBoost feature importances.
Does NOT modify any locked configurations or models.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generation.generator import generate_transactions
from evaluation.redteam_attack_families import get_redteam_attacks

V1_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_detector_retrained.joblib")
V3_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_detector_retrained_v3.joblib")
ENSEMBLE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "ensemble_config.joblib")
REPORT_PATH = os.path.join(PROJECT_ROOT, "evaluation", "stage9_diagnosis.json")

LOCKED_FEATURES = [
    "amount", "hour", "velocity_1h", "velocity_24h", 
    "device_age_days", "distance_km", "merchant_risk"
]

def load_models_and_config():
    v1 = joblib.load(V1_MODEL_PATH)
    v3 = joblib.load(V3_MODEL_PATH)
    config = joblib.load(ENSEMBLE_CONFIG_PATH)
    w1 = float(config.get("v1_weight"))
    threshold = float(config.get("threshold"))
    w3 = 1.0 - w1
    return v1, v3, w1, w3, threshold

def extract_feature_importance(v1, v3, w1, w3):
    """Extract and combine feature importances if available."""
    importance_dict = {}
    
    def get_fi(model):
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        return np.zeros(len(LOCKED_FEATURES))
        
    fi_v1 = get_fi(v1)
    fi_v3 = get_fi(v3)
    
    combined_fi = (w1 * fi_v1) + (w3 * fi_v3)
    
    for i, f in enumerate(LOCKED_FEATURES):
        importance_dict[f] = float(combined_fi[i])
        
    return importance_dict

def determine_failure_reason(auc, recall, median_prob, threshold, p90, main_feat_imp):
    """Heuristic mapping to diagnostic categories A-G."""
    reasons = []
    
    if auc < 0.50:
        reasons.append("C") # Model interprets attack as safer than benign
    if main_feat_imp < 0.05:
        reasons.append("B") # Weak model importance
    if median_prob < threshold and p90 >= threshold:
        reasons.append("E") # Threshold problem (tail is caught, median is missed)
    if 0.50 <= auc < 0.85:
        reasons.append("A") # Not sufficiently represented / partial generalization
    
    if not reasons:
        reasons.append("F") # General distribution problem
        
    return ", ".join(reasons)

def run_diagnosis():
    print("=" * 70)
    print("PAYSHIELD-AI: STAGE 9 DIAGNOSTIC & ROOT CAUSE ANALYSIS")
    print("=" * 70)

    v1, v3, w1, w3, threshold = load_models_and_config()
    feature_importance = extract_feature_importance(v1, v3, w1, w3)
    
    print("\n[INFO] Ensemble Feature Importance (Gain/Weight):")
    for f, imp in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
        print(f"  {f:<17}: {imp:.4f}")

    print("\n[INFO] Generating deterministic baseline (seed=42)...")
    test_df = generate_transactions(n=5000, seed=42)
    X_base = test_df[LOCKED_FEATURES].copy()
    
    attacks = get_redteam_attacks()
    
    FRAUD_RATIO = 0.05
    n_total = len(X_base)
    n_fraud = int(n_total * FRAUD_RATIO)
    
    diagnostic_report = {
        "ensemble_config": {"v1_weight": w1, "v3_weight": w3, "threshold": threshold},
        "feature_importance": feature_importance,
        "attacks": {}
    }
    
    summary_table = []

    for attack_name, attack_fn in attacks.items():
        print(f"\n[{attack_name.upper()}] Analyzing...")
        
        # 1. Exact Stage 9 Injection Protocol
        X_benign = X_base.iloc[n_fraud:].copy()
        X_target_orig = X_base.iloc[:n_fraud].copy()
        
        y_benign = np.zeros(len(X_benign), dtype=int)
        y_target = np.ones(n_fraud, dtype=int)
        
        # Apply Attack
        X_attacked, y_attacked = attack_fn(X_target_orig, y_target)
        X_attacked = X_attacked[LOCKED_FEATURES]
        
        # Evaluate Ensemble
        X_eval = pd.concat([X_benign, X_attacked], ignore_index=True)
        y_eval = np.concatenate([y_benign, y_attacked])
        
        p1 = v1.predict_proba(X_eval)[:, 1]
        p3 = v3.predict_proba(X_eval)[:, 1]
        p_ensemble = (w1 * p1) + (w3 * p3)
        y_pred = (p_ensemble >= threshold).astype(int)
        
        # Split probs back out for analysis
        prob_benign = p_ensemble[:len(X_benign)]
        prob_fraud = p_ensemble[len(X_benign):]
        
        # Determine what features actually changed
        changed_features = []
        feature_stats = {}
        for f in LOCKED_FEATURES:
            mean_benign = float(X_benign[f].mean())
            mean_fraud = float(X_attacked[f].mean())
            mean_orig_target = float(X_target_orig[f].mean())
            
            diff = mean_fraud - mean_orig_target
            
            if abs(diff) > 1e-5:  # Feature was altered by attack
                changed_features.append(f)
                
            feature_stats[f] = {
                "benign_mean": mean_benign,
                "attacked_mean": mean_fraud,
                "difference_from_orig": diff,
                "std": float(X_attacked[f].std()),
                "min": float(X_attacked[f].min()),
                "max": float(X_attacked[f].max()),
                "median": float(X_attacked[f].median())
            }

        # Calculate metrics
        auc = float(roc_auc_score(y_eval, p_ensemble))
        recall = float(recall_score(y_eval, y_pred, zero_division=0))
        
        percentiles = np.percentile(prob_fraud, [10, 25, 50, 75, 90])
        num_positive = int(np.sum(y_pred[len(X_benign):]))
        
        # Diagnostics
        main_imp = max([feature_importance.get(cf, 0) for cf in changed_features]) if changed_features else 0
        reason = determine_failure_reason(auc, recall, percentiles[2], threshold, percentiles[4], main_imp)
        
        cf_str = ", ".join(changed_features)
        summary_table.append((attack_name, auc, recall, cf_str, reason))
        
        # Print requested metrics to terminal
        print(f"  -> Changed Features: {cf_str}")
        print(f"  -> Prob Mean (Benign) : {float(np.mean(prob_benign)):.4f}")
        print(f"  -> Prob Mean (Fraud)  : {float(np.mean(prob_fraud)):.4f}")
        print(f"  -> Prob Median (Fraud): {percentiles[2]:.4f}")
        print(f"  -> Prob Percentiles   : P10={percentiles[0]:.4f} | P25={percentiles[1]:.4f} | P50={percentiles[2]:.4f} | P75={percentiles[3]:.4f} | P90={percentiles[4]:.4f}")
        print(f"  -> True Positives     : {num_positive} / {n_fraud} (Threshold {threshold})")
        print(f"  -> AUC / Recall       : {auc:.4f} / {recall:.4f}")
        
        # Store in JSON dictionary
        diagnostic_report["attacks"][attack_name] = {
            "changed_features": changed_features,
            "feature_stats": feature_stats,
            "prob_benign_mean": float(np.mean(prob_benign)),
            "prob_fraud_mean": float(np.mean(prob_fraud)),
            "prob_fraud_percentiles": {
                "P10": float(percentiles[0]), "P25": float(percentiles[1]),
                "P50": float(percentiles[2]), "P75": float(percentiles[3]),
                "P90": float(percentiles[4])
            },
            "true_positives": num_positive,
            "recall": recall,
            "auc": auc,
            "diagnostic_reason": reason
        }

    # Save JSON
    with open(REPORT_PATH, 'w') as f:
        json.dump(diagnostic_report, f, indent=4)

    # Print Final Summary Table
    print("\n" + "=" * 105)
    print(f"{'ATTACK':<20} | {'AUC':<6} | {'RECALL':<6} | {'MAIN CHANGED FEATURES':<35} | {'LIKELY REASON'}")
    print("-" * 105)
    for row in summary_table:
        print(f"{row[0]:<20} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:<35} | {row[4]}")
    
    print("-" * 105)
    print("DIAGNOSTIC KEYS:")
    print(" A: Feature not sufficiently represented during training")
    print(" B: Feature has weak model importance")
    print(" C: Attack changes a feature in a direction the model does not consider fraudulent")
    print(" D: Feature interaction problem")
    print(" E: Threshold problem (tail is caught, median is missed)")
    print(" F: Distribution/generalization problem")
    print(" G: Evaluation/protocol problem")
    print(f"\n[INFO] Detailed machine-readable report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    run_diagnosis()