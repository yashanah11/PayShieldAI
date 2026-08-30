"""
Stage 9: Red-Team Generalization & Robustness Evaluation
PayShieldAI System

Strictly evaluates the locked Stage 6C ensemble (v1 + v3) against
the 8 locked attack families using the authoritative 7-feature schema.

This version matches the current repository structure:
- Generator: generate_transactions(n=..., seed=...) outputs BENIGN baseline.
- Red-Team Protocol: Evaluator injects attacks into a subset of the baseline.
- v1 model: xgboost_detector_retrained.joblib
- v3 model: xgboost_detector_retrained_v3.joblib
- Ensemble config: ensemble_config.joblib
- Exactly 8 locked attack families
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generation.generator import generate_transactions
from evaluation.redteam_attack_families import get_redteam_attacks


# ============================================================
# LOCKED CONFIGURATION
# ============================================================

V1_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_detector_retrained.joblib"
)

V3_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_detector_retrained_v3.joblib"
)

ENSEMBLE_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ensemble_config.joblib"
)

# Authoritative 7-feature schema
LOCKED_FEATURES = [
    "amount",
    "hour",
    "velocity_1h",
    "velocity_24h",
    "device_age_days",
    "distance_km",
    "merchant_risk"
]


# ============================================================
# LOAD MODELS + CONFIGURATION
# ============================================================

def load_models_and_config():
    """Load the locked Stage 6C models and ensemble configuration."""

    if not os.path.exists(V1_MODEL_PATH):
        raise FileNotFoundError(f"v1 model not found at:\n{V1_MODEL_PATH}")

    if not os.path.exists(V3_MODEL_PATH):
        raise FileNotFoundError(f"v3 model not found at:\n{V3_MODEL_PATH}")

    if not os.path.exists(ENSEMBLE_CONFIG_PATH):
        raise FileNotFoundError(f"ensemble config not found at:\n{ENSEMBLE_CONFIG_PATH}")

    v1 = joblib.load(V1_MODEL_PATH)
    v3 = joblib.load(V3_MODEL_PATH)
    config = joblib.load(ENSEMBLE_CONFIG_PATH)

    if not isinstance(config, dict):
        raise ValueError("ensemble_config.joblib must contain a dictionary.")

    v1_weight = config.get("v1_weight")
    threshold = config.get("threshold")

    if v1_weight is None:
        raise ValueError("ensemble_config.joblib is missing 'v1_weight'.")
    if threshold is None:
        raise ValueError("ensemble_config.joblib is missing 'threshold'.")

    v3_weight = 1.0 - float(v1_weight)

    return v1, v3, float(v1_weight), float(v3_weight), float(threshold)


# ============================================================
# VERIFY MODEL SCHEMA
# ============================================================

def verify_model_schema(model, model_name):
    """Ensure model uses the authoritative 7-feature schema."""
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        if model_features != LOCKED_FEATURES:
            raise ValueError(
                f"\nCRITICAL SCHEMA MISMATCH\n"
                f"Model: {model_name}\n"
                f"Expected: {LOCKED_FEATURES}\n"
                f"Actual:   {model_features}\n"
            )
        print(f"[INFO] {model_name} feature schema verified.")
    else:
        print(f"[WARNING] {model_name} does not expose 'feature_names_in_'.")


# ============================================================
# MAIN STAGE 9 EVALUATION
# ============================================================

def run_stage9_evaluation():
    print("=" * 60)
    print("PAYSHIELD-AI: STAGE 9 GENERALIZATION & ROBUSTNESS EVALUATION")
    print("=" * 60)

    # 1. Load locked ensemble
    v1, v3, w1, w3, threshold = load_models_and_config()
    print(f"[INFO] Loaded Stage 6C Ensemble (v1 weight={w1:.2f}, v3 weight={w3:.2f}, threshold={threshold:.2f})")

    # 2. Verify model schemas
    verify_model_schema(v1, "v1")
    verify_model_schema(v3, "v3")

    # 3. Generate authoritative baseline dataset
    print("[INFO] Generating baseline evaluation transactions...")
    test_df = generate_transactions(n=5000, seed=42)

    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("generate_transactions() must return a pandas DataFrame.")

    # 4. Verify baseline dataset
    if "is_fraud" not in test_df.columns:
        raise ValueError("Generated dataset does not contain 'is_fraud'.")

    missing_base = [f for f in LOCKED_FEATURES if f not in test_df.columns]
    if missing_base:
        raise ValueError(f"Generated test dataset is missing canonical features: {missing_base}")

    X_base = test_df[LOCKED_FEATURES].copy()
    y_base = test_df["is_fraud"].to_numpy()

    print(f"[INFO] Baseline clean samples: {len(X_base)}")
    print(f"[INFO] Baseline generator fraud samples: {int(np.sum(y_base))} (Expected: 0)")

    # 5. Load exactly 8 attack families
    attacks = get_redteam_attacks()

    if not isinstance(attacks, dict):
        raise TypeError("get_redteam_attacks() must return a dictionary.")

    if len(attacks) != 8:
        raise ValueError(f"CRITICAL: Expected exactly 8 attack families, but received {len(attacks)}.")

    print("[INFO] Verified exactly 8 locked attack families loaded successfully.\n")

    # 6. Evaluate every attack family
    attack_results = []
    
    # Define attack ratio (e.g., inject 5% fraud into the dataset)
    FRAUD_RATIO = 0.05
    n_total = len(X_base)
    n_fraud = int(n_total * FRAUD_RATIO)

    for attack_number, (attack_name, attack_fn) in enumerate(attacks.items(), start=1):
        print(f"[{attack_number}/8] Evaluating: {attack_name}")
        try:
            # ------------------------------------------------
            # RED-TEAM INJECTION PROTOCOL
            # ------------------------------------------------
            
            # Isolate background traffic (untouched)
            X_benign = X_base.iloc[n_fraud:].copy()
            y_benign = y_base[n_fraud:].copy()
            
            # Isolate target traffic to be hijacked
            X_target = X_base.iloc[:n_fraud].copy()
            y_target = np.ones(n_fraud, dtype=int) # Inject actual fraud labels
            
            # Apply adversarial transformation ONLY to target subset
            X_attacked, y_attacked = attack_fn(X_target, y_target)

            if not isinstance(X_attacked, pd.DataFrame):
                raise TypeError(f"Attack '{attack_name}' did not return a pandas DataFrame.")
            
            y_attacked = np.asarray(y_attacked)

            # Recombine benign background with adversarial injected traffic
            X_eval = pd.concat([X_benign, X_attacked], ignore_index=True)
            y_eval = np.concatenate([y_benign, y_attacked])

            if len(X_eval) != len(y_eval):
                raise ValueError(f"Attack '{attack_name}' length mismatch after recombination.")

            # Strict feature validation
            missing_adv = [f for f in LOCKED_FEATURES if f not in X_eval.columns]
            if missing_adv:
                raise ValueError(f"Attack '{attack_name}' generated missing locked features: {missing_adv}")

            # Select canonical features in exact order
            X_eval_canonical = X_eval[LOCKED_FEATURES].copy()
            if list(X_eval_canonical.columns) != LOCKED_FEATURES:
                raise ValueError(f"Attack '{attack_name}' feature order does not match locked schema.")

            # ------------------------------------------------
            # Ensemble prediction
            # ------------------------------------------------
            p1 = v1.predict_proba(X_eval_canonical)[:, 1]
            p3 = v3.predict_proba(X_eval_canonical)[:, 1]
            p_ensemble = (w1 * p1) + (w3 * p3)

            # Locked threshold
            y_pred = (p_ensemble >= threshold).astype(int)

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------
            unique_labels = np.unique(y_eval)
            if len(unique_labels) > 1:
                auc = roc_auc_score(y_eval, p_ensemble)
            else:
                auc = 0.5

            recall = recall_score(y_eval, y_pred, zero_division=0)

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------
            attack_results.append({
                "attack_family": attack_name,
                "auc": float(auc),
                "recall": float(recall),
                "samples": int(len(y_eval))
            })

            print(f"    AUC    : {auc:.4f}")
            print(f"    Recall : {recall:.4f}")
            print(f"    Samples: {len(y_eval)} (Benign: {len(y_benign)}, Fraud: {len(y_attacked)})\n")

        except Exception as e:
            print(f"[ERROR] Attack '{attack_name}' failed:\n        {e}")
            raise

    # 7. Aggregate Stage 9 results
    results_df = pd.DataFrame(attack_results)
    mean_auc = results_df["auc"].mean()
    worst_auc = results_df["auc"].min()
    mean_recall = results_df["recall"].mean()
    worst_recall = results_df["recall"].min()

    # 8. Print attack table
    print("=" * 60)
    print("STAGE 9 ATTACK RESULTS")
    print("=" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['attack_family']:<25} AUC={row['auc']:.4f} Recall={row['recall']:.4f}")

    # 9. Global summary
    print("-" * 60)
    print(f"Global Mean AUC     : {mean_auc:.4f}")
    print(f"Worst Attack AUC    : {worst_auc:.4f}")
    print(f"Global Mean Recall  : {mean_recall:.4f}")
    print(f"Worst Attack Recall : {worst_recall:.4f}")
    print("-" * 60)

    # 10. Stage 9 gates
    mean_auc_gate = mean_auc >= 0.85
    worst_auc_gate = worst_auc >= 0.80
    mean_recall_gate = mean_recall >= 0.55

    print(f"Gate [Mean AUC >= 0.85]       : {'PASS' if mean_auc_gate else 'FAIL'} ({mean_auc:.4f})")
    print(f"Gate [Worst AUC >= 0.80]      : {'PASS' if worst_auc_gate else 'FAIL'} ({worst_auc:.4f})")
    print(f"Gate [Mean Recall >= 0.55]    : {'PASS' if mean_recall_gate else 'FAIL'} ({mean_recall:.4f})")

    # 11. Final verdict
    stage9_pass = mean_auc_gate and worst_auc_gate and mean_recall_gate
    print("=" * 60)
    if stage9_pass:
        print("VERDICT: STAGE 9 GENERALIZATION PASS")
    else:
        print("VERDICT: STAGE 9 GENERALIZATION FAIL")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_stage9_evaluation()