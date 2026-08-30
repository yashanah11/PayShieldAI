"""
Stage 10: v6_robust Generalization & Red-Team Evaluation
PayShieldAI System

Evaluates the newly trained v6_robust model against the exact
8 locked Stage 9 red-team attack families.

IMPORTANT:
- Does NOT modify v1
- Does NOT modify v3
- Does NOT modify ensemble_config.joblib
- Does NOT modify the baseline generator
- Uses the same 7-feature schema
- Uses the same Stage 9 injection protocol
- Uses threshold 0.30 for direct comparison with Stage 9
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score, precision_score

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from generation.generator import generate_transactions
from evaluation.redteam_attack_families import get_redteam_attacks


# ============================================================
# MODEL PATH
# ============================================================

V6_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_detector_v6_robust.joblib"
)


# ============================================================
# LOCKED 7-FEATURE SCHEMA
# ============================================================

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
# STAGE 9 LOCKED EVALUATION PARAMETERS
# ============================================================

N_BASELINE = 5000
FRAUD_RATIO = 0.05
SEED = 42

# Same threshold used by the Stage 6C ensemble
THRESHOLD = 0.30


# ============================================================
# LOAD V6 MODEL
# ============================================================

def load_v6_model():

    print("[INFO] Loading v6_robust model...")

    if not os.path.exists(V6_MODEL_PATH):
        raise FileNotFoundError(
            f"v6_robust model not found at:\n{V6_MODEL_PATH}"
        )

    model = joblib.load(V6_MODEL_PATH)

    print("[SUCCESS] v6_robust loaded.")

    return model


# ============================================================
# VERIFY MODEL SCHEMA
# ============================================================

def verify_model_schema(model):

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

        if model_features != LOCKED_FEATURES:

            raise ValueError(
                "\nCRITICAL FEATURE SCHEMA MISMATCH\n"
                f"Model features : {model_features}\n"
                f"Required       : {LOCKED_FEATURES}"
            )

        print("[INFO] v6 feature schema verified.")

    else:

        print(
            "[WARNING] Model does not expose feature_names_in_. "
            "Schema verification skipped."
        )


# ============================================================
# RUN STAGE 10 GENERALIZATION
# ============================================================

def run_stage10_v6_evaluation():

    print("=" * 70)
    print("PAYSHIELD-AI: STAGE 10 v6_ROBUST GENERALIZATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD MODEL
    # --------------------------------------------------------

    model = load_v6_model()

    verify_model_schema(model)

    # --------------------------------------------------------
    # 2. GENERATE CLEAN BASELINE
    # --------------------------------------------------------

    print()
    print("[INFO] Generating deterministic clean baseline...")
    print(f"[INFO] Samples: {N_BASELINE}")
    print(f"[INFO] Seed   : {SEED}")

    test_df = generate_transactions(
        n=N_BASELINE,
        seed=SEED
    )

    if not isinstance(test_df, pd.DataFrame):
        raise TypeError(
            "generate_transactions() did not return a pandas DataFrame."
        )

    print(f"[INFO] Baseline samples: {len(test_df)}")

    # Generator is intentionally clean
    if "is_fraud" in test_df.columns:

        generator_fraud = int(
            test_df["is_fraud"].sum()
        )

        print(
            f"[INFO] Generator fraud samples: "
            f"{generator_fraud} (Expected: 0)"
        )

        if generator_fraud != 0:
            raise ValueError(
                "CRITICAL: Baseline generator is not clean. "
                "Expected zero fraud labels."
            )

    # --------------------------------------------------------
    # 3. VERIFY FEATURES
    # --------------------------------------------------------

    missing_features = [
        f for f in LOCKED_FEATURES
        if f not in test_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Baseline missing locked features: {missing_features}"
        )

    X_base = test_df[LOCKED_FEATURES].copy()

    # --------------------------------------------------------
    # 4. CREATE FRAUD INJECTION TARGET
    # --------------------------------------------------------

    n_total = len(X_base)

    n_fraud = int(
        n_total * FRAUD_RATIO
    )

    n_benign = n_total - n_fraud

    print()
    print("[INFO] Stage 9 injection protocol")
    print(f"[INFO] Total samples       : {n_total}")
    print(f"[INFO] Fraud samples       : {n_fraud}")
    print(f"[INFO] Benign samples      : {n_benign}")
    print(f"[INFO] Fraud ratio         : {FRAUD_RATIO:.2%}")

    # --------------------------------------------------------
    # 5. LOAD EXACT 8 LOCKED ATTACKS
    # --------------------------------------------------------

    attacks = get_redteam_attacks()

    if len(attacks) != 8:

        raise ValueError(
            f"CRITICAL: Expected exactly 8 attack families, "
            f"but found {len(attacks)}."
        )

    print()
    print(
        "[INFO] Verified exactly 8 locked attack families."
    )

    # --------------------------------------------------------
    # 6. EVALUATE EACH ATTACK
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("STAGE 10 v6 RED-TEAM ATTACK RESULTS")
    print("=" * 70)

    for attack_number, (attack_name, attack_fn) in enumerate(
        attacks.items(),
        start=1
    ):

        print()
        print(
            f"[{attack_number}/8] Evaluating: {attack_name}"
        )

        # ----------------------------------------------------
        # BENIGN BACKGROUND
        # ----------------------------------------------------

        X_benign = X_base.iloc[n_fraud:].copy()

        y_benign = np.zeros(
            len(X_benign),
            dtype=int
        )

        # ----------------------------------------------------
        # FRAUD TARGET
        # ----------------------------------------------------

        X_target = X_base.iloc[:n_fraud].copy()

        y_target = np.ones(
            n_fraud,
            dtype=int
        )

        # ----------------------------------------------------
        # APPLY RED-TEAM ATTACK
        # ----------------------------------------------------

        X_attacked, y_attacked = attack_fn(
            X_target.copy(),
            y_target.copy()
        )

        # ----------------------------------------------------
        # VERIFY ATTACK OUTPUT
        # ----------------------------------------------------

        if not isinstance(
            X_attacked,
            pd.DataFrame
        ):

            X_attacked = pd.DataFrame(
                X_attacked,
                columns=LOCKED_FEATURES
            )

        missing_attack_features = [
            f for f in LOCKED_FEATURES
            if f not in X_attacked.columns
        ]

        if missing_attack_features:

            raise ValueError(
                f"Attack '{attack_name}' is missing "
                f"features: {missing_attack_features}"
            )

        X_attacked = X_attacked[
            LOCKED_FEATURES
        ].copy()

        y_attacked = np.asarray(
            y_attacked,
            dtype=int
        )

        if len(X_attacked) != n_fraud:

            raise ValueError(
                f"Attack '{attack_name}' returned "
                f"{len(X_attacked)} samples instead of "
                f"{n_fraud}."
            )

        if np.sum(y_attacked) != n_fraud:

            raise ValueError(
                f"Attack '{attack_name}' returned "
                f"{np.sum(y_attacked)} fraud labels instead of "
                f"{n_fraud}."
            )

        # ----------------------------------------------------
        # COMBINE BENIGN + ATTACKED FRAUD
        # ----------------------------------------------------

        X_eval = pd.concat(
            [
                X_benign,
                X_attacked
            ],
            ignore_index=True
        )

        y_eval = np.concatenate(
            [
                y_benign,
                y_attacked
            ]
        )

        # ----------------------------------------------------
        # FINAL SCHEMA CHECK
        # ----------------------------------------------------

        if list(X_eval.columns) != LOCKED_FEATURES:

            raise ValueError(
                f"Final evaluation schema mismatch for "
                f"'{attack_name}'."
            )

        # ----------------------------------------------------
        # V6 PREDICTION
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_eval
        )[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        auc = roc_auc_score(
            y_eval,
            probabilities
        )

        recall = recall_score(
            y_eval,
            predictions,
            zero_division=0
        )

        precision = precision_score(
            y_eval,
            predictions,
            zero_division=0
        )

        # ----------------------------------------------------
        # FRAUD-SPECIFIC PROBABILITIES
        # ----------------------------------------------------

        fraud_probabilities = probabilities[
            n_benign:
        ]

        benign_probabilities = probabilities[
            :n_benign
        ]

        fraud_mean = float(
            np.mean(fraud_probabilities)
        )

        fraud_median = float(
            np.median(fraud_probabilities)
        )

        benign_mean = float(
            np.mean(benign_probabilities)
        )

        true_positives = int(
            np.sum(
                predictions[n_benign:]
            )
        )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "attack_family": attack_name,
                "auc": float(auc),
                "recall": float(recall),
                "precision": float(precision),
                "fraud_probability_mean": fraud_mean,
                "fraud_probability_median": fraud_median,
                "benign_probability_mean": benign_mean,
                "true_positives": true_positives,
                "fraud_samples": n_fraud,
                "total_samples": n_total
            }
        )

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            f"    AUC       : {auc:.4f}"
        )

        print(
            f"    Recall    : {recall:.4f}"
        )

        print(
            f"    Precision : {precision:.4f}"
        )

        print(
            f"    Fraud Prob: {fraud_mean:.4f}"
        )

        print(
            f"    TP        : "
            f"{true_positives}/{n_fraud}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(results)

    mean_auc = float(
        results_df["auc"].mean()
    )

    worst_auc = float(
        results_df["auc"].min()
    )

    mean_recall = float(
        results_df["recall"].mean()
    )

    worst_recall = float(
        results_df["recall"].min()
    )

    mean_precision = float(
        results_df["precision"].mean()
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("STAGE 10 v6 ATTACK RESULTS")
    print("=" * 70)

    for _, row in results_df.iterrows():

        print(
            f"{row['attack_family']:<25} "
            f"AUC={row['auc']:.4f} "
            f"Recall={row['recall']:.4f} "
            f"Precision={row['precision']:.4f}"
        )

    print("-" * 70)

    print(
        f"Global Mean AUC     : {mean_auc:.4f}"
    )

    print(
        f"Worst Attack AUC    : {worst_auc:.4f}"
    )

    print(
        f"Global Mean Recall  : {mean_recall:.4f}"
    )

    print(
        f"Worst Attack Recall : {worst_recall:.4f}"
    )

    print(
        f"Global Mean Precision: {mean_precision:.4f}"
    )

    print("-" * 70)

    # ========================================================
    # LOCKED STAGE 9 GATES
    # ========================================================

    auc_gate = mean_auc >= 0.85

    worst_auc_gate = worst_auc >= 0.80

    recall_gate = mean_recall >= 0.55

    print(
        f"Gate [Mean AUC >= 0.85]       : "
        f"{'PASS' if auc_gate else 'FAIL'} "
        f"({mean_auc:.4f})"
    )

    print(
        f"Gate [Worst AUC >= 0.80]      : "
        f"{'PASS' if worst_auc_gate else 'FAIL'} "
        f"({worst_auc:.4f})"
    )

    print(
        f"Gate [Mean Recall >= 0.55]    : "
        f"{'PASS' if recall_gate else 'FAIL'} "
        f"({mean_recall:.4f})"
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    print()
    print("=" * 70)

    if (
        auc_gate
        and worst_auc_gate
        and recall_gate
    ):

        print(
            "VERDICT: v6_ROBUST PASSES STAGE 9 "
            "GENERALIZATION"
        )

    else:

        print(
            "VERDICT: v6_ROBUST FAILS STAGE 9 "
            "GENERALIZATION"
        )

    print("=" * 70)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = os.path.join(
        PROJECT_ROOT,
        "evaluation",
        "stage10_v6_generalization_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"[SUCCESS] Results saved to:\n"
        f"          {output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_stage10_v6_evaluation()