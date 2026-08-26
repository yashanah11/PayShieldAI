"""
STAGE 7: ENSEMBLE ROBUSTNESS / STRESS TEST

Purpose
-------
Validate the exact Stage 6C ensemble configuration under multiple
random seeds, evolved-attack samples, fraud rates, and feature noise.

Stage 6C successful configuration:
    v1 weight     = 0.50
    v3 weight     = 0.50
    threshold     = 0.30

Stage 7 DOES NOT retrain the models.

Gates
-----
Mean Standard AUC       >= 0.85
Worst Standard AUC      >= 0.82
Mean Evolved Recall     >= 0.60
Worst Evolved Recall    >= 0.50
Successful configurations >= 80%

Output
------
models/stage7_robustness_results.csv
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)

warnings.filterwarnings("ignore")

from generation.generator import FEATURES
from evaluation.hard_benchmark_stage2_repaired import (
    generate_stage2_repaired_benchmark,
)
from evaluation.redteam_stage2c_retest import make_unseen_attacks


# ============================================================
# CONFIGURATION
# ============================================================

ENSEMBLE_CONFIG_PATH = "models/ensemble_config.joblib"

V1_PATH = "models/xgboost_detector_retrained.joblib"
V3_PATH = "models/xgboost_detector_retrained_v3.joblib"

RESULT_PATH = "models/stage7_robustness_results.csv"

# Stage 6C gates
AUC_GATE = 0.85
RECALL_GATE = 0.60

# Stage 7 robustness gates
MEAN_AUC_GATE = 0.85
WORST_AUC_GATE = 0.82
MEAN_RECALL_GATE = 0.60
WORST_RECALL_GATE = 0.50
SUCCESS_RATE_GATE = 0.80

# Random seeds
SEEDS = [42, 123, 456, 789, 2026]

# Fraud-rate stress tests
FRAUD_RATES = [0.03, 0.05, 0.10]

# Feature-noise levels
NOISE_LEVELS = [0.0, 0.01, 0.03]


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():
    print("Loading Stage 6C ensemble models...")

    if not os.path.exists(V1_PATH):
        raise FileNotFoundError(
            f"Missing v1 model: {V1_PATH}"
        )

    if not os.path.exists(V3_PATH):
        raise FileNotFoundError(
            f"Missing v3 model: {V3_PATH}"
        )

    if not os.path.exists(ENSEMBLE_CONFIG_PATH):
        raise FileNotFoundError(
            f"Missing ensemble configuration: {ENSEMBLE_CONFIG_PATH}"
        )

    model_v1 = joblib.load(V1_PATH)
    model_v3 = joblib.load(V3_PATH)
    config = joblib.load(ENSEMBLE_CONFIG_PATH)

    print("Successfully loaded v1 and v3.")
    print(f"Ensemble configuration: {config}")

    return model_v1, model_v3, config


# ============================================================
# GET ENSEMBLE PARAMETERS
# ============================================================

def get_ensemble_parameters(config):
    """
    Supports the configuration produced by Stage 6C.
    """

    weight_v1 = None
    threshold = None

    # Handle dictionary-style config
    if isinstance(config, dict):

        possible_weight_keys = [
            "weight_v1",
            "v1_weight",
            "w",
            "optimal_weight",
        ]

        possible_threshold_keys = [
            "threshold",
            "optimal_threshold",
            "decision_threshold",
        ]

        for key in possible_weight_keys:
            if key in config:
                weight_v1 = float(config[key])
                break

        for key in possible_threshold_keys:
            if key in config:
                threshold = float(config[key])
                break

    if weight_v1 is None:
        weight_v1 = 0.50

    if threshold is None:
        threshold = 0.30

    weight_v3 = 1.0 - weight_v1

    return weight_v1, weight_v3, threshold


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(
    model_v1,
    model_v3,
    df,
    weight_v1,
):
    """
    Generate blended probabilities.

    final_probability =
        weight_v1 * v1_probability
        +
        (1 - weight_v1) * v3_probability
    """

    X = df[FEATURES]

    prob_v1 = model_v1.predict_proba(X)[:, 1]
    prob_v3 = model_v3.predict_proba(X)[:, 1]

    weight_v3 = 1.0 - weight_v1

    blended_probability = (
        weight_v1 * prob_v1
        + weight_v3 * prob_v3
    )

    return blended_probability


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model_v1,
    model_v3,
    df,
    weight_v1,
    threshold,
):
    """
    Evaluate ensemble on a dataset.
    """

    y = df["is_fraud"].astype(int)

    probabilities = ensemble_predict(
        model_v1,
        model_v3,
        df,
        weight_v1,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    if len(np.unique(y)) > 1:
        auc = roc_auc_score(
            y,
            probabilities,
        )
    else:
        auc = np.nan

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    return {
        "auc": auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


# ============================================================
# FEATURE NOISE
# ============================================================

def add_feature_noise(df, noise_level, seed):
    """
    Apply small Gaussian perturbations to numeric features.

    Noise is applied only to FEATURES.
    is_fraud remains untouched.
    """

    if noise_level == 0:
        return df.copy()

    rng = np.random.default_rng(seed)

    noisy_df = df.copy()

    for feature in FEATURES:

        if not pd.api.types.is_numeric_dtype(
            noisy_df[feature]
        ):
            continue

        values = noisy_df[feature].astype(float).values

        scale = np.std(values)

        if scale == 0 or np.isnan(scale):
            scale = 1.0

        noise = rng.normal(
            loc=0.0,
            scale=noise_level * scale,
            size=len(values),
        )

        noisy_df[feature] = values + noise

    return noisy_df


# ============================================================
# GENERATE STANDARD TEST
# ============================================================

def generate_standard_test(
    fraud_rate,
    seed,
):
    """
    Generate a fresh standard benchmark.
    """

    test_base = generate_stage2_repaired_benchmark(
        n=5000,
        fraud_rate=fraud_rate,
        seed=seed,
    )

    return test_base


# ============================================================
# GENERATE EVOLVED ATTACK TEST
# ============================================================

def generate_evolved_test(
    seed,
):
    """
    Generate unseen evolved attacks from a fresh standard base.
    """

    base = generate_stage2_repaired_benchmark(
        n=5000,
        fraud_rate=0.05,
        seed=seed + 10000,
    )

    evolved_test = make_unseen_attacks(
        base,
        seed=seed + 20000,
    )

    return evolved_test


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STAGE 7: ENSEMBLE ROBUSTNESS / STRESS TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load exact Stage 6C configuration
    # --------------------------------------------------------

    model_v1, model_v3, config = load_models()

    weight_v1, weight_v3, threshold = (
        get_ensemble_parameters(config)
    )

    print()
    print("Stage 6C configuration being tested:")
    print(f"v1 weight       : {weight_v1:.4f}")
    print(f"v3 weight       : {weight_v3:.4f}")
    print(f"threshold       : {threshold:.4f}")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    experiment_id = 0

    # ========================================================
    # EXPERIMENT MATRIX
    # ========================================================

    for seed in SEEDS:

        for fraud_rate in FRAUD_RATES:

            for noise_level in NOISE_LEVELS:

                experiment_id += 1

                print()
                print("-" * 70)
                print(
                    f"Experiment {experiment_id}"
                )
                print(
                    f"Seed={seed} | "
                    f"FraudRate={fraud_rate} | "
                    f"Noise={noise_level}"
                )
                print("-" * 70)

                # ------------------------------------------------
                # Standard test
                # ------------------------------------------------

                standard_test = generate_standard_test(
                    fraud_rate=fraud_rate,
                    seed=seed,
                )

                standard_test = add_feature_noise(
                    standard_test,
                    noise_level,
                    seed,
                )

                standard_metrics = evaluate(
                    model_v1,
                    model_v3,
                    standard_test,
                    weight_v1,
                    threshold,
                )

                # ------------------------------------------------
                # Evolved attacks
                # ------------------------------------------------

                evolved_test = generate_evolved_test(
                    seed=seed,
                )

                evolved_test = add_feature_noise(
                    evolved_test,
                    noise_level,
                    seed + 50000,
                )

                evolved_metrics = evaluate(
                    model_v1,
                    model_v3,
                    evolved_test,
                    weight_v1,
                    threshold,
                )

                # ------------------------------------------------
                # Gate
                # ------------------------------------------------

                auc_pass = (
                    standard_metrics["auc"]
                    >= AUC_GATE
                )

                recall_pass = (
                    evolved_metrics["recall"]
                    >= RECALL_GATE
                )

                passed = (
                    auc_pass
                    and recall_pass
                )

                print()
                print(
                    f"Standard AUC      : "
                    f"{standard_metrics['auc']:.4f}"
                )

                print(
                    f"Evolved Recall    : "
                    f"{evolved_metrics['recall']:.4f}"
                )

                print(
                    f"Standard Precision: "
                    f"{standard_metrics['precision']:.4f}"
                )

                print(
                    f"Evolved Precision : "
                    f"{evolved_metrics['precision']:.4f}"
                )

                print(
                    f"Result            : "
                    f"{'PASS' if passed else 'FAIL'}"
                )

                results.append(
                    {
                        "experiment": experiment_id,
                        "seed": seed,
                        "fraud_rate": fraud_rate,
                        "noise_level": noise_level,
                        "auc_standard": standard_metrics["auc"],
                        "recall_standard": standard_metrics["recall"],
                        "precision_standard": standard_metrics["precision"],
                        "f1_standard": standard_metrics["f1"],
                        "recall_evolved": evolved_metrics["recall"],
                        "precision_evolved": evolved_metrics["precision"],
                        "f1_evolved": evolved_metrics["f1"],
                        "auc_pass": auc_pass,
                        "recall_pass": recall_pass,
                        "passed": passed,
                    }
                )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(
        os.path.dirname(RESULT_PATH),
        exist_ok=True,
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False,
    )

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    mean_auc = results_df[
        "auc_standard"
    ].mean()

    worst_auc = results_df[
        "auc_standard"
    ].min()

    mean_recall = results_df[
        "recall_evolved"
    ].mean()

    worst_recall = results_df[
        "recall_evolved"
    ].min()

    successful_experiments = results_df[
        "passed"
    ].sum()

    total_experiments = len(
        results_df
    )

    success_rate = (
        successful_experiments
        / total_experiments
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("STAGE 7 ROBUSTNESS VERDICT")
    print("=" * 70)

    print()
    print(
        f"Experiments tested      : "
        f"{total_experiments}"
    )

    print(
        f"Successful experiments  : "
        f"{successful_experiments}"
    )

    print(
        f"Success rate            : "
        f"{success_rate:.2%}"
    )

    print()
    print(
        f"Mean Standard AUC       : "
        f"{mean_auc:.4f}"
    )

    print(
        f"Worst Standard AUC      : "
        f"{worst_auc:.4f}"
    )

    print()
    print(
        f"Mean Evolved Recall     : "
        f"{mean_recall:.4f}"
    )

    print(
        f"Worst Evolved Recall    : "
        f"{worst_recall:.4f}"
    )

    print()
    print("-" * 70)
    print("STAGE 7 GATES")
    print("-" * 70)

    mean_auc_pass = (
        mean_auc >= MEAN_AUC_GATE
    )

    worst_auc_pass = (
        worst_auc >= WORST_AUC_GATE
    )

    mean_recall_pass = (
        mean_recall >= MEAN_RECALL_GATE
    )

    worst_recall_pass = (
        worst_recall >= WORST_RECALL_GATE
    )

    success_rate_pass = (
        success_rate >= SUCCESS_RATE_GATE
    )

    print(
        f"Mean AUC >= 0.85       : "
        f"{'PASS' if mean_auc_pass else 'FAIL'}"
    )

    print(
        f"Worst AUC >= 0.82      : "
        f"{'PASS' if worst_auc_pass else 'FAIL'}"
    )

    print(
        f"Mean Recall >= 0.60    : "
        f"{'PASS' if mean_recall_pass else 'FAIL'}"
    )

    print(
        f"Worst Recall >= 0.50   : "
        f"{'PASS' if worst_recall_pass else 'FAIL'}"
    )

    print(
        f"Success Rate >= 80%    : "
        f"{'PASS' if success_rate_pass else 'FAIL'}"
    )

    stage7_pass = (
        mean_auc_pass
        and worst_auc_pass
        and mean_recall_pass
        and worst_recall_pass
        and success_rate_pass
    )

    print()
    print("=" * 70)

    if stage7_pass:

        print("VERDICT: STAGE 7 PASS")
        print()
        print(
            "The Stage 6C ensemble remained robust "
            "across the stress-test matrix."
        )

    else:

        print("VERDICT: STAGE 7 FAIL")

        print()

        if not mean_auc_pass:
            print(
                f"Reason: Mean AUC "
                f"{mean_auc:.4f} < 0.85"
            )

        if not worst_auc_pass:
            print(
                f"Reason: Worst AUC "
                f"{worst_auc:.4f} < 0.82"
            )

        if not mean_recall_pass:
            print(
                f"Reason: Mean evolved recall "
                f"{mean_recall:.4f} < 0.60"
            )

        if not worst_recall_pass:
            print(
                f"Reason: Worst evolved recall "
                f"{worst_recall:.4f} < 0.50"
            )

        if not success_rate_pass:
            print(
                f"Reason: Success rate "
                f"{success_rate:.2%} < 80%"
            )

    print("=" * 70)

    print()
    print(
        f"Detailed results saved to:"
    )
    print(
        f"{RESULT_PATH}"
    )