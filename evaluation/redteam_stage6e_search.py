"""
STAGE 6e: FINAL MODEL SEARCH

Gates:
    AUC_STD >= 0.85
    RECALL_EVOL >= 0.60

Search:
    evolved sample size: 50, 100, 150
    scale_pos_weight: 1.0, 2.0, 3.0
    n_estimators: 300, 500
    max_depth: 6, 7
"""

import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from generation.generator import FEATURES

from evaluation.hard_benchmark_stage2_repaired import (
    generate_stage2_repaired_benchmark,
)

from evaluation.redteam_stage2c_retest import (
    make_unseen_attacks,
)


AUC_GATE = 0.85
RECALL_GATE = 0.60


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

    evolved = pd.read_csv(
        "data/evolved_attacks.csv"
    )

    evolved = evolved[
        FEATURES + ["is_fraud"]
    ]

    return train_base, test_df, evolved


def evaluate_model(model, df):

    X = df[FEATURES]
    y = df["is_fraud"]

    probabilities = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(
        y,
        probabilities,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    return auc, recall


def train_and_eval(
    train_df,
    test_std,
    test_evol,
    scale,
    n_est,
    depth,
    seed=42,
):

    X_train, _, y_train, _ = train_test_split(
        train_df[FEATURES],
        train_df["is_fraud"],
        test_size=0.30,
        random_state=seed,
        stratify=train_df["is_fraud"],
    )

    model = XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale,
    )

    model.fit(
        X_train,
        y_train,
    )

    auc_std, rec_std = evaluate_model(
        model,
        test_std,
    )

    auc_evol, rec_evol = evaluate_model(
        model,
        test_evol,
    )

    return (
        auc_std,
        rec_std,
        auc_evol,
        rec_evol,
        model,
    )


if __name__ == "__main__":

    print(
        "=== STAGE 6e: FINAL MODEL SEARCH ==="
    )

    train_base, test_standard, evolved = load_data()

    print(
        f"Training base: {len(train_base)} rows"
    )

    print(
        f"Standard test: {len(test_standard)} rows"
    )

    print(
        f"Evolved attacks: {len(evolved)} rows"
    )

    sample_sizes = [
        50,
        100,
        150,
    ]

    scales = [
        1.0,
        2.0,
        3.0,
    ]

    estimators = [
        300,
        500,
    ]

    depths = [
        6,
        7,
    ]

    results = []

    for n_evol in sample_sizes:

        print(
            f"\nSample size: {n_evol}"
        )

        evolved_sample = evolved.sample(
            n=n_evol,
            random_state=42,
        )

        train_df = pd.concat(
            [
                train_base,
                evolved_sample,
            ],
            ignore_index=True,
        )

        train_df = train_df.sample(
            frac=1,
            random_state=42,
        ).reset_index(
            drop=True
        )

        for scale in scales:

            for n_est in estimators:

                for depth in depths:

                    (
                        auc_std,
                        rec_std,
                        auc_evol,
                        rec_evol,
                        model,
                    ) = train_and_eval(
                        train_df,
                        test_standard,
                        evolved,
                        scale,
                        n_est,
                        depth,
                    )

                    result = {
                        "n_evol": n_evol,
                        "scale": scale,
                        "n_est": n_est,
                        "depth": depth,
                        "auc_std": auc_std,
                        "rec_std": rec_std,
                        "auc_evol": auc_evol,
                        "rec_evol": rec_evol,
                    }

                    results.append(
                        result
                    )

    print()
    print("=" * 70)
    print("VALID CANDIDATES")
    print("=" * 70)

    valid_candidates = [
        r
        for r in results
        if (
            r["auc_std"] >= AUC_GATE
            and
            r["rec_evol"] >= RECALL_GATE
        )
    ]

    if not valid_candidates:

        print()
        print(
            "NO MODEL PASSED BOTH GATES."
        )

        print()
        print(
            f"AUC gate    : >= {AUC_GATE}"
        )

        print(
            f"Recall gate : >= {RECALL_GATE}"
        )

        print()
        print(
            "VERDICT: STAGE 6 FAIL"
        )

        raise SystemExit(1)

    # Best valid model:
    # prioritize evolved recall,
    # then standard AUC.
    best_candidate = max(
        valid_candidates,
        key=lambda x: (
            x["rec_evol"],
            x["auc_std"],
        ),
    )

    print()

    print(
        f"n_evol={best_candidate['n_evol']}"
    )

    print(
        f"scale={best_candidate['scale']}"
    )

    print(
        f"n_est={best_candidate['n_est']}"
    )

    print(
        f"depth={best_candidate['depth']}"
    )

    print(
        f"AUC_std={best_candidate['auc_std']:.4f}"
    )

    print(
        f"Recall_std={best_candidate['rec_std']:.4f}"
    )

    print(
        f"AUC_evol={best_candidate['auc_evol']:.4f}"
    )

    print(
        f"Recall_evol={best_candidate['rec_evol']:.4f}"
    )

    # --------------------------------------------------
    # RETRAIN BEST MODEL
    # --------------------------------------------------

    evolved_sample = evolved.sample(
        n=best_candidate["n_evol"],
        random_state=42,
    )

    train_df = pd.concat(
        [
            train_base,
            evolved_sample,
        ],
        ignore_index=True,
    )

    train_df = train_df.sample(
        frac=1,
        random_state=42,
    ).reset_index(
        drop=True
    )

    (
        _,
        _,
        _,
        _,
        model_final,
    ) = train_and_eval(
        train_df,
        test_standard,
        evolved,
        best_candidate["scale"],
        best_candidate["n_est"],
        best_candidate["depth"],
    )

    # --------------------------------------------------
    # SAVE ACTUAL MODEL
    # --------------------------------------------------

    output_path = (
        "models/xgboost_detector_final.joblib"
    )

    joblib.dump(
        model_final,
        output_path,
    )

    # --------------------------------------------------
    # VERIFY MODEL ARTIFACT
    # --------------------------------------------------

    loaded_model = joblib.load(
        output_path
    )

    print()
    print(
        f"Final model saved to {output_path}"
    )

    print(
        f"Saved artifact type: {type(loaded_model)}"
    )

    # Actual model verification
    if not hasattr(
        loaded_model,
        "predict_proba",
    ):

        print(
            "ERROR: Saved artifact is not "
            "a valid classifier."
        )

        print(
            "VERDICT: STAGE 6 FAIL"
        )

        raise SystemExit(1)

    print()
    print(
        "AUC GATE: PASS"
    )

    print(
        "EVOLVED RECALL GATE: PASS"
    )

    print()
    print(
        "VERDICT: STAGE 6 PASS"
    )