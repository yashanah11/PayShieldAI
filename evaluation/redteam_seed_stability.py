import joblib
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from evaluation.hard_benchmark import generate_hard_benchmark
from generation.generator import FEATURES


def train_candidate(X, y, seed):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X, y)
    return model


def run_seed(seed):
    df = generate_hard_benchmark(
        n=10000,
        fraud_rate=0.05,
        seed=seed,
    )

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    baseline = joblib.load(
        "models/xgboost_detector.joblib"
    )

    baseline_auc = roc_auc_score(
        y_holdout,
        baseline.predict_proba(X_holdout)[:, 1],
    )

    train_prob = baseline.predict_proba(X_train)[:, 1]

    hard_mask = (
        ((y_train.to_numpy() == 1) & (train_prob < 0.5))
        |
        ((y_train.to_numpy() == 0) & (train_prob >= 0.5))
    )

    hard_X = X_train.loc[hard_mask]
    hard_y = y_train.loc[hard_mask]

    X_candidate = np.concatenate(
        [X_train.to_numpy(), hard_X.to_numpy()]
    )

    y_candidate = np.concatenate(
        [y_train.to_numpy(), hard_y.to_numpy()]
    )

    candidate = train_candidate(
        X_candidate,
        y_candidate,
        seed,
    )

    candidate_auc = roc_auc_score(
        y_holdout,
        candidate.predict_proba(X_holdout)[:, 1],
    )

    return {
        "seed": seed,
        "baseline_auc": float(baseline_auc),
        "candidate_auc": float(candidate_auc),
        "delta": float(candidate_auc - baseline_auc),
        "hard_examples": int(len(hard_X)),
    }


if __name__ == "__main__":
    seeds = [7, 21, 42, 99, 123]

    print("=== RED-TEAM STAGE 1: SEED STABILITY ===")

    results = []

    for seed in seeds:
        result = run_seed(seed)
        results.append(result)

        print(
            f"Seed {seed:3d} | "
            f"Baseline {result['baseline_auc']:.4f} | "
            f"Candidate {result['candidate_auc']:.4f} | "
            f"Delta {result['delta']:+.4f} | "
            f"Hard {result['hard_examples']}"
        )

    deltas = [r["delta"] for r in results]

    print()
    print(f"Minimum delta : {min(deltas):+.4f}")
    print(f"Maximum delta : {max(deltas):+.4f}")
    print(f"Mean delta    : {np.mean(deltas):+.4f}")
    print(f"Std delta     : {np.std(deltas):.4f}")

    print()

    if min(deltas) > 0:
        print("VERDICT: CANDIDATE IMPROVES ON ALL SEEDS")
    else:
        print("VERDICT: CANDIDATE IS NOT STABLE")
