import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from generation.generator import FEATURES, generate_transactions
from evaluation.redteam_context_hard_negative_repair import (
    make_stealth_fraud,
    make_hard_benign,
)


SEED = 42


def summarize(name, df):
    print()
    print(f"=== {name} ===")
    print(
        df[FEATURES]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )


print("=== STAGE 15A: DISTRIBUTION COMPATIBILITY DIAGNOSIS ===")

fraud = make_stealth_fraud(
    n=12000,
    fraud_rate=0.05,
    seed=9001,
)

benign = make_hard_benign(
    n=12000,
    seed=9002,
)

summarize("STEALTH FRAUD", fraud)
summarize("HARD BENIGN", benign)

print()
print("=== FEATURE MEAN DIFFERENCE ===")

difference = (
    fraud[FEATURES].mean()
    - benign[FEATURES].mean()
)

print(
    difference
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False,
    )
    .to_string()
)

print()
print("=== SINGLE-FEATURE AUC ===")

y = np.concatenate([
    np.ones(len(fraud)),
    np.zeros(len(benign)),
])

for feature in FEATURES:
    values = np.concatenate([
        fraud[feature].values,
        benign[feature].values,
    ])

    auc = roc_auc_score(y, values)

    # AUC below 0.5 simply means the direction is reversed.
    useful_auc = max(auc, 1.0 - auc)

    print(
        f"{feature:20s} "
        f"raw={auc:.4f} "
        f"useful={useful_auc:.4f}"
    )

print()
print("=== CONTROLLED MODEL ===")

combined = pd.concat(
    [fraud, benign],
    ignore_index=True,
)

X = combined[FEATURES]
y_model = combined["is_fraud"]

model = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_lambda=2.0,
    eval_metric="logloss",
    random_state=SEED,
)

model.fit(X, y_model)

scores = model.predict_proba(X)[:, 1]

auc = roc_auc_score(
    y_model,
    scores,
)

print(
    f"Training-distribution AUC: {auc:.4f}"
)

print()
print("=== SCORE SEPARATION ===")

fraud_scores = scores[:len(fraud)]
benign_scores = scores[len(fraud):]

print(
    f"Fraud mean score  : "
    f"{fraud_scores.mean():.4f}"
)

print(
    f"Benign mean score : "
    f"{benign_scores.mean():.4f}"
)

print()
print("=== DIAGNOSIS ===")

if auc >= 0.90:
    print(
        "Fraud and hard-benign distributions are "
        "separable with current features."
    )
    print(
        "The problem is likely generalization/training design."
    )
else:
    print(
        "CURRENT FEATURES CANNOT CLEANLY SEPARATE "
        "THE TWO DISTRIBUTIONS."
    )
    print(
        "FEATURE ENGINEERING IS REQUIRED BEFORE "
        "FURTHER MODEL DEVELOPMENT."
    )
