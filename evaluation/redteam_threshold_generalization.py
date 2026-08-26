import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from evaluation.redteam_geo_curriculum import (
    make_geo_dataset,
    build_curriculum,
)
from generation.generator import FEATURES


def train_model(df, seed=42):
    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(
        df[FEATURES],
        df["is_fraud"],
    )

    return model


print("=== STAGE 6: THRESHOLD GENERALIZATION ===")

# -------------------------
# TRAINING
# -------------------------

curriculum = build_curriculum(seed=42)

training = pd.concat(
    curriculum,
    ignore_index=True,
)

model = train_model(training)

# -------------------------
# VALIDATION
# -------------------------

validation = make_geo_dataset(
    n=10000,
    fraud_rate=0.05,
    seed=5555,
    level=5,
)

validation_prob = model.predict_proba(
    validation[FEATURES]
)[:, 1]

validation_y = validation["is_fraud"]

print()
print("VALIDATION")

best_threshold = None
best_f1 = -1

for threshold in np.arange(
    0.05,
    0.51,
    0.01,
):
    pred = (
        validation_prob >= threshold
    ).astype(int)

    precision = precision_score(
        validation_y,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        validation_y,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        validation_y,
        pred,
        zero_division=0,
    )

    # Require at least 60% recall.
    if recall >= 0.60 and f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(
    f"Selected threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1      : "
    f"{best_f1:.4f}"
)

# -------------------------
# UNTOUCHED TEST
# -------------------------

test = make_geo_dataset(
    n=10000,
    fraud_rate=0.05,
    seed=7777,
    level=5,
)

test_prob = model.predict_proba(
    test[FEATURES]
)[:, 1]

test_y = test["is_fraud"]

test_pred = (
    test_prob >= best_threshold
).astype(int)

auc = roc_auc_score(
    test_y,
    test_prob,
)

precision = precision_score(
    test_y,
    test_pred,
    zero_division=0,
)

recall = recall_score(
    test_y,
    test_pred,
    zero_division=0,
)

f1 = f1_score(
    test_y,
    test_pred,
    zero_division=0,
)

print()
print("UNTOUCHED TEST")

print(f"ROC-AUC   : {auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")

print()

if recall >= 0.60 and precision >= 0.50:
    print("VERDICT: THRESHOLD GENERALIZES")
else:
    print("VERDICT: THRESHOLD DOES NOT GENERALIZE")
