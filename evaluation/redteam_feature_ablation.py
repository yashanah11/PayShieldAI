import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

from evaluation.redteam_false_positive import (
    make_benign_stress_set,
    make_true_fraud_set,
)

from evaluation.redteam_geo_curriculum import (
    build_curriculum,
)

from generation.generator import FEATURES


print("=== STAGE 9: FEATURE ABLATION ===")

ABLATION_FEATURES = [
    f for f in FEATURES
    if f != "distance_km"
]

print()
print("Removed feature: distance_km")
print("Remaining features:")
print(ABLATION_FEATURES)

# -------------------------
# TRAIN
# -------------------------

training = pd.concat(
    build_curriculum(seed=42),
    ignore_index=True,
)

model = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.06,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

model.fit(
    training[ABLATION_FEATURES],
    training["is_fraud"],
)

# -------------------------
# BENIGN STRESS
# -------------------------

benign = make_benign_stress_set(
    n=10000,
    seed=4242,
)

benign_prob = model.predict_proba(
    benign[ABLATION_FEATURES]
)[:, 1]

threshold = 0.28

benign_pred = (
    benign_prob >= threshold
).astype(int)

fpr = benign_pred.mean()

print()
print("BENIGN STRESS")
print(
    f"False positives    : "
    f"{int(benign_pred.sum())}"
)

print(
    f"False-positive rate : "
    f"{fpr:.4f}"
)

# -------------------------
# FRAUD
# -------------------------

fraud = make_true_fraud_set(
    n=10000,
    seed=5151,
)

fraud_prob = model.predict_proba(
    fraud[ABLATION_FEATURES]
)[:, 1]

fraud_pred = (
    fraud_prob >= threshold
).astype(int)

fraud_y = fraud["is_fraud"]

fraud_recall = recall_score(
    fraud_y,
    fraud_pred,
    zero_division=0,
)

print()
print("FRAUD STRESS")
print(
    f"Recall : {fraud_recall:.4f}"
)

# -------------------------
# MIXED
# -------------------------

mixed = pd.concat(
    [benign, fraud],
    ignore_index=True,
)

probabilities = model.predict_proba(
    mixed[ABLATION_FEATURES]
)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

y = mixed["is_fraud"]

auc = roc_auc_score(
    y,
    probabilities,
)

precision = precision_score(
    y,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0,
)

print()
print("MIXED DATASET")
print(f"ROC-AUC   : {auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")

print()
print("DIAGNOSTIC ONLY")
print("No model has been promoted.")
