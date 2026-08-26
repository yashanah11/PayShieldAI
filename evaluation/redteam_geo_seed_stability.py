import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score

from evaluation.redteam_geo_curriculum import (
    build_curriculum,
    train_model,
    make_geo_dataset,
)

from generation.hard_negatives import generate_legitimate_travelers
from generation.generator import FEATURES


SEEDS = [7, 21, 42, 99, 123]
THRESHOLD = 0.10

print("=== STAGE 12: GEOGRAPHIC SEED STABILITY ===")

rows = []

for seed in SEEDS:

    training = pd.concat(
        build_curriculum(seed=seed),
        ignore_index=True,
    )

    model = train_model(training, seed=seed)

    fraud = make_geo_dataset(
        n=12000,
        fraud_rate=0.05,
        seed=seed + 1000,
        level=5,
    )

    benign = generate_legitimate_travelers(
        n=12000,
        seed=seed + 2000,
    )

    fraud_prob = model.predict_proba(
        fraud[FEATURES]
    )[:, 1]

    benign_prob = model.predict_proba(
        benign[FEATURES]
    )[:, 1]

    fraud_predictions = (
        fraud_prob >= THRESHOLD
    ).astype(int)

    auc = roc_auc_score(
        fraud["is_fraud"],
        fraud_prob,
    )

    recall = recall_score(
        fraud["is_fraud"],
        fraud_predictions,
        zero_division=0,
    )

    fpr = float(
        (benign_prob >= THRESHOLD).mean()
    )

    rows.append((seed, auc, recall, fpr))

    print(
        f"Seed {seed:3d} | "
        f"AUC {auc:.4f} | "
        f"Recall {recall:.4f} | "
        f"FPR {fpr:.4f}"
    )

min_recall = min(row[2] for row in rows)
max_fpr = max(row[3] for row in rows)

mean_recall = np.mean([row[2] for row in rows])
mean_fpr = np.mean([row[3] for row in rows])

print()
print(f"Minimum recall : {min_recall:.4f}")
print(f"Maximum FPR    : {max_fpr:.4f}")
print(f"Mean recall    : {mean_recall:.4f}")
print(f"Mean FPR       : {mean_fpr:.4f}")

print()

if min_recall >= 0.70 and max_fpr <= 0.05:
    print("VERDICT: PASS")
else:
    print("VERDICT: FAIL")
