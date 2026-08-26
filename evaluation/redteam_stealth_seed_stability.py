import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score

from evaluation.redteam_stealth_curriculum import (
    build_stealth_curriculum,
    train_model,
    make_stealth_dataset,
)

from generation.hard_negatives import generate_legitimate_travelers
from generation.generator import FEATURES


SEEDS = [7, 21, 42, 99, 123]
THRESHOLD = 0.10

rows = []

print("=== STAGE 14A: STEALTH SEED STABILITY ===")

for seed in SEEDS:

    training = pd.concat(
        build_stealth_curriculum(seed=seed),
        ignore_index=True,
    )

    model = train_model(
        training,
        seed=seed,
    )

    fraud = make_stealth_dataset(
        n=12000,
        fraud_rate=0.05,
        seed=seed + 1000,
        level=4,
    )

    benign = generate_legitimate_travelers(
        n=12000,
        seed=seed + 2000,
    )

    fraud_scores = model.predict_proba(
        fraud[FEATURES]
    )[:, 1]

    benign_scores = model.predict_proba(
        benign[FEATURES]
    )[:, 1]

    fraud_predictions = (
        fraud_scores >= THRESHOLD
    ).astype(int)

    auc = roc_auc_score(
        fraud["is_fraud"],
        fraud_scores,
    )

    recall = recall_score(
        fraud["is_fraud"],
        fraud_predictions,
        zero_division=0,
    )

    fpr = float(
        (benign_scores >= THRESHOLD).mean()
    )

    rows.append(
        (seed, auc, recall, fpr)
    )

    print(
        f"Seed {seed:3d} | "
        f"AUC {auc:.4f} | "
        f"Recall {recall:.4f} | "
        f"FPR {fpr:.4f}"
    )


min_auc = min(r[1] for r in rows)
min_recall = min(r[2] for r in rows)
max_fpr = max(r[3] for r in rows)

mean_auc = np.mean([r[1] for r in rows])
mean_recall = np.mean([r[2] for r in rows])
mean_fpr = np.mean([r[3] for r in rows])

print()
print(f"Minimum AUC    : {min_auc:.4f}")
print(f"Minimum recall : {min_recall:.4f}")
print(f"Maximum FPR    : {max_fpr:.4f}")
print(f"Mean AUC       : {mean_auc:.4f}")
print(f"Mean recall    : {mean_recall:.4f}")
print(f"Mean FPR       : {mean_fpr:.4f}")

print()

if (
    min_auc >= 0.90
    and min_recall >= 0.70
    and max_fpr <= 0.05
):
    print("VERDICT: PASS")
else:
    print("VERDICT: FAIL")
