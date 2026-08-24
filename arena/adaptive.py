import json
from pathlib import Path

import joblib
import numpy as np

from arena.engine import FraudArena
from attacks.adversarial import generate_adversarial_cases
from generation.fraud_injector import generate_fraud_dataset
from generation.generator import FEATURES


def run_adaptive_arena(rounds=5, n=2000, seed=42):
    model = joblib.load("models/xgboost_detector.joblib")
    arena = FraudArena()

    df = generate_fraud_dataset(n, fraud_rate=0.05, seed=seed)

    history = []

    for round_number in range(1, rounds + 1):
        attacked = generate_adversarial_cases(
            df,
            seed=seed + round_number,
        )

        probabilities = model.predict_proba(
            attacked[FEATURES]
        )[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        attack_mask = attacked["is_fraud"] == 1

        attack_count = int(attack_mask.sum())
        detected_count = int(predictions[attack_mask].sum())

        result = arena.record_round(
            round_number,
            attack_count,
            detected_count,
        )

        history.append({
            "round": round_number,
            "attacks": attack_count,
            "detected": detected_count,
            "detection_rate": result.detection_rate,
        })

        print(
            f"ROUND {round_number}: "
            f"{detected_count}/{attack_count} "
            f"({result.detection_rate:.4f})"
        )

        # Create the next generation of synthetic cases.
        df = generate_fraud_dataset(
            n,
            fraud_rate=0.05,
            seed=seed + round_number + 100,
        )

    output = {
        "rounds": rounds,
        "transactions_per_round": n,
        "history": history,
        "summary": arena.summary(),
    }

    Path("evaluation").mkdir(exist_ok=True)

    with open(
        "evaluation/adaptive_arena_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output, f, indent=2)

    print("ADAPTIVE ARENA: OK")
    print("ROUNDS:", rounds)
    print(
        "BEST DETECTION RATE:",
        f"{arena.summary()['best_detection_rate']:.4f}",
    )


if __name__ == "__main__":
    run_adaptive_arena()
