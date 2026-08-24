import joblib

from arena.engine import FraudArena
from generation.fraud_injector import generate_fraud_dataset
from generation.generator import FEATURES


def run_arena_round(round_number=1, n=10000, seed=42):
    df = generate_fraud_dataset(n, fraud_rate=0.05, seed=seed)

    model = joblib.load("models/xgboost_detector.joblib")

    probabilities = model.predict_proba(df[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    attack_mask = df["is_fraud"] == 1

    attack_count = int(attack_mask.sum())
    detected_count = int(predictions[attack_mask].sum())

    arena = FraudArena()

    result = arena.record_round(
        round_number,
        attack_count,
        detected_count,
    )

    return result


if __name__ == "__main__":
    result = run_arena_round()

    print("ARENA + DETECTOR: OK")
    print("ATTACKS:", result.attack_count)
    print("DETECTED:", result.detected_count)
    print(f"DETECTION RATE: {result.detection_rate:.4f}")
