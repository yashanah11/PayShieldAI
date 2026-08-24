import json
from pathlib import Path

import joblib
import shap

from generation.fraud_injector import generate_fraud_dataset
from generation.generator import FEATURES


def explain_predictions(n=1000, seed=42):
    df = generate_fraud_dataset(n, fraud_rate=0.05, seed=seed)

    model = joblib.load("models/xgboost_detector.joblib")

    X = df[FEATURES]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = {}

    for i, feature in enumerate(FEATURES):
        importance[feature] = float(
            abs(shap_values[:, i]).mean()
        )

    importance = dict(
        sorted(
            importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    Path("explainability").mkdir(exist_ok=True)

    with open(
        "explainability/feature_importance.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(importance, f, indent=2)

    print("SHAP EXPLAINABILITY: OK")
    print("FEATURE IMPORTANCE:")

    for feature, value in importance.items():
        print(f"  {feature}: {value:.6f}")

    return importance


if __name__ == "__main__":
    explain_predictions()
