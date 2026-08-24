import json
from pathlib import Path

import joblib
import shap

from generation.fraud_injector import generate_fraud_dataset
from generation.generator import FEATURES


def run_pipeline(n=1000, seed=42):
    df = generate_fraud_dataset(n, fraud_rate=0.05, seed=seed)

    model = joblib.load("models/xgboost_detector.joblib")

    X = df[FEATURES]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    results = []

    for i in range(len(df)):
        if predictions[i] == 1:
            contributions = {
                feature: float(shap_values[i, j])
                for j, feature in enumerate(FEATURES)
            }

            top_features = sorted(
                contributions.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:3]

            results.append({
                "transaction_index": int(i),
                "fraud_probability": float(probabilities[i]),
                "predicted_fraud": True,
                "actual_fraud": int(df.iloc[i]["is_fraud"]),
                "top_risk_features": [
                    {
                        "feature": feature,
                        "contribution": value,
                    }
                    for feature, value in top_features
                ],
            })

    output = {
        "transactions_analyzed": len(df),
        "fraud_predictions": int(predictions.sum()),
        "results_with_explanations": len(results),
        "results": results[:100],
    }

    Path("evaluation").mkdir(exist_ok=True)

    with open(
        "evaluation/pipeline_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output, f, indent=2)

    print("CLOSED-LOOP PIPELINE: OK")
    print("TRANSACTIONS:", len(df))
    print("FRAUD PREDICTIONS:", int(predictions.sum()))
    print("EXPLANATIONS:", len(results))


if __name__ == "__main__":
    run_pipeline()
