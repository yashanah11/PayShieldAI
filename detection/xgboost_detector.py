import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from generation.fraud_injector import generate_fraud_dataset
from generation.generator import FEATURES


def train_detector(n=10000, seed=42):
    df = generate_fraud_dataset(n, fraud_rate=0.05, seed=seed)

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probabilities)

    print("XGBOOST DETECTOR: OK")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, predictions, digits=4))

    return model


if __name__ == "__main__":
    model = train_detector()
    joblib.dump(model, "models/xgboost_detector.joblib")
    print("MODEL SAVED: models/xgboost_detector.joblib")
