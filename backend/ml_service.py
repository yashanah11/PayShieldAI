import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is accessible to import evaluation modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.redteam_attack_families import get_redteam_attacks

V6_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_detector_v6_robust.joblib")

LOCKED_FEATURES = [
    "amount", "hour", "velocity_1h", "velocity_24h", 
    "device_age_days", "distance_km", "merchant_risk"
]

class MLService:
    def __init__(self):
        if not os.path.exists(V6_MODEL_PATH):
            raise FileNotFoundError(f"Production model not found at {V6_MODEL_PATH}")
        self.model = joblib.load(V6_MODEL_PATH)
        # Decision threshold used consistently in the project
        self.threshold = 0.30

    def predict(self, transaction_data: dict) -> dict:
        """Runs inference using the v6 robust model."""
        df = pd.DataFrame([transaction_data], columns=LOCKED_FEATURES)
        
        prob = float(self.model.predict_proba(df)[0][1])
        decision = "BLOCK" if prob >= self.threshold else "ALLOW"
        
        if prob >= 0.70:
            classification = "HIGH"
        elif prob >= self.threshold:
            classification = "MEDIUM"
        else:
            classification = "LOW"

        return {
            "fraud_probability": round(prob, 4),
            "risk_classification": classification,
            "decision": decision,
            "input_transaction": transaction_data
        }

    def explain(self, transaction_data: dict) -> dict:
        """Transparent feature-based explanation using model's global feature importance."""
        if hasattr(self.model, "feature_importances_"):
            importances = dict(zip(LOCKED_FEATURES, self.model.feature_importances_))
            sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            return {"global_feature_importance": {k: round(float(v), 4) for k, v in sorted_importances}}
        return {"error": "Feature importance not available for this model type."}

    def simulate_attack(self, transaction_data: dict, attack_family: str) -> dict:
        """Applies a locked red-team attack without altering original transaction."""
        attacks = get_redteam_attacks()
        if attack_family not in attacks:
            raise ValueError(f"Invalid attack. Allowed: {list(attacks.keys())}")

        attack_fn = attacks[attack_family]
        
        X_base = pd.DataFrame([transaction_data], columns=LOCKED_FEATURES)
        y_base = np.zeros(1, dtype=int)
        
        X_adv, _ = attack_fn(X_base.copy(), y_base.copy())
        
        # Enforce canonical schema
        X_adv = X_adv[LOCKED_FEATURES]
        
        attacked_data = X_adv.iloc[0].to_dict()
        
        # Determine changed features
        changed = [f for f in LOCKED_FEATURES if float(transaction_data[f]) != float(attacked_data[f])]
        
        prediction = self.predict(attacked_data)
        
        return {
            "original_features": transaction_data,
            "attacked_features": attacked_data,
            "changed_features": changed,
            "v6_prediction": prediction
        }

ml_service = MLService()