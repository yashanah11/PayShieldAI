import os
import joblib
import pandas as pd
import numpy as np

try:
    from evaluation.redteam_attack_families import get_redteam_attacks
except ImportError:
    from backend.evaluation.redteam_attack_families import get_redteam_attacks


# ============================================================
# Backend root & Model path
# ============================================================
BACKEND_ROOT = os.path.abspath(os.path.dirname(__file__))

V6_MODEL_PATH = os.path.join(
    BACKEND_ROOT,
    "models",
    "xgboost_detector_v6_robust.joblib"
)

if not os.path.exists(V6_MODEL_PATH):
    # Try root models directory if running from repo root
    _root_model = os.path.abspath(os.path.join(BACKEND_ROOT, "..", "models", "xgboost_detector_v6_robust.joblib"))
    if os.path.exists(_root_model):
        V6_MODEL_PATH = _root_model


# ============================================================
# Authoritative Feature Schema
# ============================================================
LOCKED_FEATURES = [
    "amount",
    "hour",
    "velocity_1h",
    "velocity_24h",
    "device_age_days",
    "distance_km",
    "merchant_risk"
]


# ============================================================
# ML Service
# ============================================================
class MLService:

    def __init__(self):
        """
        Load the production v6 robust fraud detection model.
        """

        if not os.path.exists(V6_MODEL_PATH):
            raise FileNotFoundError(
                f"Production model not found at: {V6_MODEL_PATH}"
            )

        self.model = joblib.load(V6_MODEL_PATH)

        # Decision threshold used consistently across the project
        self.threshold = 0.30

    # ========================================================
    # Prediction
    # ========================================================
    def predict(self, transaction_data: dict) -> dict:
        """
        Run fraud prediction using the v6 robust model.
        """

        # Enforce canonical feature ordering
        df = pd.DataFrame(
            [transaction_data],
            columns=LOCKED_FEATURES
        )

        probability = float(
            self.model.predict_proba(df)[0][1]
        )

        # Decision
        decision = (
            "BLOCK"
            if probability >= self.threshold
            else "ALLOW"
        )

        # Risk classification
        if probability >= 0.70:
            classification = "HIGH"
        elif probability >= self.threshold:
            classification = "MEDIUM"
        else:
            classification = "LOW"

        return {
            "fraud_probability": round(probability, 4),
            "risk_classification": classification,
            "decision": decision,
            "input_transaction": transaction_data
        }

    # ========================================================
    # Explainability
    # ========================================================
    def explain(self, transaction_data: dict) -> dict:
        """
        Return global feature importance from the trained model.
        """

        if hasattr(self.model, "feature_importances_"):

            importances = dict(
                zip(
                    LOCKED_FEATURES,
                    self.model.feature_importances_
                )
            )

            sorted_importances = sorted(
                importances.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return {
                "global_feature_importance": {
                    feature: round(float(importance), 4)
                    for feature, importance in sorted_importances
                }
            }

        return {
            "error": "Feature importance not available for this model type."
        }

    # ========================================================
    # Red-Team Attack Simulation
    # ========================================================
    def simulate_attack(
        self,
        transaction_data: dict,
        attack_family: str
    ) -> dict:
        """
        Apply a locked red-team attack to a transaction
        without modifying the original transaction.
        """

        attacks = get_redteam_attacks()

        # Validate attack family
        if attack_family not in attacks:
            raise ValueError(
                f"Invalid attack. Allowed: {list(attacks.keys())}"
            )

        attack_fn = attacks[attack_family]

        # Original transaction
        X_base = pd.DataFrame(
            [transaction_data],
            columns=LOCKED_FEATURES
        )

        y_base = np.zeros(
            1,
            dtype=int
        )

        # Generate adversarial transaction
        X_adv, _ = attack_fn(
            X_base.copy(),
            y_base.copy()
        )

        # Enforce canonical schema
        X_adv = X_adv[LOCKED_FEATURES]

        attacked_data = X_adv.iloc[0].to_dict()

        # Detect which features changed
        changed_features = [
            feature
            for feature in LOCKED_FEATURES
            if float(transaction_data[feature])
            != float(attacked_data[feature])
        ]

        # Run the production model against attacked transaction
        prediction = self.predict(attacked_data)

        return {
            "original_features": transaction_data,
            "attacked_features": attacked_data,
            "changed_features": changed_features,
            "v6_prediction": prediction
        }


# ============================================================
# Global Production ML Service
# ============================================================
ml_service = MLService()