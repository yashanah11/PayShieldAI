from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TransactionInput(BaseModel):
    amount: float = Field(..., description="Transaction amount")
    hour: float = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    velocity_1h: float = Field(..., ge=0, description="Transactions in the last 1 hour")
    velocity_24h: float = Field(..., ge=0, description="Transactions in the last 24 hours")
    device_age_days: float = Field(..., ge=0, description="Age of device in days")
    distance_km: float = Field(..., ge=0, description="Distance from typical location in km")
    merchant_risk: float = Field(..., ge=0.0, le=1.0, description="Risk score of the merchant (0.0-1.0)")

class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_classification: str
    decision: str
    input_transaction: dict

class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionInput]

class AttackSimulationRequest(BaseModel):
    transaction: TransactionInput
    attack_family: str

class AttackSimulationResponse(BaseModel):
    original_features: dict
    attacked_features: dict
    changed_features: List[str]
    v6_prediction: PredictionResponse