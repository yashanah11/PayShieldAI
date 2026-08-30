from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import (
    TransactionInput, PredictionResponse, BatchPredictionRequest, 
    AttackSimulationRequest, AttackSimulationResponse
)
from backend.ml_service import ml_service, LOCKED_FEATURES
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.redteam_attack_families import get_redteam_attacks

app = FastAPI(
    title="PayShieldAI Backend API",
    description="Production API for the robust v6 fraud detection system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": ml_service.model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(transaction: TransactionInput):
    try:
        return ml_service.predict(transaction.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
def batch_predict(request: BatchPredictionRequest):
    results = []
    for txn in request.transactions:
        results.append(ml_service.predict(txn.model_dump()))
    return {"results": results}

@app.get("/attacks")
def get_attack_families():
    attacks = get_redteam_attacks()
    return {"attack_families": list(attacks.keys())}

@app.post("/simulate-attack", response_model=AttackSimulationResponse)
def simulate_attack(request: AttackSimulationRequest):
    try:
        return ml_service.simulate_attack(request.transaction.model_dump(), request.attack_family)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_version": "v6_robust",
        "authoritative_schema": LOCKED_FEATURES,
        "decision_threshold": ml_service.threshold
    }

@app.post("/explain")
def explain_decision(transaction: TransactionInput):
    try:
        return ml_service.explain(transaction.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/demo")
def run_demo():
    """Provides a convenient end-to-end demonstration."""
    demo_txn = {
        "amount": 120.50,
        "hour": 14.0,
        "velocity_1h": 1.0,
        "velocity_24h": 4.0,
        "device_age_days": 350.0,
        "distance_km": 15.0,
        "merchant_risk": 0.1
    }
    benign_pred = ml_service.predict(demo_txn)
    attacked_result = ml_service.simulate_attack(demo_txn, "Merchant Compromise")
    
    return {
        "scenario": "End-to-End Demo",
        "baseline_transaction": benign_pred,
        "attack_simulation": attacked_result
    }