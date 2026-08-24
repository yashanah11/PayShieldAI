import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

MODEL_PATH = Path("models/xgboost_detector.joblib")
BACKUP_PATH = Path("models/xgboost_detector.backup.joblib")
AUDIT_PATH = Path("evaluation/model_audit.json")


def _load_audit():
    if AUDIT_PATH.exists():
        with open(AUDIT_PATH, encoding="utf-8") as f:
            return json.load(f)

    return []


def _save_audit(records):
    AUDIT_PATH.parent.mkdir(exist_ok=True)

    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def record_decision(
    decision,
    old_auc,
    candidate_auc,
    reason,
):
    records = _load_audit()

    records.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "old_roc_auc": float(old_auc),
        "candidate_roc_auc": float(candidate_auc),
        "reason": reason,
    })

    _save_audit(records)


def promote_candidate(candidate, old_auc, candidate_auc):
    if not hasattr(candidate, "predict_proba"):
        record_decision(
            "REJECT",
            old_auc,
            candidate_auc,
            "Candidate does not provide predict_proba",
        )
        return False

    if candidate_auc < old_auc:
        record_decision(
            "REJECT",
            old_auc,
            candidate_auc,
            "Candidate performance is worse",
        )
        return False

    if not MODEL_PATH.exists():
        record_decision(
            "REJECT",
            old_auc,
            candidate_auc,
            "Current production model does not exist",
        )
        return False

    shutil.copy2(
        MODEL_PATH,
        BACKUP_PATH,
    )

    joblib.dump(
        candidate,
        MODEL_PATH,
    )

    record_decision(
        "PROMOTE",
        old_auc,
        candidate_auc,
        "Candidate performance is equal or better",
    )

    return True


def rollback_model():
    if not BACKUP_PATH.exists():
        raise FileNotFoundError(
            "No model backup is available"
        )

    shutil.copy2(
        BACKUP_PATH,
        MODEL_PATH,
    )

    records = _load_audit()

    records.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "ROLLBACK",
        "reason": "Previous production model restored",
    })

    _save_audit(records)

    return True


def audit_history():
    return _load_audit()


if __name__ == "__main__":
    print("MODEL ROLLBACK + AUDIT: OK")
    print("MODEL EXISTS:", MODEL_PATH.exists())
    print("BACKUP EXISTS:", BACKUP_PATH.exists())
    print("AUDIT ENTRIES:", len(_load_audit()))
