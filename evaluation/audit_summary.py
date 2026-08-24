import json
from pathlib import Path

BASE = Path("evaluation")

with open(BASE / "model_promotion.json", encoding="utf-8") as f:
    promotion = json.load(f)

with open(BASE / "model_audit.json", encoding="utf-8") as f:
    audit = json.load(f)

report = {
    "historical_audit": {
        "total_events": len(audit),
        "rejected_candidates": sum(
            1 for x in audit if x.get("decision") == "REJECT"
        ),
        "promotions_recorded_in_historical_audit": sum(
            1 for x in audit if x.get("decision") == "PROMOTE"
        ),
        "rollbacks_recorded_in_historical_audit": sum(
            1 for x in audit if x.get("decision") == "ROLLBACK"
        ),
    },

    "current_model_decision": {
        "candidate_promoted": promotion.get("promoted", False),
        "rollback_available": promotion.get(
            "rollback_available", False
        ),
        "old_roc_auc": promotion.get("old_roc_auc"),
        "candidate_roc_auc": promotion.get(
            "candidate_roc_auc"
        ),
    },

    "interpretation": (
        "The historical audit contains prior candidate-rejection "
        "events. The current promotion state is represented "
        "separately by model_promotion.json."
    ),
}

with open(BASE / "audit_summary.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("AUDIT SUMMARY: OK")
print()
print("Historical audit events:", len(audit))
print(
    "Historical rejections:",
    report["historical_audit"]["rejected_candidates"]
)
print(
    "Current candidate promoted:",
    report["current_model_decision"]["candidate_promoted"]
)
print(
    "Rollback available:",
    report["current_model_decision"]["rollback_available"]
)
