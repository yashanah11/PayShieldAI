import json
from pathlib import Path

BASE = Path("evaluation")


def load(name):
    with open(BASE / name, encoding="utf-8") as f:
        return json.load(f)


baseline = load("baseline_metrics.json")
adaptive = load("adaptive_arena_results.json")
self_improvement = load("self_improvement.json")
promotion = load("model_promotion.json")
audit = load("model_audit.json")


rounds = adaptive.get("history", [])

rates = [
    float(r["detection_rate"])
    for r in rounds
]

best_rate = max(rates) if rates else None
worst_rate = min(rates) if rates else None
latest_rate = rates[-1] if rates else None

old_auc = float(self_improvement["old_roc_auc"])
new_auc = float(self_improvement["new_roc_auc"])

auc_delta = new_auc - old_auc

promotion_delta = (
    float(promotion["candidate_roc_auc"])
    - float(promotion["old_roc_auc"])
)

report = {
    "report": "PayShield AI Quantitative Evaluation",
    "data_type": "synthetic",

    "baseline": {
        "precision": baseline["precision"],
        "recall": baseline["recall"],
        "f1": baseline["f1"],
        "roc_auc": baseline["roc_auc"],
        "false_positive_rate": baseline["false_positive_rate"],
        "true_negatives": baseline["true_negatives"],
        "false_positives": baseline["false_positives"],
        "false_negatives": baseline["false_negatives"],
        "true_positives": baseline["true_positives"],
    },

    "adaptive_adversarial_evaluation": {
        "rounds": adaptive["rounds"],
        "transactions_per_round": adaptive["transactions_per_round"],
        "best_detection_rate": best_rate,
        "worst_detection_rate": worst_rate,
        "latest_detection_rate": latest_rate,
        "round_to_round": [
            {
                "round": r["round"],
                "attacks": r["attacks"],
                "detected": r["detected"],
                "detection_rate": r["detection_rate"],
            }
            for r in rounds
        ],
    },

    "self_improvement": {
        "training_transactions": self_improvement[
            "training_transactions"
        ],
        "hard_examples_found": self_improvement[
            "hard_examples_found"
        ],
        "augmented_training_size": self_improvement[
            "augmented_training_size"
        ],
        "old_roc_auc": old_auc,
        "new_roc_auc": new_auc,
        "roc_auc_delta": auc_delta,
        "interpretation": (
            "performance maintained"
            if auc_delta == 0
            else "performance improved"
            if auc_delta > 0
            else "performance decreased"
        ),
    },

    "model_promotion": {
        "old_roc_auc": promotion["old_roc_auc"],
        "candidate_roc_auc": promotion[
            "candidate_roc_auc"
        ],
        "roc_auc_delta": promotion_delta,
        "promoted": promotion["promoted"],
        "rollback_available": promotion[
            "rollback_available"
        ],
    },

    "audit": {
        "total_events": len(audit),
        "rejections": sum(
            1 for x in audit
            if x.get("decision") == "REJECT"
        ),
        "promotions": sum(
            1 for x in audit
            if x.get("decision") == "PROMOTE"
        ),
        "rollbacks": sum(
            1 for x in audit
            if x.get("decision") == "ROLLBACK"
        ),
    },

    "research_interpretation": {
        "baseline_is_perfect": baseline["roc_auc"] == 1.0,
        "self_improvement_delta": auc_delta,
        "adaptive_min_detection_rate": worst_rate,
        "adaptive_recovered_to_latest": (
            latest_rate == best_rate
            if latest_rate is not None
            else False
        ),
        "hard_examples_available": (
            self_improvement["hard_examples_found"] > 0
        ),
        "caveat": (
            "The current synthetic benchmark produces perfect "
            "baseline performance and zero hard examples. "
            "Therefore the system demonstrates the evaluation, "
            "adversarial testing, promotion, and audit mechanisms, "
            "but does not establish a measurable model-performance "
            "improvement over baseline."
        ),
    },

    "status": "COMPLETE",
}


output = BASE / "quantitative_report.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("QUANTITATIVE REPORT: OK")
print("BASELINE ROC-AUC:", baseline["roc_auc"])
print("ADAPTIVE BEST:", best_rate)
print("ADAPTIVE WORST:", worst_rate)
print("ROC-AUC DELTA:", auc_delta)
print("PROMOTED:", promotion["promoted"])
print("AUDIT EVENTS:", len(audit))
print("REPORT:", output)
