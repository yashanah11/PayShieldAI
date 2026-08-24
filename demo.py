import json
from pathlib import Path

BASE = Path("evaluation")


def load_json(name):
    path = BASE / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def main():
    baseline = load_json("baseline_metrics.json")
    pipeline = load_json("pipeline_results.json")
    adaptive = load_json("adaptive_arena_results.json")
    improvement = load_json("self_improvement.json")
    promotion = load_json("model_promotion.json")
    audit = load_json("model_audit.json")
    quantitative = load_json("quantitative_report.json")

    print()
    print("=" * 72)
    print("                 PAYSHIELD AI")
    print("        ADVERSARIAL FRAUD DEFENSE DEMO")
    print("=" * 72)

    print()
    print("1. ATTACK")
    print("-" * 72)

    rounds = adaptive.get("history", [])
    total_attacks = sum(r.get("attacks", 0) for r in rounds)
    total_detected = sum(r.get("detected", 0) for r in rounds)

    print(f"Adversarial rounds : {len(rounds)}")
    print(f"Attack cases       : {total_attacks}")
    print(f"Detected attacks   : {total_detected}")

    print()
    print("2. DETECT")
    print("-" * 72)

    print(f"Baseline ROC-AUC   : {baseline.get('roc_auc', 'N/A'):.3f}")
    print(f"Precision          : {baseline.get('precision', 'N/A'):.3f}")
    print(f"Recall             : {baseline.get('recall', 'N/A'):.3f}")
    print(f"F1                 : {baseline.get('f1', 'N/A'):.3f}")
    print(
        f"False-positive rate: "
        f"{baseline.get('false_positive_rate', 'N/A'):.3f}"
    )

    print()
    print("3. EXPLAIN")
    print("-" * 72)

    explanations = pipeline.get("results_with_explanations", 0)
    predictions = pipeline.get("fraud_predictions", 0)

    print(f"Transactions analyzed : {pipeline.get('transactions_analyzed', 0)}")
    print(f"Fraud predictions     : {predictions}")
    print(f"SHAP explanations     : {explanations}")

    results = pipeline.get("results", [])

    if results:
        print()
        print("Top risk signals:")
        shown = set()

        for result in results:
            for feature in result.get("top_risk_features", []):
                name = feature.get("feature")

                if name and name not in shown:
                    shown.add(name)
                    contribution = feature.get("contribution", 0)
                    print(f"  • {name}: {contribution:.4f}")

    print()
    print("4. ADAPT")
    print("-" * 72)

    print(f"Rounds evaluated     : {adaptive.get('rounds', 0)}")
    print(f"Best detection rate  : {pct(adaptive.get('summary', {}).get('best_detection_rate'))}")
    print(f"Latest detection rate: {pct(adaptive.get('summary', {}).get('latest_detection_rate'))}")

    if rounds:
        print()
        for r in rounds:
            print(
                f"  Round {r['round']}: "
                f"{r['detected']}/{r['attacks']} "
                f"({pct(r['detection_rate'])})"
            )

    print()
    print("5. EVALUATE")
    print("-" * 72)

    old_auc = improvement.get("old_roc_auc", 0)
    new_auc = improvement.get("new_roc_auc", 0)
    delta = new_auc - old_auc

    print(f"Old ROC-AUC        : {old_auc:.3f}")
    print(f"New ROC-AUC        : {new_auc:.3f}")
    print(f"ROC-AUC delta      : {delta:+.3f}")
    print(f"Hard examples      : {improvement.get('hard_examples_found', 0)}")

    if delta > 0:
        interpretation = "MODEL IMPROVED"
    elif delta == 0:
        interpretation = "PERFORMANCE MAINTAINED"
    else:
        interpretation = "PERFORMANCE DECREASED"

    print(f"Result             : {interpretation}")

    print()
    print("6. PROMOTE / ROLLBACK")
    print("-" * 72)

    promoted = promotion.get("promoted", False)
    rollback = promotion.get("rollback_available", False)

    print(f"Candidate promoted : {'YES' if promoted else 'NO'}")
    print(f"Rollback available : {'YES' if rollback else 'NO'}")

    print()
    print("7. AUDIT")
    print("-" * 72)

    if isinstance(audit, list):
        rejects = sum(
            1 for x in audit
            if x.get("decision") == "REJECT"
        )
        promotions = sum(
            1 for x in audit
            if x.get("decision") == "PROMOTE"
        )
        rollbacks = sum(
            1 for x in audit
            if x.get("decision") == "ROLLBACK"
        )

        print(f"Audit events       : {len(audit)}")
        print(f"Rejected candidates: {rejects}")
        print(f"Promotions         : {promotions}")
        print(f"Rollbacks          : {rollbacks}")

    print()
    print("=" * 72)
    print("                    DEMO COMPLETE")
    print("=" * 72)
    print()
    print("PayShield AI continuously evaluates its fraud detector")
    print("against synthetic adversarial payment attacks.")
    print()
    print("IMPORTANT:")
    print("This demonstration uses synthetic benchmark data.")
    print("The current benchmark has perfect baseline ROC-AUC,")
    print("so no measurable ROC-AUC improvement is claimed.")
    print()


if __name__ == "__main__":
    main()
