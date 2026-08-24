import json
from datetime import datetime, timezone
from pathlib import Path

from arena.detector_runner import run_arena_round
from arena.pipeline import run_pipeline
from arena.adaptive import run_adaptive_arena
from detection.self_improvement import retrain_with_hard_examples
from detection.model_promotion import run_model_promotion
from detection.model_rollback import audit_history


EVALUATION_DIR = Path("evaluation")


def load_json(filename):
    path = EVALUATION_DIR / filename

    if not path.exists():
        return {
            "status": "MISSING",
            "file": filename,
        }

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_end_to_end(
    seed=42,
    pipeline_n=1000,
    adaptive_rounds=3,
    adaptive_n=1000,
    arena_n=1000,
):
    EVALUATION_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("PAYSHIELD AI — END-TO-END EXPERIMENT")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Closed-loop detection pipeline
    # ---------------------------------------------------------
    print("\n[1/6] CLOSED-LOOP PIPELINE")

    run_pipeline(
        n=pipeline_n,
        seed=seed,
    )

    pipeline_results = load_json(
        "pipeline_results.json"
    )

    # ---------------------------------------------------------
    # 2. Single adversarial arena round
    # ---------------------------------------------------------
    print("\n[2/6] ADVERSARIAL ARENA")

    arena_result = run_arena_round(
        round_number=1,
        n=arena_n,
        seed=seed,
    )

    arena_results = {
        "round_number": arena_result.round_number,
        "attack_count": arena_result.attack_count,
        "detected_count": arena_result.detected_count,
        "detection_rate": arena_result.detection_rate,
    }

    print(
        f"ARENA DETECTION RATE: "
        f"{arena_result.detection_rate:.4f}"
    )

    # ---------------------------------------------------------
    # 3. Adaptive adversarial arena
    # ---------------------------------------------------------
    print("\n[3/6] ADAPTIVE ARENA")

    run_adaptive_arena(
        rounds=adaptive_rounds,
        n=adaptive_n,
        seed=seed,
    )

    adaptive_results = load_json(
        "adaptive_arena_results.json"
    )

    # ---------------------------------------------------------
    # 4. Hard-example self improvement
    # ---------------------------------------------------------
    print("\n[4/6] SELF-IMPROVEMENT")

    retrain_with_hard_examples(
        n=10000,
        seed=seed,
    )

    self_improvement = load_json(
        "self_improvement.json"
    )

    # ---------------------------------------------------------
    # 5. Safe model promotion
    # ---------------------------------------------------------
    print("\n[5/6] MODEL PROMOTION")

    run_model_promotion(
        seed=seed,
    )

    promotion_results = load_json(
        "model_promotion.json"
    )

    # ---------------------------------------------------------
    # 6. Audit trail
    # ---------------------------------------------------------
    print("\n[6/6] MODEL AUDIT")

    audit_results = audit_history()

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------
    report = {
        "experiment": "PayShield AI End-to-End Experiment",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed": seed,

        "configuration": {
            "pipeline_transactions": pipeline_n,
            "adaptive_rounds": adaptive_rounds,
            "adaptive_transactions_per_round": adaptive_n,
            "arena_transactions": arena_n,
        },

        "pipeline": pipeline_results,

        "arena": arena_results,

        "adaptive_arena": adaptive_results,

        "self_improvement": self_improvement,

        "model_promotion": promotion_results,

        "model_audit": audit_results,

        "status": "COMPLETED",
    }

    output_path = (
        EVALUATION_DIR /
        "end_to_end_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("END-TO-END EXPERIMENT: COMPLETE")
    print("=" * 60)
    print(
        f"RESULTS: {output_path}"
    )


if __name__ == "__main__":
    run_end_to_end()
