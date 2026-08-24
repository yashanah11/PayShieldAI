from pathlib import Path

from detection.model_rollback import (
    _save_audit,
    audit_history,
    promote_candidate,
)


class FakeModel:
    def predict_proba(self, X):
        return None


def test_reject_invalid_candidate():
    promoted = promote_candidate(
        object(),
        0.90,
        0.95,
    )

    assert promoted is False


def test_reject_worse_candidate():
    promoted = promote_candidate(
        FakeModel(),
        0.95,
        0.90,
    )

    assert promoted is False


def test_audit_storage(tmp_path, monkeypatch):
    import detection.model_rollback as module

    audit_file = tmp_path / "audit.json"
    monkeypatch.setattr(module, "AUDIT_PATH", audit_file)

    _save_audit([
        {
            "decision": "TEST",
        }
    ])

    assert len(audit_history()) == 1
    assert audit_history()[0]["decision"] == "TEST"
