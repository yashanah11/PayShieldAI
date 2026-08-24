import json
from pathlib import Path

from evaluation.end_to_end import load_json


def test_end_to_end_result_loader_missing():
    result = load_json(
        "definitely_missing_file.json"
    )

    assert result["status"] == "MISSING"


def test_end_to_end_result_loader_existing(tmp_path, monkeypatch):
    import evaluation.end_to_end as module

    test_dir = tmp_path / "evaluation"
    test_dir.mkdir()

    test_file = test_dir / "test.json"

    test_file.write_text(
        json.dumps({"status": "OK"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "EVALUATION_DIR",
        test_dir,
    )

    result = load_json("test.json")

    assert result["status"] == "OK"
