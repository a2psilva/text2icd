"""The silver-label audit, on hand-written synthetic phrases."""

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.eval.datasets import Example, write_dataset
from text2icd.eval.mimic import audit as audit_module
from text2icd.eval.mimic.audit import (
    by_score,
    precision,
    read_review,
    review,
    sample_for_review,
    save_review,
    wilson,
    write_review,
)


def silver(i: int, code: str, split: str = "dev", note: str | None = None) -> Example:
    data: dict[str, Any] = {
        "id": f"n{i}:0",
        "phrase": f"phrase {i}",
        "gold": [{"code": code, "role": "primary"}],
        "release": "fy2027",
        "source": "mimic-silver",
        "split": split,
        "meta": {
            "note_id": note or f"n{i}",
            "subject_id": str(i),
            "hadm_id": str(100 + i),
            "score": 0.70 + (i % 30) / 100,
        },
    }
    return Example.model_validate(data)


def test_wilson() -> None:
    low, high = wilson(180, 200)
    assert low == pytest.approx(0.8505, abs=1e-3)
    assert high == pytest.approx(0.9346, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == pytest.approx(1.0)


def test_review_file(tmp_path: Path) -> None:
    examples = [silver(i, "I10", split="dev" if i % 2 else "test") for i in range(40)]
    sample = sample_for_review(examples, size=10, seed=1)
    assert len(sample) == 10
    assert all(e.split == "dev" for e in sample)
    assert sample == sample_for_review(examples, size=10, seed=1)
    path = tmp_path / "audit" / "review.csv"
    assert write_review(path, sample, lambda code: "Essential (primary) hypertension") == 10
    rows = read_review(path)
    assert rows[0]["description"] == "Essential (primary) hypertension"
    for row, verdict in zip(rows, ["y"] * 7 + ["n", "?", ""], strict=True):
        row["verdict"] = verdict
    save_review(path, rows)
    result = precision(read_review(path))
    assert (result.right, result.wrong, result.unsure, result.unreviewed) == (7, 1, 1, 1)
    assert result.line().startswith("precision: 87.5% (95% CI ")
    assert by_score(rows, [0.7, 0.9])[0].startswith("score 0.70-0.90:")
    with pytest.raises(FileExistsError, match="already has verdicts"):
        write_review(path, sample, lambda code: "")
    assert precision([]).line() == "precision: nothing reviewed yet (0 to review)"


def test_review_loop(tmp_path: Path) -> None:
    rows = [
        {
            "id": str(i),
            "phrase": f"p{i}",
            "code": "I10",
            "description": "d",
            "score": "0.8",
            "verdict": "",
            "comment": "",
        }
        for i in range(3)
    ]
    rows[0]["verdict"] = "y"
    typed = iter(["maybe", "n wrong organ", "b", "y", "?", "q"])
    saved: list[int] = []
    shown: list[str] = []
    given = review(rows, lambda prompt: next(typed), shown.append, lambda r: saved.append(1))
    assert given == 3
    assert [r["verdict"] for r in rows] == ["y", "y", "?"]
    assert any("cannot tell" in line for line in shown)
    assert len(saved) == 3
    assert review(rows, lambda prompt: "y", shown.append, lambda r: None) == 0


def test_audit_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    settings = Settings(data_dir=data)
    write_dataset(settings.eval_dir() / "mimic-silver.jsonl", [silver(i, "I10") for i in range(20)])

    descriptions = SimpleNamespace(descriptions={"I10": "Essential (primary) hypertension"})
    monkeypatch.setattr(cli, "load_reconciler", lambda release, settings: descriptions)
    monkeypatch.setattr(cli, "_full_release", lambda release: type("R", (), {"id": "fy2027"})())
    runner = CliRunner()
    base = ["--data-dir", str(data)]
    sampled = runner.invoke(cli.app, ["eval", "mimic", "audit", "sample", "-n", "5", *base])
    assert sampled.exit_code == 0, sampled.output
    assert "5 rows to review" in sampled.output
    path = settings.eval_dir() / "audit" / "mimic-silver.csv"
    with path.open() as fh:
        assert len(list(csv.DictReader(fh))) == 5
    reviewed = runner.invoke(
        cli.app, ["eval", "mimic", "audit", "review", *base], input="y\ny\nn\nq\n"
    )
    assert reviewed.exit_code == 0, reviewed.output
    assert "3 verdicts given" in reviewed.output
    scored = runner.invoke(cli.app, ["eval", "mimic", "audit", "score", *base])
    assert "precision: 66.7%" in scored.output
    again = runner.invoke(cli.app, ["eval", "mimic", "audit", "sample", *base])
    assert again.exit_code == 1
    assert "already has verdicts" in again.output
    for command in ("review", "score"):
        missing = runner.invoke(
            cli.app, ["eval", "mimic", "audit", command, "--data-dir", str(tmp_path)]
        )
        assert missing.exit_code == 1
    assert audit_module.SAMPLE_SIZE == 200
