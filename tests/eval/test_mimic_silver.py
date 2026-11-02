"""The silver-label pipeline on hand-written synthetic notes and codes (no MIMIC-IV data)."""

import csv
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.embeddings import Vectors, normalize
from text2icd.eval import runner as runner_module
from text2icd.eval.datasets import Example, write_dataset
from text2icd.eval.mimic.align import assign
from text2icd.eval.mimic.files import Note
from text2icd.eval.mimic.gold import BilledCode, admission_codes
from text2icd.eval.mimic.silver import build_silver
from text2icd.eval.mimic.splits import (
    FrozenSplitError,
    check_frozen,
    freeze,
    frozen,
    split_of_subject,
)
from text2icd.eval.runner import RunError, evaluate
from text2icd.eval.systems import BM25System
from text2icd.eval.versions import Reconciler
from text2icd.graph.model import Graph


def test_admission_codes(tmp_path: Path) -> None:
    path = tmp_path / "diagnoses_icd.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
        writer.writerows(
            [
                ["1", "11", "2", "I10", "10"],
                ["1", "11", "1", "J189", "10"],
                ["1", "11", "3", "4019", "9"],
                ["1", "11", "4", "J189", "10"],
                ["1", "11", "5", "HIV", "10"],
                ["2", "22", "1", "N183", "10"],
            ]
        )
    found = admission_codes(path)
    assert found["11"] == [BilledCode(1, "J18.9"), BilledCode(2, "I10")]
    assert admission_codes(path, {"22"}) == {"22": [BilledCode(1, "N18.3")]}


def test_assign() -> None:
    similarity = np.array(
        [
            [0.95, 0.20, 0.10],  # item 0 clearly code 0
            [0.80, 0.78, 0.10],  # item 1 close to codes 0 and 1: margin too small
            [0.10, 0.20, 0.60],  # item 2 below the score
            [0.10, 0.90, 0.10],  # item 3 clearly code 1
        ],
        dtype=np.float32,
    )
    found = assign(similarity, min_score=0.7, min_margin=0.05)
    assert [(a.item, a.code) for a in found] == [(0, 0), (3, 1)]
    assert found[0].margin == pytest.approx(0.75)
    assert assign(np.zeros((0, 2), dtype=np.float32)) == []
    (single,) = assign(np.array([[0.9]], dtype=np.float32))
    assert single.margin == 1.0


def test_splits_by_patient(tmp_path: Path) -> None:
    splits = [split_of_subject(str(n)) for n in range(1000)]
    assert 400 < splits.count("test") < 600
    assert split_of_subject("123") == split_of_subject("123")


def example(i: int, split: str) -> Example:
    return Example.model_validate(
        {
            "id": f"n{i}",
            "phrase": "type 2 diabetes mellitus with diabetic chronic kidney disease",
            "gold": [{"code": "E11.22", "role": "primary"}],
            "release": "fy2099",
            "source": "mimic-silver",
            "split": split,
        }
    )


def test_freeze(tmp_path: Path) -> None:
    examples = [example(1, "dev"), example(2, "test")]
    digest = freeze(tmp_path, "mimic-silver", examples)
    assert frozen(tmp_path) == {"mimic-silver": digest}
    assert freeze(tmp_path, "mimic-silver", [*examples, example(3, "dev")]) == digest
    check_frozen(tmp_path, "mimic-silver", [*examples, example(4, "dev")])
    with pytest.raises(FrozenSplitError):
        check_frozen(tmp_path, "mimic-silver", [*examples, example(5, "test")])
    with pytest.raises(FrozenSplitError):
        freeze(tmp_path, "mimic-silver", [example(6, "test")])
    check_frozen(tmp_path, "other", [example(7, "test")])


def test_runner_refuses_a_changed_frozen_test_split(graph: Graph, tmp_path: Path) -> None:
    path = tmp_path / "mimic-silver.jsonl"
    write_dataset(path, [example(1, "dev"), example(2, "test")])
    eval_dir = tmp_path / "eval"
    freeze(eval_dir, "mimic-silver", [example(1, "dev"), example(3, "test")])
    system = BM25System(graph, {})
    evaluate(system, graph, path, split="dev", resamples=0, frozen_dir=eval_dir)
    with pytest.raises(RunError, match="differs from the one frozen"):
        evaluate(system, graph, path, split="test", resamples=0, frozen_dir=eval_dir)
    assert runner_module.is_restricted([example(1, "dev")])


class Words:
    """A stand-in embedder: hashed words, so a phrase and a description sharing words match."""

    model = "test/words"

    def encode(self, texts: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(texts), 128), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.casefold().replace(",", " ").split():
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1
        return normalize(vectors)


NOTE = """Discharge Diagnosis:
Primary:
- pneumonia due to streptococcus pneumoniae
Secondary:
- essential hypertension
- chronic kidney disease stage 3
- something never billed

Discharge Condition:
stable
"""


def reconciler() -> Reconciler:
    billable = {
        "fy2016": frozenset({"J13", "I10", "N18.3"}),
        "fy2099": frozenset({"J13", "I10", "N18.30", "N18.31"}),
    }
    descriptions = {
        "J13": "Pneumonia due to Streptococcus pneumoniae",
        "I10": "Essential (primary) hypertension",
        "N18.3": "Chronic kidney disease, stage 3 (moderate)",
    }
    return Reconciler("fy2099", billable, [], descriptions)


def test_build_silver() -> None:
    notes = [
        Note("10-DS-1", "10", "100", NOTE),
        Note("11-DS-1", "11", "111", NOTE),  # no codes for this admission
        Note("12-DS-1", "12", "122", "Chief Complaint:\ncough"),  # no section
    ]
    codes = {
        "100": [BilledCode(1, "J13"), BilledCode(2, "I10"), BilledCode(3, "N18.3")],
        "122": [BilledCode(1, "I10")],
    }
    examples, stats = build_silver(notes, codes, reconciler(), Words(), min_score=0.5, chunk=1)
    by_phrase = {e.phrase: e for e in examples}
    assert set(by_phrase) == {
        "pneumonia due to streptococcus pneumoniae",
        "essential hypertension",
        "chronic kidney disease stage 3",
    }
    ckd = by_phrase["chronic kidney disease stage 3"].primary
    assert (ckd.code, ckd.status, ckd.mapped_to) == ("N18.3", "mapped", ("N18.30", "N18.31"))
    pneumonia = by_phrase["pneumonia due to streptococcus pneumoniae"]
    assert pneumonia.id == "10-DS-1:0"
    assert pneumonia.meta["group"] == "primary"
    assert pneumonia.meta["seq_num"] == 1
    assert pneumonia.split == split_of_subject("10")
    assert pneumonia.source == "mimic-silver"
    assert (stats.notes, stats.with_codes, stats.with_items, stats.items) == (3, 2, 1, 4)
    assert stats.aligned == 3
    assert stats.first_code == 1
    assert stats.statuses == {"valid": 2, "mapped": 1}
    assert "aligned 3 (75.0%)" in "\n".join(stats.lines())


def test_silver_and_freeze_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    mimic = data / "raw" / "mimic-iv"
    mimic.mkdir(parents=True)
    with (mimic / "discharge.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "note_id",
                "subject_id",
                "hadm_id",
                "note_type",
                "note_seq",
                "charttime",
                "storetime",
                "text",
            ]
        )
        writer.writerow(["10-DS-1", "10", "100", "DS", "1", "", "", NOTE])
    with (mimic / "diagnoses_icd.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
        writer.writerows([["10", "100", "1", "J13", "10"], ["10", "100", "2", "I10", "10"]])
    monkeypatch.setattr(cli, "load_reconciler", lambda release, settings: reconciler())
    monkeypatch.setattr(cli, "OnnxEmbedder", lambda model, file: Words())
    monkeypatch.setattr(cli, "_full_release", lambda release: type("R", (), {"id": "fy2099"})())
    runner = CliRunner()
    args = ["eval", "mimic", "silver", "--min-score", "0.5", "--data-dir", str(data)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    output = Settings(data_dir=data).eval_dir() / "mimic-silver.jsonl"
    assert output.is_file()
    assert "aligned 2" in result.output
    assert "pneumonia" not in result.output
    froze = runner.invoke(
        cli.app, ["eval", "mimic", "freeze", "-d", str(output), "--data-dir", str(data)]
    )
    assert froze.exit_code == 0, froze.output
    assert "froze the test split of mimic-silver" in froze.output
    missing = runner.invoke(cli.app, ["eval", "mimic", "silver", "--data-dir", str(tmp_path)])
    assert missing.exit_code == 1
