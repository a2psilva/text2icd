"""Measuring text understanding on datasets of hand-written phrases (no MIMIC-IV data)."""

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.eval.datasets import Example, write_dataset
from text2icd.eval.parsing import expected_diagnoses, parse_stats
from text2icd.text.parse import Parser
from text2icd.text.terms import Terms

PARSER = Parser(Terms.build(["Hypertension", "Hyperlipidemia", "Injury", "Dehydration", "Nausea"]))


def example(i: int, phrase: str, source: str = "cms-synthetic", **meta: Any) -> Example:
    return Example.model_validate(
        {
            "id": f"x{i}",
            "phrase": phrase,
            "gold": [{"code": "I10", "role": "primary"}],
            "release": "fy2099",
            "source": source,
            "split": "dev",
            "meta": meta,
        }
    )


def test_expected_diagnoses() -> None:
    assert expected_diagnoses(example(1, "x")) == 1  # one per cms-synthetic phrase
    assert expected_diagnoses(example(2, "x", source="mimic-gold", diagnoses=2)) == 2
    assert expected_diagnoses(example(3, "x", source="mimic-gold")) is None  # not recorded
    assert expected_diagnoses(example(4, "x", source="mimic-silver")) is None
    assert expected_diagnoses(example(5, "x", source="mimic-gold", diagnoses=True)) is None


def test_parse_stats() -> None:
    examples = [
        example(1, "HTN"),
        example(2, "HTN, HLD"),  # expected one: too many
        example(3, "AKI due to dehydration", source="mimic-gold", diagnoses=1),
        example(4, "HTN, HLD", source="mimic-gold", diagnoses=2),
        example(5, "possible HTN", source="mimic-silver"),  # no expected number
    ]
    stats = parse_stats(examples, PARSER, keep=5)
    assert stats.phrases == 5
    assert stats.diagnoses == {1: 3, 2: 2}
    assert (stats.expected, stats.agreed, stats.too_many, stats.too_few) == (4, 3, 1, 0)
    assert stats.flags["uncertain"] == 1
    assert stats.relations["due_to"] == 1
    assert stats.modifiers["acuity"] == 1
    (mismatch,) = stats.mismatches
    assert (mismatch.id, mismatch.expected, mismatch.found) == ("x2", 1, 2)
    lines = "\n".join(stats.lines())
    assert "same number of diagnoses as expected: 75.0% (95% CI " in lines
    assert "diagnoses per phrase: 1: 60.0%, 2: 40.0%" in lines
    assert "phrases flagged: uncertain 20.0%" in lines
    assert parse_stats(examples, PARSER).mismatches == []
    empty = "\n".join(parse_stats([example(1, "x", source="mimic-silver")], PARSER).lines())
    assert "no phrase has an expected number" in empty


def test_parse_command(tmp_path: Path, graph_file: Path) -> None:
    data = tmp_path / "data"
    settings = Settings(data_dir=data)
    write_dataset(
        settings.eval_dir() / "set.jsonl",
        [example(1, "pneumonia"), example(2, "pneumonia, pyelitis")],
    )
    write_dataset(
        settings.eval_dir() / "mimic-gold.jsonl",
        [example(1, "pneumonia", source="mimic-gold", diagnoses=1)],
    )
    runner = CliRunner()
    base = ["--graph", str(graph_file), "--data-dir", str(data)]
    result = runner.invoke(cli.app, ["eval", "parse", "-d", "set", "--show", "5", *base])
    assert result.exit_code == 0, result.output
    assert "set.jsonl (dev), inpatient, 2 examples" in result.output
    assert "same number of diagnoses as expected: 50.0%" in result.output
    assert "x2: expected 1, found 2: pneumonia | pyelitis" in result.output
    restricted = runner.invoke(cli.app, ["eval", "parse", "-d", "mimic-gold", "--show", "1", *base])
    assert restricted.exit_code == 2
    assert "refused" in restricted.output
    gold = runner.invoke(cli.app, ["eval", "parse", "-d", "mimic-gold", *base])
    assert gold.exit_code == 0, gold.output
    assert "pneumonia" not in gold.output.split("examples", 1)[1]  # aggregates only
    bad = runner.invoke(cli.app, ["eval", "parse", "-d", "set", "--setting", "x", *base])
    assert bad.exit_code == 2
    missing = runner.invoke(cli.app, ["eval", "parse", "-d", "nothing", *base])
    assert missing.exit_code == 1
