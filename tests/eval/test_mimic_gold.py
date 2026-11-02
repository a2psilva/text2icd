"""The gold annotation tools, on hand-written synthetic phrases."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from text2icd.eval.datasets import Example
from text2icd.eval.mimic.annotate import (
    Answer,
    Candidate,
    Session,
    agreement_with_silver,
    answers_to_jsonl,
    build_gold,
    frequency_bands,
    parse_answer,
    read_answers,
    read_candidates,
    sample_candidates,
    write_jsonl,
)
from text2icd.eval.mimic.splits import split_of_subject


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


def test_frequency_bands() -> None:
    examples = [silver(i, "I10") for i in range(6)]
    examples += [silver(10 + i, "E78.5") for i in range(3)]
    examples += [silver(20, "J18.9")]
    assert frequency_bands(examples) == {"I10": "head", "E78.5": "torso", "J18.9": "tail"}


def test_sample_candidates() -> None:
    examples = [silver(i, "I10") for i in range(30)]
    examples += [silver(100 + i, "E78.5") for i in range(15)]
    examples += [silver(200 + i, "J18.9") for i in range(4)]
    examples += [silver(300, "J18.9", note="n200")]  # same note as another item

    def chapter(e: Example) -> str:
        return {"I": "9", "E": "4", "J": "10"}[e.primary.code[0]]

    candidates = sample_candidates(
        examples, lambda e: [(e.primary.code, "description")], chapter, size=9
    )
    bands = [c.band for c in candidates]
    assert bands.count("head") == 3
    assert bands.count("torso") == 3
    assert bands.count("tail") == 3
    assert len({c.note_id for c in candidates}) == len(candidates)
    assert candidates[0].choices == ((candidates[0].silver, "description"),)
    assert sample_candidates(examples, lambda e: [], chapter, size=9) != []


def candidate() -> Candidate:
    return Candidate(
        id="n1:0",
        phrase="DM2 with CKD stage 3a",
        subject_id="1",
        hadm_id="101",
        note_id="n1",
        silver="E11.22",
        band="head",
        chapter="4",
        choices=(("E11.22", "Type 2 diabetes mellitus with diabetic CKD"), ("I10", "HTN")),
    )


BILLABLE = {"E11.22", "N18.30", "I10"}.__contains__


@pytest.mark.parametrize(
    ("text", "codes", "verdict"),
    [
        ("1", ("E11.22",), "coded"),
        ("1 n1830", ("E11.22", "N18.30"), "coded"),
        ("E11.22, 1", ("E11.22",), "coded"),
        ("s a plan, not a diagnosis", (), "skip"),
        ("?", (), "unsure"),
    ],
)
def test_parse_answers(text: str, codes: tuple[str, ...], verdict: str) -> None:
    parsed = parse_answer(text, candidate(), BILLABLE)
    assert parsed.answer is not None
    assert (parsed.answer.codes, parsed.answer.verdict) == (codes, verdict)


@pytest.mark.parametrize(
    ("text", "diagnoses"),
    [
        ("1", 1),
        ("1 n1830", 1),  # a combination code and its companion: one diagnosis
        ("1 n1830, 2", 2),
        ("I10, E11.22 N18.30", 2),
        ("E11.22, 1", 1),  # the same code twice
        (",1,", 1),
    ],
)
def test_diagnoses(text: str, diagnoses: int) -> None:
    parsed = parse_answer(text, candidate(), BILLABLE)
    assert parsed.answer is not None
    assert parsed.answer.diagnoses == diagnoses
    skipped = parse_answer("s plan", candidate(), BILLABLE).answer
    assert skipped is not None
    assert skipped.diagnoses is None


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("", "type codes"),
        ("3", "no candidate 3"),
        ("E11.9", "E11.9 is not a billable code"),
        ("diabetes", "neither a candidate number nor a code"),
        ("l", "l needs words"),
    ],
)
def test_parse_errors(text: str, error: str) -> None:
    parsed = parse_answer(text, candidate(), BILLABLE)
    assert parsed.error is not None
    assert error in parsed.error


def test_parse_commands() -> None:
    assert parse_answer("q", candidate(), BILLABLE).command == "quit"
    assert parse_answer("b", candidate(), BILLABLE).command == "back"
    lookup = parse_answer("l diabetes ckd", candidate(), BILLABLE)
    assert (lookup.command, lookup.argument) == ("lookup", "diabetes ckd")
    assert parse_answer("c E11.22", candidate(), BILLABLE).command == "show"


def test_session(tmp_path: Path) -> None:
    second = candidate().model_copy(update={"id": "n2:0", "subject_id": "2", "note_id": "n2"})
    third = candidate().model_copy(update={"id": "n3:0", "subject_id": "3", "note_id": "n3"})
    path = tmp_path / "answers.jsonl"
    typed: Iterator[str] = iter(["h", "l ckd", "c I10", "zzz", "1 N18.30", "b", "1", "?", "q"])
    shown: list[str] = []
    session = Session(
        candidates=[candidate(), second, third],
        answers={},
        billable=BILLABLE,
        save=answers_to_jsonl(path),
        lookup=lambda words: [f"Index: {words}"],
        show=lambda code: [f"Code: {code}"],
        ask=lambda prompt: next(typed),
        say=shown.append,
    )
    assert session.run() == 3
    answers = read_answers(path)
    assert answers["n1:0"].codes == ("E11.22",)  # answered, then changed after going back
    assert answers["n2:0"].verdict == "unsure"
    assert "n3:0" not in answers
    assert any("Index: ckd" in line for line in shown)
    assert any("(answered: E11.22 N18.30)" in line for line in shown)
    # Resuming starts at the first unanswered phrase.
    resumed = Session(
        candidates=[candidate(), second, third],
        answers=answers,
        billable=BILLABLE,
        save=answers_to_jsonl(path),
        lookup=lambda words: [],
        show=lambda code: [],
        ask=lambda prompt: "s",
        say=lambda line: None,
    )
    assert resumed.run() == 1
    assert read_answers(path)["n3:0"].verdict == "skip"


def test_build_gold(tmp_path: Path) -> None:
    candidates = [
        candidate(),
        candidate().model_copy(update={"id": "n2:0", "subject_id": "2"}),
        candidate().model_copy(update={"id": "n3:0", "subject_id": "3"}),
    ]
    path = tmp_path / "candidates.jsonl"
    write_jsonl(path, candidates)
    assert read_candidates(path) == candidates
    answers = {
        "n1:0": Answer(id="n1:0", verdict="coded", codes=("E11.22", "N18.30"), diagnoses=1),
        "n2:0": Answer(id="n2:0", verdict="skip"),
    }
    examples, verdicts = build_gold(candidates, answers, "fy2027")
    assert verdicts == {"coded": 1, "skip": 1, "unanswered": 1}
    (gold,) = examples
    assert gold.meta["diagnoses"] == 1
    assert [(g.code, g.role) for g in gold.gold] == [("E11.22", "primary"), ("N18.30", "companion")]
    assert gold.source == "mimic-gold"
    assert gold.split == split_of_subject("1")
    assert agreement_with_silver(examples) == "primary code equals the silver code for 100.0% of 1"
    assert agreement_with_silver([]) == "no coded phrases"


def test_gold_commands(graph_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import csv

    from typer.testing import CliRunner

    from text2icd import cli
    from text2icd.config import Settings
    from text2icd.eval.datasets import read_dataset, write_dataset
    from text2icd.eval.versions import Reconciler

    data = tmp_path / "data"
    settings = Settings(data_dir=data)
    # Skewed counts, so that the three frequency bands all have codes.
    codes = ["E11.22", "J69.0", "I43", "J12.9", "E85.4"]
    weights = [12, 6, 3, 2, 1]
    sequence = [code for code, n in zip(codes, weights, strict=True) for _ in range(n)]
    examples = [
        silver(i, code).model_copy(update={"release": "fy2099"}) for i, code in enumerate(sequence)
    ]
    write_dataset(settings.eval_dir() / "mimic-silver.jsonl", examples)
    mimic = data / "raw" / "mimic-iv"
    mimic.mkdir(parents=True)
    with (mimic / "diagnoses_icd.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
        for e in examples:
            writer.writerow([e.meta["subject_id"], e.meta["hadm_id"], 1, e.primary.code, 10])
            writer.writerow([e.meta["subject_id"], e.meta["hadm_id"], 2, "N18.3", 10])
    reconciler = Reconciler(
        "fy2099",
        {"fy2016": frozenset({"N18.3"}), "fy2099": frozenset(codes)},
        [],
    )
    monkeypatch.setattr(cli, "load_reconciler", lambda release, settings: reconciler)
    monkeypatch.setattr(cli, "_full_release", lambda release: type("R", (), {"id": "fy2099"})())
    runner = CliRunner()
    base = ["--data-dir", str(data)]
    sampled = runner.invoke(
        cli.app, ["eval", "mimic", "gold", "sample", "-n", "6", "--graph", str(graph_file), *base]
    )
    assert sampled.exit_code == 0, sampled.output
    assert "5 phrases (head 2, torso 2, tail 1; 1 chapters)" in sampled.output
    folder = settings.eval_dir() / "annotation"
    first = read_candidates(folder / "candidates.jsonl")[0]
    assert first.choices[0][0] == first.silver
    # N18.3 is no longer billable; its subdivisions would replace it, but the release has none.
    assert all(code != "N18.3" for code, _ in first.choices)

    typed = "l pneumonia\nc E11.22\n1\ns plan\n?\nq\n"
    annotated = runner.invoke(
        cli.app,
        ["eval", "mimic", "gold", "annotate", "--graph", str(graph_file), *base],
        input=typed,
    )
    assert annotated.exit_code == 0, annotated.output
    assert "3 phrases answered this time; 3 of 5" in annotated.output
    assert "index: Pneumonia" in annotated.output
    built = runner.invoke(cli.app, ["eval", "mimic", "gold", "build", *base])
    assert built.exit_code == 0, built.output
    assert "1 examples (coded 1, skip 1, unsure 1, unanswered 2)" in built.output
    assert "primary code equals the silver code for 100.0% of 1" in built.output
    (gold,) = read_dataset(settings.eval_dir() / "mimic-gold.jsonl")
    assert gold.source == "mimic-gold"
    again = runner.invoke(
        cli.app, ["eval", "mimic", "gold", "sample", "--graph", str(graph_file), *base]
    )
    assert again.exit_code == 1
    assert "already has answers" in again.output
    empty = ["--data-dir", str(tmp_path / "none")]
    for command in ("annotate", "build"):
        assert runner.invoke(cli.app, ["eval", "mimic", "gold", command, *empty]).exit_code == 1
