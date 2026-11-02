import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from text2icd import cli
from text2icd.eval.datasets import (
    DatasetError,
    Example,
    GoldCode,
    check_dataset,
    dataset_hash,
    read_dataset,
    summarize,
    to_line,
    write_dataset,
)
from text2icd.ontology.files import ReleaseFiles


def example(**changes: Any) -> Example:
    data: dict[str, Any] = {
        "id": "x:1",
        "phrase": "type 2 diabetes with diabetic CKD",
        "gold": [{"code": "E1122", "role": "primary"}, {"code": "N18.30", "role": "companion"}],
        "release": "fy2027",
        "source": "test",
        "split": "dev",
    }
    data.update(changes)
    return Example.model_validate(data)


def test_example_normalizes_codes_and_phrase() -> None:
    item = example(phrase="  type 2\tdiabetes \n")
    assert item.phrase == "type 2 diabetes"
    assert item.primary == GoldCode(code="E11.22", role="primary")
    assert [gold.code for gold in item.companions] == ["N18.30"]
    assert item.primary.status == "valid"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gold": []}, "at least 1 item"),
        ({"gold": [{"code": "N18.30", "role": "companion"}]}, "exactly one primary code"),
        (
            {
                "gold": [
                    {"code": "E11.22", "role": "primary"},
                    {"code": "E1122", "role": "companion"},
                ]
            },
            "lists a code twice",
        ),
        ({"gold": [{"code": "HIV", "role": "primary"}]}, "not an ICD-10-CM code"),
        ({"gold": [{"code": "N18.3", "role": "primary", "status": "mapped"}]}, "mapped_to"),
        (
            {"gold": [{"code": "N18.3", "role": "primary", "mapped_to": ["N18.30"]}]},
            "mapped_to",
        ),
        ({"release": "2027"}, "release must look like"),
        ({"split": "train"}, "'dev' or 'test'"),
        ({"phrase": " \n"}, "phrase is empty"),
        ({"extra": 1}, "Extra inputs"),
    ],
)
def test_example_rejects(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        example(**changes)


def test_round_trip_leaves_out_defaults(tmp_path: Path) -> None:
    mapped = example(
        id="x:2",
        gold=[{"code": "N18.3", "role": "primary", "status": "mapped", "mapped_to": ["N1830"]}],
        meta={"generator": "test", "rank": 2},
    )
    path = tmp_path / "set.jsonl"
    assert write_dataset(path, [example(), mapped]) == 2
    first, second = path.read_text().splitlines()
    assert json.loads(first)["gold"][0] == {"code": "E11.22", "role": "primary"}
    assert "meta" not in json.loads(first)
    assert json.loads(second)["gold"][0]["mapped_to"] == ["N18.30"]
    assert read_dataset(path) == [example(), mapped]
    assert to_line(mapped) == second


def test_read_names_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(to_line(example()) + "\n\n" + '{"id": "x:3"}\n')
    with pytest.raises(DatasetError, match=r"bad\.jsonl:3: phrase: Field required"):
        read_dataset(path)


def test_hash_ignores_order_and_formatting() -> None:
    a, b = example(), example(id="x:2", split="test")
    assert dataset_hash([a, b]) == dataset_hash([b, a])
    assert dataset_hash([a, b]) != dataset_hash([a, example(id="x:2")])


def test_check_dataset() -> None:
    billable = {"E11.22", "N18.30", "N18.31"}.__contains__
    examples = [
        example(),
        example(),
        example(id="x:2", release="fy2026"),
        example(
            id="x:3",
            gold=[
                {"code": "N18.3", "role": "primary", "status": "mapped", "mapped_to": ["N18.39"]},
                {"code": "N18.31", "role": "companion", "status": "unmappable"},
            ],
        ),
        example(id="x:4", gold=[{"code": "E11.9", "role": "primary"}]),
    ]
    assert check_dataset(examples) == [
        "id 'x:1' appears 2 times",
        "more than one release: fy2026, fy2027",
    ]
    assert check_dataset(examples, billable)[2:] == [
        "x:3: N18.3 maps to codes that are not billable: N18.39",
        "x:3: N18.31 is marked unmappable but is billable",
        "x:4: E11.9 is marked valid but is not billable",
    ]


def test_summarize() -> None:
    summary = summarize([example(), example(id="x:2", split="test", source="other")])
    assert summary.splits == {"dev": 1, "test": 1}
    assert summary.sources == {"other": 1, "test": 1}
    assert summary.statuses == {"valid": 4, "mapped": 0, "unmappable": 0}
    assert summary.companions == 2
    assert summary.lines()[0] == "  2 examples, release fy2027"
    assert summarize([]).release is None


runner = CliRunner()


@pytest.fixture
def data_dir(fixture_release: ReleaseFiles, tmp_path: Path) -> Path:
    """A data directory whose `fy2027` holds the fixture order file."""
    root = tmp_path / "data"
    shutil.copytree(fixture_release.root, root / "raw" / "cms" / "fy2027")
    return root


def test_validate_command(data_dir: Path, tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    write_dataset(good, [example(gold=[{"code": "E11.22", "role": "primary"}])])
    result = runner.invoke(cli.app, ["eval", "validate", str(good), "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    assert "1 examples, release fy2027" in result.output
    assert "gold codes checked against fy2027: ok" in result.output


def test_validate_command_reports_problems(data_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    write_dataset(bad, [example()])
    result = runner.invoke(cli.app, ["eval", "validate", str(bad), "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "problem: x:1: N18.30 is marked valid but is not billable" in result.output
    missing = runner.invoke(
        cli.app, ["eval", "validate", str(bad), "--data-dir", str(tmp_path / "none")]
    )
    assert missing.exit_code == 0, missing.output
    assert "gold codes not checked: fy2027 is not fetched" in missing.output
    unreadable = runner.invoke(cli.app, ["eval", "validate", str(tmp_path / "absent.jsonl")])
    assert unreadable.exit_code == 1
    assert "error:" in unreadable.output
