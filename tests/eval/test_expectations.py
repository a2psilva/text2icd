import json
import math
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.eval.check import (
    Bound,
    Expectations,
    ExpectationsError,
    check,
    check_lines,
    load_expectations,
    measure,
)
from text2icd.eval.metrics import Estimate, Report, report_to_json

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "benchmarks" / "cms-synthetic"


def report(accuracy: float, **changes: Any) -> Report:
    data: dict[str, Any] = {
        "examples": 10,
        "scored": 10,
        "unmappable": 0,
        "mapped": 0,
        "missing": 0,
        "errors": 0,
        "metrics": {
            "accuracy@1": Estimate(accuracy, accuracy, accuracy),
            "invalid-code rate": Estimate(0.0, 0.0, 0.0),
            "calibration error": Estimate(math.nan, math.nan, math.nan),
        },
        "selective": {},
        "latency_ms": {"p50": 1.0, "p95": 2.0},
    }
    data.update(changes)
    return Report(**data)


def header(name: str = "coder", **dataset: Any) -> dict[str, Any]:
    fields = {"name": "tiny", "split": "all", "limit": None, "sha256": "abc"}
    return {"system": {"name": name}, "release": "fy2099", "dataset": {**fields, **dataset}}


def expectations(**changes: Any) -> Expectations:
    data: dict[str, Any] = {
        "dataset": "tiny",
        "split": "all",
        "release": "fy2099",
        "sha256": "abc",
        "systems": {
            "coder": {
                "accuracy@1": {"min": 0.9},
                "invalid-code rate": {"max": 0},
                "kept/accuracy@1": {"min": 0.95},
                "errors": {"max": 0},
            }
        },
    }
    data.update(changes)
    return Expectations.model_validate(data)


def test_bounds() -> None:
    assert Bound(min=0.9).holds(0.9)
    assert not Bound(min=0.9).holds(0.89)
    assert Bound(max=0).holds(0.0)
    assert not Bound(max=0).holds(0.001)
    assert Bound(min=1, max=2).holds(1.5)
    assert [str(Bound(min=0.9)), str(Bound(max=0)), str(Bound(min=1, max=2))] == [
        ">= 0.9000",
        "<= 0",
        "1 to 2",
    ]
    with pytest.raises(ValueError, match="min, max or both"):
        Bound()
    with pytest.raises(ValueError, match="above max"):
        Bound(min=2, max=1)


def test_expectations_reject_unknown_numbers() -> None:
    with pytest.raises(ValueError, match="unknown number 'acuracy@1'"):
        expectations(systems={"coder": {"acuracy@1": {"min": 0.9}}})
    with pytest.raises(ValueError, match="coder: no bounds"):
        expectations(systems={"coder": {}})
    with pytest.raises(ValueError, match="at least 1 item"):
        expectations(systems={})
    with pytest.raises(ValueError, match="Extra inputs"):
        expectations(seed=0)
    assert expectations(systems={"coder": {"escalated/scored": {"min": 1}}})


def test_measure() -> None:
    kept = report(0.97)
    full = report(0.92, subsets={"kept": kept})
    assert measure(full, "accuracy@1") == 0.92
    assert measure(full, "kept/accuracy@1") == 0.97
    assert measure(full, "errors") == 0.0
    assert measure(full, "kept/scored") == 10.0
    assert measure(full, "escalated/accuracy@1") is None  # no such subset
    assert measure(full, "calibration error") is None  # undefined (NaN)
    assert measure(full, "code-set F1") is None  # not in the report


def test_check() -> None:
    passing = report(0.92, subsets={"kept": report(0.97)})
    outcomes = check([(header(), passing)], expectations())
    assert [(o.name, o.measured, o.bound, o.ok) for o in outcomes] == [
        ("accuracy@1", "0.9200", ">= 0.9000", True),
        ("invalid-code rate", "0", "<= 0", True),
        ("kept/accuracy@1", "0.9700", ">= 0.9500", True),
        ("errors", "0", "<= 0", True),
    ]
    lines = check_lines(outcomes)
    assert lines[0] == "| System | Number | Measured | Bound | Result |"
    assert lines[2] == "| coder | accuracy@1 | 0.9200 | >= 0.9000 | ok |"

    failing = check([(header(), report(0.85, errors=1))], expectations())
    assert [(o.name, o.measured, o.ok) for o in failing if not o.ok] == [
        ("accuracy@1", "0.8500", False),
        ("kept/accuracy@1", "not measured", False),
        ("errors", "1", False),
    ]
    assert check_lines(failing)[2].endswith("| FAIL |")


def test_check_refuses_other_runs() -> None:
    other = check([(header(split="dev", limit=5, sha256="def"), report(0.99))], expectations())
    assert [(o.name, o.measured, o.bound, o.ok) for o in other] == [
        ("split", "dev", "all", False),
        ("limit", "5", "None", False),
        ("dataset sha256", "def", "abc", False),
    ]
    unlisted = check([(header("bm25"), report(0.99))], expectations())
    assert [(o.system, o.name, o.ok) for o in unlisted] == [("bm25", "system", False)]
    unpinned = check([(header(sha256="def"), report(0.99))], expectations(sha256=None))
    assert not any(o.name == "dataset sha256" for o in unpinned)
    l2 = {**header(), "system": {"name": "coder", "retrieval": "test/model"}}
    keyed = expectations(systems={"coder + L2": {"accuracy@1": {"min": 0.9}}})
    assert [(o.system, o.ok) for o in check([(l2, report(0.99))], keyed)] == [("coder + L2", True)]
    l3 = {**header(), "system": {"name": "coder", "retrieval": "m", "llm": {"model": "x-1"}}}
    assert [o.system for o in check([(l3, report(0.99))], keyed)] == ["coder + L2 + L3 (x-1)"]


def test_committed_sample_expectations() -> None:
    expected = load_expectations(SAMPLE / "fy2027-sample.expected.toml")
    manifest = json.loads((SAMPLE / "fy2027-sample.manifest.json").read_text())
    assert (expected.dataset, expected.split) == ("cms-synthetic-sample", "all")
    assert (expected.release, expected.sha256) == (manifest["release"], manifest["sha256"])
    assert {"coder", "coder + L2", "bm25"} <= set(expected.systems)
    for label in ("coder", "coder + L2"):
        bounds = expected.systems[label]
        assert bounds["invalid-code rate"] == Bound(max=0)
        assert bounds["cost per 1,000 phrases"] == Bound(max=0)


cli_runner = CliRunner()


def test_check_command(tmp_path: Path) -> None:
    rules = tmp_path / "expected.toml"
    rules.write_text(
        'dataset = "tiny"\nsplit = "all"\nrelease = "fy2099"\n\n'
        '[systems.coder]\n"accuracy@1" = { min = 0.9 }\n\n'
        '[systems.bm25]\n"accuracy@1" = { min = 0.3 }\n'
    )
    good, bad = tmp_path / "good.metrics.json", tmp_path / "bad.metrics.json"
    for path, accuracy in ((good, 0.92), (bad, 0.5)):
        data = {"run": header(), "report": report_to_json(report(accuracy))}
        path.write_text(json.dumps(data))

    passed = cli_runner.invoke(cli.app, ["eval", "check", str(good), "-e", str(rules)])
    assert passed.exit_code == 0, passed.output
    assert "| coder | accuracy@1 | 0.9200 | >= 0.9000 | ok |" in passed.output
    assert "not run, not checked: bm25" in passed.output
    assert "all 1 checks passed" in passed.output

    failed = cli_runner.invoke(cli.app, ["eval", "check", str(good), str(bad), "-e", str(rules)])
    assert failed.exit_code == 1
    assert "| FAIL |" in failed.output
    assert "1 of 2 checks failed" in failed.output

    rules.write_text('dataset = "tiny"\n')
    broken = cli_runner.invoke(cli.app, ["eval", "check", str(good), "-e", str(rules)])
    assert broken.exit_code == 1
    assert "error: " in broken.output
    with pytest.raises(ExpectationsError, match=r"expected\.toml"):
        load_expectations(rules)
