import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import text2icd
from text2icd import cli
from text2icd.eval.calibrate import (
    CalibrationOutputError,
    Sample,
    collect,
    ece,
    fit,
    lines,
    refuse_output,
)
from text2icd.eval.datasets import Example, write_dataset, write_manifest
from text2icd.graph.model import Graph
from text2icd.resolve.coder import Coder
from text2icd.resolve.confidence import FEATURES, read_calibration

pytest.importorskip("sklearn")


def sample(n: int = 4000, seed: int = 0) -> Sample:
    """Right more often the more of the phrase the path covers."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, len(FEATURES)))
    coverage = rng.uniform(0, 1, n)
    x[:, FEATURES.index("coverage")] = coverage
    y = (rng.uniform(0, 1, n) < coverage).astype(np.float64)
    return Sample(x, y, [f"G{i % 200}" for i in range(n)], uncoded=40)


def test_fit() -> None:
    fitted = fit(sample(), target=0.9, fitted={"name": "toy"})
    calibration = fitted.calibration
    assert calibration.features == FEATURES
    assert calibration.weights[FEATURES.index("coverage")] > 0
    assert fitted.held_out["calibration error"] < 0.05
    assert 0.8 < fitted.held_out["auroc"] < 0.95
    assert fitted.chosen.precision >= 0.9
    assert calibration.threshold == fitted.chosen.threshold
    assert calibration.fitted["name"] == "toy"
    assert calibration.fitted["examples"] == 4040
    low = calibration.probability({"coverage": 0.1})
    high = calibration.probability({"coverage": 0.9})
    assert low < 0.3 < 0.7 < high
    assert list(fitted.selective) == [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    assert fitted.selective[0.5] > fitted.selective[1.0]
    text = "\n".join(lines(fitted))
    assert "calibration error" in text
    assert "weights: coverage +" in text
    assert "threshold " in text
    unreachable = "\n".join(lines(fit(sample(), target=0.999)))
    assert "no threshold reaches precision 100%; the highest" in unreachable


def test_fit_refuses() -> None:
    one_class = sample()
    one_class = Sample(one_class.x, np.ones(len(one_class.y)), one_class.groups, 0)
    with pytest.raises(ValueError, match="both right and wrong"):
        fit(one_class)
    with pytest.raises(ValueError, match="target precision"):
        fit(sample(), target=1.5)


def test_ece() -> None:
    assert ece(np.array([0.95, 0.95, 0.05, 0.05]), np.array([1.0, 1.0, 0.0, 0.0])) == (
        pytest.approx(0.05)
    )
    assert np.isnan(ece(np.array([]), np.array([])))


def test_refuse_output(tmp_path: Path) -> None:
    shipped = Path(text2icd.__file__).parent / "resources" / "calibration.json"
    refuse_output(shipped, restricted=False)
    refuse_output(tmp_path / "local.json", restricted=True)
    with pytest.raises(CalibrationOutputError, match="local only"):
        refuse_output(shipped, restricted=True)


def _examples(source: str = "test") -> list[Example]:
    rows = [
        ("aspiration pneumonia", "J69.0"),
        ("amyloid heart disease", "E85.4"),
        ("type 2 diabetes mellitus with diabetic chronic kidney disease", "E11.22"),
        ("pneumonia", "J18.9"),
    ]
    return [
        Example.model_validate(
            {
                "id": f"x{i}",
                "phrase": phrase,
                "gold": [{"code": code, "role": "primary"}],
                "release": "fy2099",
                "source": source,
                "split": "dev",
            }
        )
        for i, (phrase, code) in enumerate(rows)
    ]


def test_collect(graph: Graph) -> None:
    found = collect(Coder.from_graph(graph), _examples())
    assert found.x.shape == (3, len(FEATURES))
    assert found.y.tolist() == [1.0, 1.0, 0.0]  # N11.9 is not E11.22
    assert found.groups == ["J69.0", "E85.4", "E11.22"]
    assert found.uncoded == 1  # "pneumonia" gets no code in the fixture
    assert found.examples == 4


def test_calibrate_command(graph_file: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "set.jsonl"
    examples = _examples()
    write_dataset(dataset, examples)
    write_manifest(dataset, examples, name="tiny")
    output = tmp_path / "table.json"
    args = ["eval", "calibrate", "-d", str(dataset), "--graph", str(graph_file)]
    result = CliRunner().invoke(cli.app, [*args, "-o", str(output), "--folds", "2"])
    assert result.exit_code == 0, result.output
    assert "4 scored, 1 without a code" in result.output
    assert "threshold" in result.output
    table = read_calibration(output)
    assert table is not None
    assert table.fitted["name"] == "tiny dev fy2099"
    assert "calibration" not in json.dumps(table.fitted["options"])


def test_calibrate_command_keeps_restricted_tables_local(graph_file: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "mimic.jsonl"
    write_dataset(dataset, _examples(source="mimic-iv-note"))
    shipped = Path(text2icd.__file__).parent / "resources" / "calibration.json"
    args = ["eval", "calibrate", "-d", str(dataset), "--graph", str(graph_file)]
    result = CliRunner().invoke(cli.app, [*args, "-o", str(shipped)])
    assert result.exit_code == 2
    assert "local only" in result.output
