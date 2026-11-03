import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import text2icd
from text2icd import cli
from text2icd.eval.calibrate import (
    CalibrationOutputError,
    Choices,
    Sample,
    choice_lines,
    collect,
    collect_choices,
    ece,
    escalated,
    fit,
    fit_choice,
    lines,
    refuse_output,
)
from text2icd.eval.datasets import Example, write_dataset, write_manifest
from text2icd.graph.model import Graph
from text2icd.resolve.coder import Coder
from text2icd.resolve.confidence import (
    FEATURES,
    RETRIEVAL_FEATURES,
    Calibration,
    read_calibration,
)
from text2icd.resolve.retrieve import Hits

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
    assert (found.rows, found.missing) == ((0, 1, 2), (3,))


def _table(threshold: float = 0.6) -> Calibration:
    return Calibration.model_validate(
        {
            "features": ("coverage",),
            "weights": (2.0,),
            "intercept": -1.0,
            "knots": (-1.0, 1.0),
            "values": (0.1, 0.9),
            "threshold": threshold,
            "target": 0.9,
        }
    )


def test_escalated() -> None:
    x = np.zeros((3, len(FEATURES)))
    x[:, FEATURES.index("coverage")] = [1.0, 0.2, 0.9]
    found = Sample(x, np.ones(3), ["A", "B", "C"], 1, rows=(0, 2, 5), missing=(4,))
    assert escalated(found, _table()) == [2, 4]  # coverage 0.2 is below; 4 has no code


class Fixed:
    name = "test/fixed"

    def __init__(self, hits: Hits) -> None:
        self.hits = hits

    def search(self, queries: list[str], k: int) -> list[Hits]:
        return [list(self.hits) for _ in queries]


def test_collect_choices(graph: Graph) -> None:
    coder = Coder.from_graph(graph, calibration=_table(), retriever=Fixed([("E11.22", 0.9)]))
    examples = [e for e in _examples() if e.id in ("x2", "x3")]  # escalated: diabetes, pneumonia
    found = collect_choices(coder, examples)
    assert found.cases == 2
    assert found.empty == 0
    assert found.x.shape == (len(found.y), len(RETRIEVAL_FEATURES))
    assert found.case.tolist() == [0, 0, 0, 1]  # N11.9, its alternate E85.4, E11.22; E11.22
    assert found.y.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert found.coder.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert found.index.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert found.coder_right.tolist() == [0.0, 0.0]
    assert found.groups == ["E11.22", "J18.9"]


def choices(n: int = 1500, seed: int = 0) -> Choices:
    """Pools of the coder's code and two retrieved ones: the first retrieved is right as often
    as its similarity says, else the coder's."""
    rng = np.random.default_rng(seed)
    rows, labels, cases, coder = [], [], [], []
    right = []
    for case in range(n):
        similarity = rng.uniform(0, 1)
        retrieved = rng.uniform(0, 1) < similarity
        for kind in range(3):
            row = np.zeros(len(RETRIEVAL_FEATURES))
            row[RETRIEVAL_FEATURES.index("coder")] = float(kind == 0)
            row[RETRIEVAL_FEATURES.index("retrieval_first")] = float(kind == 1)
            row[RETRIEVAL_FEATURES.index("top_similarity")] = similarity
            row[RETRIEVAL_FEATURES.index("similarity")] = similarity if kind == 1 else 0.1
            rows.append(row)
            labels.append(float((kind == 1 and retrieved) or (kind == 0 and not retrieved)))
            cases.append(case)
            coder.append(float(kind == 0))
        right.append(float(not retrieved))
    groups = [f"G{case % 50}" for case in range(n)]
    coder_rows = np.array(coder)
    return Choices(
        np.array(rows),
        np.array(labels),
        np.array(cases),
        groups,
        np.array(right),
        coder_rows,
        coder_rows,
        empty=3,
    )


def test_fit_choice() -> None:
    first = fit(sample(), target=0.9, fitted={"name": "toy"})
    found = choices()
    staged = fit_choice(found, first, model="test/model", fitted={"dataset": "toy"})
    stage = staged.stage
    assert (stage.features, stage.model, stage.fitted["dataset"]) == (
        RETRIEVAL_FEATURES,
        "test/model",
        "toy",
    )
    held = staged.held_out
    assert held["coder accuracy"] == pytest.approx(0.5, abs=0.05)
    assert held["accuracy"] > held["coder accuracy"] + 0.15  # the better of the two, mostly
    assert 0.3 < held["retrieved"] < 0.7
    assert held["calibration error"] < 0.05
    assert stage.fitted["examples"] == 1503
    assert stage.fitted["empty"] == 3
    high = {"retrieval_first": 1.0, "similarity": 0.9, "top_similarity": 0.9}
    low = {"retrieval_first": 1.0, "similarity": 0.1, "top_similarity": 0.1}
    assert stage.score(high) > stage.score(low)
    assert staged.chosen.threshold == stage.threshold
    text = "\n".join(choice_lines(staged))
    assert "retrieval stage on 1,503 escalated examples (4,500 codes)" in text
    assert f"threshold {stage.threshold:.2f}: L2 keeps " in text
    one_class = Choices(
        found.x,
        np.zeros(len(found.y)),
        found.case,
        found.groups,
        found.coder_right,
        found.coder,
        found.index,
        0,
    )
    with pytest.raises(ValueError, match="both right and wrong"):
        fit_choice(one_class, first, model="m")


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


def test_calibrate_command_with_retrieval(
    graph_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = [("J69.0", 0.9), ("E11.22", 0.8), ("E85.4", 0.5)]
    monkeypatch.setattr(cli, "_retriever", lambda graph: Fixed(hits))
    dataset = tmp_path / "set.jsonl"
    examples = [
        e.model_copy(update={"id": f"{e.id}-{copy}"}) for e in _examples() for copy in range(3)
    ]
    write_dataset(dataset, examples)
    write_manifest(dataset, examples, name="tiny")
    output = tmp_path / "table.json"
    args = ["eval", "calibrate", "-d", str(dataset), "--graph", str(graph_file), "--retrieval"]
    result = CliRunner().invoke(cli.app, [*args, "-o", str(output), "--folds", "2"])
    assert result.exit_code == 0, result.output
    assert "retrieval stage on " in result.output
    table = read_calibration(output)
    assert table is not None
    assert table.retrieval is not None
    assert table.retrieval.model == "test/fixed"
    assert table.retrieval.fitted["dataset"] == "set.jsonl"
    plain = CliRunner().invoke(cli.app, [*args[:-1], "-o", str(output), "--folds", "2"])
    assert plain.exit_code == 0, plain.output  # no vectors file next to the graph: no stage
    assert "retrieval stage" not in plain.output
    again = read_calibration(output)
    assert again is not None
    assert again.retrieval is None


def test_calibrate_command_keeps_restricted_tables_local(graph_file: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "mimic.jsonl"
    write_dataset(dataset, _examples(source="mimic-iv-note"))
    shipped = Path(text2icd.__file__).parent / "resources" / "calibration.json"
    args = ["eval", "calibrate", "-d", str(dataset), "--graph", str(graph_file)]
    result = CliRunner().invoke(cli.app, [*args, "-o", str(shipped)])
    assert result.exit_code == 2
    assert "local only" in result.output
