import json
from pathlib import Path
from typing import Any

import pytest

from text2icd.graph.model import Code, Graph
from text2icd.resolve.coder import Coder, describe_code_set
from text2icd.resolve.confidence import (
    FEATURES,
    Calibration,
    CalibrationError,
    interpolate,
    read_calibration,
    write_calibration,
)
from text2icd.resolve.tabular import TIE, _Descent, _displaced


def table(**changes: Any) -> Calibration:
    data: dict[str, Any] = {
        "features": ("coverage", "margin"),
        "weights": (2.0, 1.0),
        "intercept": -1.0,
        "knots": (-1.0, 0.0, 2.0),
        "values": (0.1, 0.5, 0.9),
        "threshold": 0.6,
        "target": 0.98,
    }
    data.update(changes)
    return Calibration.model_validate(data)


def test_interpolate() -> None:
    knots, values = (-1.0, 0.0, 2.0), (0.1, 0.5, 0.9)
    assert interpolate(-5.0, knots, values) == 0.1  # flat outside the knots
    assert interpolate(5.0, knots, values) == 0.9
    assert interpolate(1.0, knots, values) == pytest.approx(0.7)
    assert interpolate(0.0, (0.0, 0.0, 1.0), (0.2, 0.4, 0.6)) == 0.2


def test_probability_and_escalation() -> None:
    calibration = table()
    assert calibration.score({"coverage": 0.5, "margin": 1.0}) == 1.0
    assert calibration.probability({"coverage": 0.5, "margin": 1.0}) == 0.7
    assert calibration.probability({"coverage": 0.5}) == 0.5  # a missing feature counts 0
    assert calibration.probability({}) == 0.0  # no code
    assert calibration.escalate(0.59)
    assert not calibration.escalate(0.6)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"weights": (1.0,)}, "one weight per feature"),
        ({"knots": (1.0, 0.0, 2.0)}, "increase"),
        ({"values": (0.1, 0.5, 1.5)}, "probabilities"),
        ({"knots": (), "values": ()}, "one value per knot"),
    ],
)
def test_bad_tables_are_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        table(**changes)


def test_write_and_read(tmp_path: Path) -> None:
    path = write_calibration(tmp_path / "sub" / "table.json", table())
    assert read_calibration(path) == table()
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({**table().model_dump(), "features": ["coverage", "nope"]}))
    with pytest.raises(CalibrationError, match="unknown features nope"):
        read_calibration(unknown)
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    with pytest.raises(CalibrationError, match=r"broken\.json"):
        read_calibration(broken)


def test_shipped_table() -> None:
    """The shipped table reads this version's features (refit after changing them) and was
    fitted on public data only."""
    shipped = read_calibration()
    assert shipped is not None
    assert shipped.features == FEATURES
    assert 0 < shipped.threshold < 1
    assert str(shipped.fitted["dataset"]).startswith("cms-synthetic")
    held_out = shipped.fitted["held_out"]
    assert isinstance(held_out, dict)
    assert held_out["calibration error"] < 0.05  # the Phase 7 exit criterion


def test_coder_features_and_confidence(graph: Graph) -> None:
    sure = Coder.from_graph(graph, calibration=table())
    code_set = sure.code("aspiration pneumonia")
    assert set(code_set.features) == set(FEATURES)
    assert code_set.features["coverage"] == 1.0
    assert code_set.features["margin"] == 3.0  # no other code: the cap
    assert code_set.features["default_code"] == 0.0
    assert code_set.confidence == 0.9
    assert not code_set.escalate
    assert "confidence: 0.90" in describe_code_set(code_set)

    defaulted = sure.code("amyloid heart disease").features
    assert defaulted["default_code"] == 1.0

    part = sure.code("type 2 diabetes mellitus with diabetic chronic kidney disease")
    assert part.features["coverage"] < 0.5
    assert part.features["undocumented"] == 1.0
    assert part.escalate
    assert "confidence: " in "\n".join(describe_code_set(part))
    assert "(low: escalate)" in "\n".join(describe_code_set(part))

    missing = sure.code("pneumonia")
    assert (missing.features, missing.confidence, missing.escalate) == ({}, 0.0, True)


def test_tabular_tie_is_flagged() -> None:
    def code(key: str, description: str) -> Code:
        return Code(key, "diag", "S72.0", 4, description, True, False, None)

    descent = _Descent()
    tied = [code("S72.01", "Fracture of A"), code("S72.02", "Fracture of B")]
    assert _displaced(tied, descent).key == "S72.01"
    assert descent.flags == [TIE]
    descent = _Descent()
    fracture = [code("S72.031", "Displaced fracture"), code("S72.034", "Nondisplaced fracture")]
    assert _displaced(fracture, descent).key == "S72.031"
    assert descent.flags == []
