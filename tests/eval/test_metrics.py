import math
from typing import Any

import numpy as np
import pytest

from text2icd.eval import metrics
from text2icd.eval.datasets import Example, GoldCode
from text2icd.eval.metrics import Estimate, Report, category_matches, matches, score
from text2icd.eval.predictions import Prediction

BILLABLE = {"E11.22", "N18.30", "N18.31", "J69.0", "J69.8", "A00.9"}.__contains__


def example(i: int, *gold: dict[str, Any]) -> Example:
    return Example.model_validate(
        {
            "id": str(i),
            "phrase": f"phrase {i}",
            "gold": list(gold),
            "release": "fy2027",
            "source": "test",
            "split": "dev",
        }
    )


EXAMPLES = [
    example(1, {"code": "E11.22", "role": "primary"}, {"code": "N18.30", "role": "companion"}),
    example(2, {"code": "J69.0", "role": "primary"}),
    example(
        3,
        {"code": "N18.3", "role": "primary", "status": "mapped", "mapped_to": ["N18.30", "N18.31"]},
    ),
    example(4, {"code": "V29.88XA", "role": "primary", "status": "unmappable"}),
    example(5, {"code": "A00.9", "role": "primary"}),
    example(6, {"code": "A00.9", "role": "primary"}),
]
PREDICTIONS = {
    p.id: p
    for p in [
        Prediction(id="1", codes=("E1122", "N18.30"), confidence=0.9, latency_ms=10),
        Prediction(
            id="2",
            codes=("J69.8",),
            alternates=("J69.8", "J69.0"),
            confidence=0.6,
            llm_calls=2,
            cost_usd=0.002,
            latency_ms=20,
        ),
        Prediction(id="3", codes=("N18.31",), confidence=0.8, llm_calls=1, cost_usd=0.001),
        Prediction(id="4", codes=("V29.88XA",), latency_ms=40),
        Prediction(id="6", codes=("zzz",), confidence=0.1, error="parse error", latency_ms=30),
    ]
}


@pytest.fixture(scope="module")
def report() -> Report:
    return score(EXAMPLES, PREDICTIONS, BILLABLE, group_of=lambda e: e.primary.code[0])


def value(report: Report, name: str) -> float:
    return report.metrics[name].value


def test_counts(report: Report) -> None:
    assert (report.examples, report.scored, report.unmappable, report.mapped) == (6, 5, 1, 1)
    assert (report.missing, report.errors) == (1, 1)


def test_accuracy_and_code_set(report: Report) -> None:
    assert value(report, "accuracy@1") == pytest.approx(2 / 5)
    assert value(report, "accuracy@5") == pytest.approx(3 / 5)
    assert value(report, "category accuracy") == pytest.approx(3 / 5)
    # True positives 2+0+1, false positives 0+1+0+0+1, false negatives 0+1+0+1+1.
    assert value(report, "code-set precision") == pytest.approx(3 / 5)
    assert value(report, "code-set recall") == pytest.approx(3 / 6)
    assert value(report, "code-set F1") == pytest.approx(6 / 11)


def test_invalid_codes_llm_and_cost(report: Report) -> None:
    # Seven distinct predicted codes over all six examples; V29.88XA and ZZZ are not billable.
    assert value(report, "invalid-code rate") == pytest.approx(2 / 7)
    assert value(report, "LLM call rate") == pytest.approx(2 / 6)
    assert value(report, "LLM calls per phrase") == pytest.approx(3 / 6)
    assert value(report, "cost per 1,000 phrases") == pytest.approx(0.5)
    assert report.latency_ms == {
        "p50": 20.0,
        "p95": pytest.approx(38.0),
        "p99": pytest.approx(39.6),
        "max": 40.0,
    }


def test_calibration_and_selective_accuracy(report: Report) -> None:
    # Bins: 0.9 right, 0.8 right, 0.6 wrong, 0.1 wrong, and example 5 (no prediction) at 0.
    assert value(report, "calibration error") == pytest.approx((0.1 + 0.2 + 0.6 + 0.1) / 5)
    selective = {coverage: e.value for coverage, e in report.selective.items()}
    assert selective[1.0] == value(report, "accuracy@1")
    assert selective[0.8] == pytest.approx(2 / 4)
    assert selective[0.5] == pytest.approx(2 / 3)
    assert selective[0.6] == pytest.approx(2 / 3)


def test_subsets_and_groups(report: Report) -> None:
    assert report.subsets["valid"].examples == 4
    assert value(report.subsets["valid"], "accuracy@1") == pytest.approx(1 / 4)
    assert value(report.subsets["mapped"], "accuracy@1") == 1.0
    assert list(report.groups) == ["A", "E", "J", "N", "V"]
    assert report.groups["V"].scored == 0
    assert math.isnan(value(report.groups["V"], "accuracy@1"))
    assert not report.groups["V"].metrics["accuracy@1"].defined


def test_kept_and_escalated_subsets(report: Report) -> None:
    assert "kept" not in report.subsets  # the predictions do not say
    said = dict(PREDICTIONS)
    said["1"] = said["1"].model_copy(update={"escalated": ()})
    said["2"] = said["2"].model_copy(update={"escalated": ("index",)})
    said["3"] = said["3"].model_copy(update={"escalated": ("index", "retrieval")})
    split = score(EXAMPLES, said, BILLABLE, resamples=20)
    assert split.subsets["kept"].scored == 1
    assert value(split.subsets["kept"], "accuracy@1") == 1.0
    assert split.subsets["escalated"].scored == 2  # escalated by the coder, whatever L2 did
    assert value(split.subsets["escalated"], "accuracy@1") == 0.5


def test_intervals_hold_the_point_estimate(report: Report) -> None:
    for name, estimate in report.metrics.items():
        assert estimate.low <= estimate.value <= estimate.high, name
    again = score(EXAMPLES, PREDICTIONS, BILLABLE)
    assert again.metrics == report.metrics


def test_no_confidence_means_no_selective_accuracy() -> None:
    plain = {k: p.model_copy(update={"confidence": None}) for k, p in PREDICTIONS.items()}
    report = score(EXAMPLES, plain, BILLABLE, resamples=50)
    assert report.selective == {}
    assert "calibration error" not in report.metrics


def test_empty_dataset() -> None:
    report = score([], {}, BILLABLE)
    assert report.examples == 0
    assert all(not e.defined for e in report.metrics.values())
    assert report.latency_ms == {}


def test_chunked_resampling_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = np.arange(12, dtype=np.float64).reshape(4, 3)
    whole = metrics._resampled_sums(matrix, 7, seed=3)
    assert whole.shape == (7, 3)
    assert np.allclose(whole.sum(axis=1) % 1, 0)
    monkeypatch.setattr(metrics, "_CHUNK_CELLS", 4)
    assert metrics._resampled_sums(matrix, 7, seed=3).shape == (7, 3)


def test_interval_narrows_with_more_examples() -> None:
    many = [example(i, {"code": "A00.9", "role": "primary"}) for i in range(400)]
    right = {str(i): Prediction(id=str(i), codes=("A00.9",)) for i in range(0, 400, 2)}
    estimate = score(many, right, BILLABLE).metrics["accuracy@1"]
    assert estimate.value == 0.5
    assert 0.44 < estimate.low < 0.5 < estimate.high < 0.56


def test_matching_rules() -> None:
    valid = GoldCode(code="E11.22", role="primary")
    mapped = GoldCode(code="N18.3", role="primary", status="mapped", mapped_to=("N18.30",))
    lost = GoldCode(code="V29.88XA", role="primary", status="unmappable")
    assert matches("E11.22", valid)
    assert not matches("N18.3", mapped)
    assert matches("N18.30", mapped)
    assert not matches("V29.88XA", lost)
    assert category_matches("E11.9", valid)
    assert category_matches("N18.9", mapped)
    assert not category_matches("V29.88XA", lost)
    assert not category_matches("ZZZ", valid)
    assert Estimate(1.0, 0.5, 1.0).defined
