"""Evaluation metrics with 95% bootstrap confidence intervals (docs/evaluation.md#metrics).

Each example becomes a row of numbers (correct at 1, true positives, cost, ...) and every metric is
a ratio of column sums, so one bootstrap pass (resampling examples with replacement) gives the
interval of every metric at once.

Gold codes are matched by their status against the release: a `valid` code by exact match, a
`mapped` code when the prediction is one of its replacements. Examples with an `unmappable` gold
code cannot be scored and are counted, not scored.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from text2icd.eval.datasets import Example, GoldCode
from text2icd.eval.predictions import Prediction
from text2icd.ontology.codes import category, is_code

RESAMPLES = 1000
CONFIDENCE = 0.95
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
CALIBRATION_BINS = 10
TOP_K = 5
# Rows of resampled counts held in memory at once (resamples x examples).
_CHUNK_CELLS = 10_000_000

Matrix = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float
    low: float
    high: float

    @property
    def defined(self) -> bool:
        return not math.isnan(self.value)


@dataclass(frozen=True, slots=True)
class Report:
    examples: int
    scored: int  # examples whose gold codes are all valid or mapped
    unmappable: int  # examples left out because a gold code is unmappable
    mapped: int  # scored examples with at least one mapped gold code
    missing: int  # examples the system returned nothing for
    errors: int  # predictions that carry an error
    metrics: dict[str, Estimate]
    selective: dict[float, Estimate]  # coverage -> accuracy@1 on the most confident examples
    latency_ms: dict[str, float]  # p50 and p95
    subsets: dict[str, "Report"] = field(default_factory=dict)  # "valid" and "mapped" examples
    groups: dict[str, "Report"] = field(default_factory=dict)  # for example per chapter


def matches(code: str, gold: GoldCode) -> bool:
    if gold.status == "valid":
        return code == gold.code
    return gold.status == "mapped" and code in gold.mapped_to


def category_matches(code: str, gold: GoldCode) -> bool:
    if not is_code(code) or gold.status == "unmappable":
        return False
    return category(code) in {category(c) for c in (gold.code, *gold.mapped_to)}


# Per-example columns; every metric is a ratio of their sums.
_COLUMNS = (
    "one",
    "scored",
    "acc1",
    "acc5",
    "category",
    "tp",
    "fp",
    "fn",
    "invalid",
    "predicted",
    "llm_used",
    "llm_calls",
    "cost",
)


@dataclass(frozen=True, slots=True)
class _Table:
    """The per-example numbers of a whole dataset, sliced for subsets and groups."""

    matrix: Matrix  # one row per example, `names` columns
    names: tuple[str, ...]
    confidence: Vector  # NaN where the example is not scored or the system gives no confidence
    latency: Vector  # NaN where there is no prediction
    flags: dict[str, npt.NDArray[np.bool_]]  # scored, unmappable, mapped, missing, error


def score(
    examples: Sequence[Example],
    predictions: Mapping[str, Prediction],
    billable: Callable[[str], bool],
    *,
    group_of: Callable[[Example], str] | None = None,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> Report:
    """Score predictions (by example id) against a dataset."""
    table = _table(examples, predictions, billable)
    every = np.arange(len(examples))
    report = _report(table, every, resamples, seed)
    scored, mapped = table.flags["scored"], table.flags["mapped"]
    for name, mask in (("valid", scored & ~mapped), ("mapped", mapped)):
        if mask.any():
            report.subsets[name] = _report(table, every[mask], resamples, seed)
    if group_of is not None:
        groups: dict[str, list[int]] = {}
        for i, example in enumerate(examples):
            groups.setdefault(group_of(example), []).append(i)
        for name, indices in sorted(groups.items()):
            report.groups[name] = _report(table, np.array(indices), resamples, seed)
    return report


def _table(
    examples: Sequence[Example],
    predictions: Mapping[str, Prediction],
    billable: Callable[[str], bool],
) -> _Table:
    n = len(examples)
    matrix = np.zeros((n, len(_COLUMNS)))
    confidence = np.full(n, math.nan)
    latency = np.full(n, math.nan)
    flags = {name: np.zeros(n, dtype=bool) for name in ("unmappable", "mapped", "missing", "error")}
    for i, example in enumerate(examples):
        prediction = predictions.get(example.id)
        matrix[i] = _row(example, prediction, billable)
        flags["unmappable"][i] = not _scored(example)
        flags["mapped"][i] = _scored(example) and _has_mapped(example)
        if prediction is None:
            flags["missing"][i] = True
            continue
        flags["error"][i] = prediction.error is not None
        latency[i] = prediction.latency_ms
        if prediction.confidence is not None:
            confidence[i] = prediction.confidence
    flags["scored"] = ~flags["unmappable"]
    # Once the system reports any confidence, a scored example without one (or without a
    # prediction) counts as 0, so selective accuracy at full coverage equals accuracy@1.
    if np.any(flags["scored"] & ~np.isnan(confidence)):
        confidence = np.where(flags["scored"], np.nan_to_num(confidence, nan=0.0), math.nan)
        bins, names = _calibration_columns(confidence, matrix[:, _COLUMNS.index("acc1")])
        return _Table(np.hstack([matrix, bins]), _COLUMNS + names, confidence, latency, flags)
    return _Table(matrix, _COLUMNS, np.full(n, math.nan), latency, flags)


def _scored(example: Example) -> bool:
    return all(gold.status != "unmappable" for gold in example.gold)


def _has_mapped(example: Example) -> bool:
    return any(gold.status == "mapped" for gold in example.gold)


def _row(
    example: Example, prediction: Prediction | None, billable: Callable[[str], bool]
) -> tuple[float, ...]:
    """The `_COLUMNS` of one example."""
    prediction = prediction or Prediction(id=example.id)
    predicted = list(dict.fromkeys(prediction.codes + prediction.alternates))
    invalid = sum(not billable(code) for code in predicted)
    usage = (float(prediction.llm_calls > 0), float(prediction.llm_calls), prediction.cost_usd)
    if not _scored(example):
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, invalid, len(predicted), *usage)
    primary, gold = prediction.primary, example.primary
    tp = _matched(example.gold, prediction.codes)
    return (
        1.0,
        1.0,
        float(primary is not None and matches(primary, gold)),
        float(any(matches(code, gold) for code in prediction.ranked[:TOP_K])),
        float(primary is not None and category_matches(primary, gold)),
        tp,
        len(set(prediction.codes)) - tp,
        len(example.gold) - tp,
        invalid,
        len(predicted),
        *usage,
    )


def _matched(gold: Sequence[GoldCode], codes: Sequence[str]) -> int:
    """Gold codes matched by a predicted code, each predicted code used once."""
    unused = list(dict.fromkeys(codes))
    found = 0
    for code in gold:
        for candidate in unused:
            if matches(candidate, code):
                unused.remove(candidate)
                found += 1
                break
    return found


def _calibration_columns(confidence: Vector, correct: Vector) -> tuple[Matrix, tuple[str, ...]]:
    """Per bin of equal-width confidence: membership, confidence and correctness columns."""
    known = ~np.isnan(confidence)
    bins = np.minimum(
        (np.nan_to_num(confidence) * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1
    )
    columns, names = [known.astype(np.float64)], ["conf_n"]
    for b in range(CALIBRATION_BINS):
        member = (known & (bins == b)).astype(np.float64)
        columns += [member * np.nan_to_num(confidence), member * correct]
        names += [f"bin{b}_conf", f"bin{b}_acc"]
    return np.column_stack(columns), tuple(names)


def _report(table: _Table, indices: npt.NDArray[np.int_], resamples: int, seed: int) -> Report:
    matrix, names = table.matrix[indices], table.names
    confidence = table.confidence[indices]
    selective_names: list[float] = []
    if not np.isnan(confidence).all():
        masks, selective_names = _selective_columns(confidence, matrix[:, names.index("acc1")])
        matrix = np.hstack([matrix, masks])
        names = names + tuple(
            f"sel{coverage}_{part}" for coverage in selective_names for part in ("n", "acc")
        )
    column = {name: position for position, name in enumerate(names)}
    point = matrix.sum(axis=0)
    samples = _resampled_sums(matrix, resamples, seed)
    metrics = {
        name: _estimate(formula, column, point, samples) for name, formula in _FORMULAS.items()
    }
    if "conf_n" in column and point[column["conf_n"]] > 0:
        metrics["calibration error"] = _estimate(_ece, column, point, samples)
    latency = table.latency[indices]
    latency = latency[~np.isnan(latency)]
    flags = {name: int(values[indices].sum()) for name, values in table.flags.items()}
    return Report(
        examples=len(indices),
        scored=flags["scored"],
        unmappable=flags["unmappable"],
        mapped=flags["mapped"],
        missing=flags["missing"],
        errors=flags["error"],
        metrics=metrics,
        selective={
            coverage: _estimate(_selective(coverage), column, point, samples)
            for coverage in selective_names
        },
        latency_ms=(
            {"p50": float(np.percentile(latency, 50)), "p95": float(np.percentile(latency, 95))}
            if latency.size
            else {}
        ),
    )


def _selective_columns(confidence: Vector, correct: Vector) -> tuple[Matrix, list[float]]:
    """For each coverage, the most confident examples (by rank, on the full sample; the masks are
    then resampled with the rows) and whether each was right."""
    known = np.flatnonzero(~np.isnan(confidence))
    ranked = known[np.argsort(-confidence[known], kind="stable")]
    columns = []
    for coverage in COVERAGES:
        mask = np.zeros(len(confidence))
        mask[ranked[: math.ceil(coverage * len(ranked))]] = 1.0
        columns += [mask, mask * correct]
    return np.column_stack(columns), list(COVERAGES)


def _resampled_sums(matrix: Matrix, resamples: int, seed: int) -> Matrix:
    """Column sums of `resamples` bootstrap samples of the rows."""
    n, k = matrix.shape
    if n == 0 or resamples == 0:
        return np.zeros((0, k))
    rng = np.random.default_rng(seed)
    chunk = max(1, _CHUNK_CELLS // n)
    parts = []
    for start in range(0, resamples, chunk):
        b = min(chunk, resamples - start)
        drawn = rng.integers(0, n, size=(b, n)) + (np.arange(b)[:, None] * n)
        counts = np.bincount(drawn.ravel(), minlength=b * n).reshape(b, n)
        parts.append(counts @ matrix)
    return np.vstack(parts)


Sums = Callable[[str], Vector]  # a column's sum in each sample
Formula = Callable[[Sums], Vector]


def _estimate(formula: Formula, column: dict[str, int], point: Matrix, samples: Matrix) -> Estimate:
    def reader(sums: Matrix) -> Sums:
        zeros = np.zeros(sums.shape[0])
        return lambda name: sums[:, column[name]] if name in column else zeros

    value = float(formula(reader(point[None, :]))[0])
    values = formula(reader(samples))
    values = values[~np.isnan(values)]
    if math.isnan(value) or values.size == 0:
        return Estimate(value, math.nan, math.nan)
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return Estimate(value, float(low), float(high))


def _divide(numerator: Vector, denominator: Vector) -> Vector:
    out = np.full(numerator.shape, math.nan)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _ratio(numerator: str, denominator: str, scale: float = 1.0) -> Formula:
    return lambda s: scale * _divide(s(numerator), s(denominator))


def _f1(s: Sums) -> Vector:
    return _divide(2 * s("tp"), 2 * s("tp") + s("fp") + s("fn"))


def _precision(s: Sums) -> Vector:
    return _divide(s("tp"), s("tp") + s("fp"))


def _recall(s: Sums) -> Vector:
    return _divide(s("tp"), s("tp") + s("fn"))


def _ece(s: Sums) -> Vector:
    """Expected calibration error over equal-width confidence bins."""
    gaps = sum(np.abs(s(f"bin{b}_acc") - s(f"bin{b}_conf")) for b in range(CALIBRATION_BINS))
    return _divide(np.asarray(gaps, dtype=np.float64), s("conf_n"))


def _selective(coverage: float) -> Formula:
    return _ratio(f"sel{coverage}_acc", f"sel{coverage}_n")


_FORMULAS: dict[str, Formula] = {
    "accuracy@1": _ratio("acc1", "scored"),
    "accuracy@5": _ratio("acc5", "scored"),
    "category accuracy": _ratio("category", "scored"),
    "code-set precision": _precision,
    "code-set recall": _recall,
    "code-set F1": _f1,
    "invalid-code rate": _ratio("invalid", "predicted"),
    "LLM call rate": _ratio("llm_used", "one"),
    "LLM calls per phrase": _ratio("llm_calls", "one"),
    "cost per 1,000 phrases": _ratio("cost", "one", 1000.0),
}
