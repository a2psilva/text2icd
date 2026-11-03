"""Fit the coder's confidence calibration (tasks 7.2 and 7.3): `text2icd eval calibrate`.

The coder codes every example of a split; each gives its features (`resolve.confidence`) and
whether its primary code is right. A logistic regression over the features gives a score, and an
isotonic regression maps the score to a probability (scikit-learn, from the `eval` dependency
group; the table it writes needs neither). Folds grouped by gold code, so that the variants of
one code never sit on both sides, give the held-out calibration error and the selective-accuracy
curve. The escalation threshold is read off that curve: the lowest confidence at which the code
sets kept still reach the target precision, which keeps as many as possible.

With a retriever, the retrieval stage is fitted next (task 7.6). The examples the new table
escalates are coded again, and L2's pool for each (`Coder.options`) gives one row per code: its
features (`RETRIEVAL_FEATURES`) and whether it is the gold code. A logistic regression over the
rows scores the codes; the best of each pool is L2's choice, and an isotonic regression over the
choices maps its score to a probability. Held-out folds, grouped by gold code again, give the
accuracy of the choice and its calibration. The stage's threshold is the lowest at which all the
code sets kept, the coder's and L2's, still reach the target precision.

Only aggregates leave this module. The isotonic map is sampled on a fixed grid between two
percentiles of the scores, so the table holds weights, grid points and summary numbers, nothing
of any one example: a table fitted on restricted data (MIMIC-IV) can be kept and used locally,
but is never shipped (`refuse_output`).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

import text2icd
from text2icd.eval.datasets import Example
from text2icd.eval.metrics import CALIBRATION_BINS, COVERAGES, matches
from text2icd.resolve.coder import Coder
from text2icd.resolve.confidence import (
    FEATURES,
    RETRIEVAL_FEATURES,
    Calibration,
    RetrievalStage,
)

FOLDS = 5
TARGET = 0.98  # precision of the code sets kept without escalation
GRID = 101  # points of the isotonic map
_TAILS = (0.5, 99.5)  # percentiles of the scores the grid spans
_THRESHOLDS = np.round(np.arange(0, 101) / 100, 2)

Matrix = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]


class CalibrationUnavailableError(RuntimeError):
    """scikit-learn is not installed (it comes with the `eval` dependency group)."""


class CalibrationOutputError(ValueError):
    """An output path a table fitted on restricted data may not be written to."""


@dataclass(frozen=True, slots=True)
class Sample:
    """The scored examples: the features of those the coder gave a code, whether each primary
    code is right, and the gold code each is grouped by; `uncoded`, those it gave no code."""

    x: Matrix
    y: Vector
    groups: list[str]
    uncoded: int
    rows: tuple[int, ...] = ()  # the example of each row, by position in the examples given
    missing: tuple[int, ...] = ()  # the examples without a code, likewise

    @property
    def examples(self) -> int:
        return len(self.y) + self.uncoded


@dataclass(frozen=True, slots=True)
class Point:
    threshold: float
    coverage: float  # share of the scored examples kept
    precision: float  # accuracy@1 of those kept


@dataclass(frozen=True, slots=True)
class Fitted:
    calibration: Calibration
    held_out: dict[str, float]  # accuracy, calibration error, Brier score, AUROC
    selective: dict[float, float]  # coverage -> held-out accuracy@1 of the most confident
    curve: list[Point]  # held out, by threshold
    chosen: Point


def collect(
    coder: Coder, examples: Sequence[Example], progress: Callable[[int], None] | None = None
) -> Sample:
    """Code every scored example (gold codes valid or mapped) and read its features."""
    rows: list[list[float]] = []
    labels: list[float] = []
    groups: list[str] = []
    positions: list[int] = []
    missing: list[int] = []
    for position, example in enumerate(examples):
        if progress is not None:
            progress(1)
        if any(gold.status == "unmappable" for gold in example.gold):
            continue
        code_set = coder.code(example.phrase)
        if not code_set.features:
            missing.append(position)
            continue
        rows.append([code_set.features[name] for name in FEATURES])
        labels.append(float(matches(code_set.codes[0], example.primary)))
        groups.append(example.primary.code)
        positions.append(position)
    x = np.array(rows, dtype=np.float64).reshape(len(rows), len(FEATURES))
    y = np.array(labels, dtype=np.float64)
    return Sample(x, y, groups, len(missing), tuple(positions), tuple(missing))


def escalated(sample: Sample, calibration: Calibration) -> list[int]:
    """The examples (positions) a table escalates: those below its threshold and those
    without a code."""
    below = [
        position
        for position, row in zip(sample.rows, sample.x, strict=True)
        if calibration.escalate(calibration.probability(dict(zip(FEATURES, row, strict=True))))
    ]
    return sorted([*below, *sample.missing])


def fit(
    sample: Sample,
    *,
    target: float = TARGET,
    folds: int = FOLDS,
    fitted: dict[str, JsonValue] | None = None,
) -> Fitted:
    """Fit on the whole sample; estimate how it does on held-out folds; choose the threshold."""
    if len(set(sample.y.tolist())) < 2:
        raise ValueError("calibration needs both right and wrong primary codes")
    if not 0 < target <= 1:
        raise ValueError("the target precision must be in (0, 1]")
    group_kfold = _sklearn()["GroupKFold"]
    held = np.zeros(len(sample.y))
    splits = group_kfold(n_splits=min(folds, len(set(sample.groups)))).split(
        sample.x, sample.y, sample.groups
    )
    for train, test in splits:
        model = _fit_once(sample.x[train], sample.y[train])
        held[test] = _predict(model, sample.x[test])
    # Uncoded examples count as wrong at confidence 0, as the coder reports them.
    p = np.concatenate([held, np.zeros(sample.uncoded)])
    y = np.concatenate([sample.y, np.zeros(sample.uncoded)])
    curve = _curve(p, y)
    chosen = next((point for point in curve if point.precision >= target), curve[-1])
    model = _fit_once(sample.x, sample.y)
    held_out = {
        "accuracy": float(y.mean()),
        "calibration error": ece(p, y),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc": float(_sklearn()["roc_auc_score"](y, p)),
    }
    summary: dict[str, JsonValue] = {
        **(fitted or {}),
        "examples": sample.examples,
        "uncoded": sample.uncoded,
        "folds": folds,
        "held_out": {name: round(value, 4) for name, value in held_out.items()},
        "kept": round(chosen.coverage, 4),
        "kept_precision": round(chosen.precision, 4),
    }
    calibration = Calibration(
        features=FEATURES,
        weights=tuple(round(float(w), 6) for w in model.weights),
        intercept=round(float(model.intercept), 6),
        knots=tuple(round(float(k), 4) for k in model.knots),
        values=tuple(round(float(v), 4) for v in model.values),
        threshold=chosen.threshold,
        target=target,
        fitted=summary,
    )
    return Fitted(calibration, held_out, _selective(p, y), curve, chosen)


def ece(p: Vector, y: Vector) -> float:
    """Expected calibration error over equal-width bins, as `eval.metrics` reports it."""
    bins = np.minimum((p * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    gaps = sum(abs(float(y[bins == b].sum() - p[bins == b].sum())) for b in range(CALIBRATION_BINS))
    return gaps / len(p) if len(p) else math.nan


def refuse_output(path: Path, restricted: bool) -> None:
    """A table fitted on restricted data is never written inside the package (it would ship)."""
    package = Path(text2icd.__file__).resolve().parent
    if restricted and path.resolve().is_relative_to(package):
        raise CalibrationOutputError(
            "a table fitted on restricted (MIMIC-IV) data is local only: write it under data/ "
            "and point TEXT2ICD_CALIBRATION at it"
        )


def lines(fitted: Fitted) -> list[str]:
    """The aggregate summary printed after a fit."""
    h, c = fitted.held_out, fitted.calibration
    out = [
        f"held out ({c.fitted.get('folds')} folds by gold code): accuracy@1 {h['accuracy']:.1%}, "
        f"calibration error {h['calibration error']:.3f}, Brier {h['brier']:.3f}, "
        f"AUROC {h['auroc']:.3f}",
        "selective accuracy@1: "
        + ", ".join(f"{cov:.0%} {acc:.1%}" for cov, acc in fitted.selective.items()),
        (
            f"threshold {c.threshold:.2f} for precision {c.target:.0%}: keeps "
            if fitted.chosen.precision >= c.target
            else f"no threshold reaches precision {c.target:.0%}; the highest, "
            f"{c.threshold:.2f}, keeps "
        )
        + f"{fitted.chosen.coverage:.1%} at {fitted.chosen.precision:.1%}, escalates the rest",
        "weights: "
        + ", ".join(
            f"{name} {weight:+.2f}"
            for name, weight in sorted(
                zip(c.features, c.weights, strict=True), key=lambda pair: -abs(pair[1])
            )
        ),
    ]
    return out


# -- fitting ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Model:
    weights: Vector
    intercept: float
    knots: Vector
    values: Vector


def _fit_once(x: Matrix, y: Vector) -> _Model:
    if len(set(y.tolist())) < 2:  # a fold of one class: its rate, whatever the features
        rate = np.array([float(y.mean())])
        return _Model(np.zeros(x.shape[1]), 0.0, np.zeros(1), rate)
    sk = _sklearn()
    scaler = sk["StandardScaler"]().fit(x)
    regression = sk["LogisticRegression"](C=1.0, max_iter=2000).fit(scaler.transform(x), y)
    weights = regression.coef_[0] / scaler.scale_
    intercept = float(regression.intercept_[0] - np.sum(weights * scaler.mean_))
    scores = x @ weights + intercept
    isotonic = sk["IsotonicRegression"](out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(scores, y)
    low, high = np.percentile(scores, _TAILS)
    knots = np.linspace(low, high, GRID)
    return _Model(weights, intercept, knots, np.clip(isotonic.predict(knots), 0.0, 1.0))


def _predict(model: _Model, x: Matrix) -> Vector:
    return np.interp(x @ model.weights + model.intercept, model.knots, model.values)


def _curve(p: Vector, y: Vector) -> list[Point]:
    out = []
    for threshold in _THRESHOLDS:
        kept = p >= threshold
        if kept.any():
            out.append(Point(float(threshold), float(kept.mean()), float(y[kept].mean())))
    return out


def _selective(p: Vector, y: Vector) -> dict[float, float]:
    ranked = np.argsort(-p, kind="stable")
    return {
        coverage: float(y[ranked[: math.ceil(coverage * len(p))]].mean()) for coverage in COVERAGES
    }


def _sklearn() -> dict[str, Any]:
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise CalibrationUnavailableError(
            "fitting a calibration needs scikit-learn: uv sync --group eval"
        ) from exc
    return {
        "IsotonicRegression": IsotonicRegression,
        "LogisticRegression": LogisticRegression,
        "roc_auc_score": roc_auc_score,
        "GroupKFold": GroupKFold,
        "StandardScaler": StandardScaler,
    }


# -- the retrieval stage ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Choices:
    """L2's pools for the escalated examples: one row per code, the rows of one example
    together (`case`); per example, its gold code and whether the coder's code is right."""

    x: Matrix
    y: Vector
    case: npt.NDArray[np.int_]
    groups: list[str]
    coder_right: Vector
    coder: Vector  # the row is the coder's code
    index: Vector  # the row is the coder's code or an alternate
    empty: int  # escalated examples with an empty pool (no code anywhere)

    @property
    def cases(self) -> int:
        return len(self.groups)


@dataclass(frozen=True, slots=True)
class FittedChoice:
    stage: RetrievalStage
    held_out: dict[str, float]  # accuracy of the choice and of the coder, calibration, AUROC
    chosen: Point  # all the code sets kept, the coder's and L2's


def collect_choices(
    coder: Coder, examples: Sequence[Example], progress: Callable[[int], None] | None = None
) -> Choices:
    """Code the escalated examples again (the coder holds the new table, without a retrieval
    stage, and the retriever) and read L2's pool for each."""
    rows: list[list[float]] = []
    labels: list[float] = []
    cases: list[int] = []
    groups: list[str] = []
    coder_right: list[float] = []
    coder_rows: list[float] = []
    index_rows: list[float] = []
    empty = 0
    for example in examples:
        if progress is not None:
            progress(1)
        code_set = coder.code(example.phrase)
        pool = coder.options(code_set)
        if not pool:
            empty += 1
            continue
        for option in pool:
            rows.append([option.features[name] for name in RETRIEVAL_FEATURES])
            labels.append(float(matches(option.code, example.primary)))
            cases.append(len(groups))
            coder_rows.append(float(option.coder))
            index_rows.append(float(option.index))
        groups.append(example.primary.code)
        coder_right.append(
            float(bool(code_set.codes) and matches(code_set.codes[0], example.primary))
        )
    x = np.array(rows, dtype=np.float64).reshape(len(rows), len(RETRIEVAL_FEATURES))
    return Choices(
        x,
        np.array(labels, dtype=np.float64),
        np.array(cases, dtype=np.int_),
        groups,
        np.array(coder_right, dtype=np.float64),
        np.array(coder_rows, dtype=np.float64),
        np.array(index_rows, dtype=np.float64),
        empty,
    )


def fit_choice(
    choices: Choices,
    first: Fitted,
    *,
    model: str,
    folds: int = FOLDS,
    fitted: dict[str, JsonValue] | None = None,
) -> FittedChoice:
    """Fit L2's choice on the pools; estimate it on held-out folds; choose the stage's
    threshold given the code sets the coder's table keeps (`first`)."""
    if len(set(choices.y.tolist())) < 2:
        raise ValueError("the retrieval stage needs both right and wrong codes in the pools")
    cases = np.arange(choices.cases)
    held = np.zeros(choices.cases)  # probability of the chosen code
    right = np.zeros(choices.cases)  # the chosen code is right
    took = np.zeros(choices.cases)  # the chosen code is not from the Index walk
    splits = _sklearn()["GroupKFold"](n_splits=min(folds, len(set(choices.groups)))).split(
        cases, cases, choices.groups
    )
    for train, test in splits:
        model_ = _fit_choice_once(*_rows(choices, train))
        x, y, case = _rows(choices, test)
        best = _best(x @ model_.weights + model_.intercept, case)
        where = test[np.unique(case, return_inverse=True)[1][best]]
        held[where] = np.interp(
            x[best] @ model_.weights + model_.intercept, model_.knots, model_.values
        )
        right[where] = y[best]
        took[where] = 1.0 - choices.index[np.flatnonzero(np.isin(choices.case, test))][best]
    # All the scored examples: those the coder keeps (its held-out point), then L2's choices.
    total = first.calibration.fitted.get("examples")
    examples = int(total) if isinstance(total, int) else choices.cases
    kept_first = first.chosen.coverage * examples
    right_first = first.chosen.precision * kept_first
    curve = []
    for threshold in _THRESHOLDS:
        kept = held >= threshold
        count = kept_first + kept.sum()
        if count:
            precision = (right_first + right[kept].sum()) / count
            curve.append(Point(float(threshold), float(count / examples), float(precision)))
    target = first.calibration.target
    chosen = next((point for point in curve if point.precision >= target), curve[-1])
    final = _fit_choice_once(choices.x, choices.y, choices.case)
    # Over all the escalated examples: one with an empty pool keeps no code, at confidence 0.
    none = np.zeros(choices.empty)
    p, y = np.concatenate([held, none]), np.concatenate([right, none])
    auroc = _sklearn()["roc_auc_score"](y, p) if len(set(y.tolist())) == 2 else math.nan
    held_out = {
        "accuracy": float(y.mean()),
        "coder accuracy": float(np.concatenate([choices.coder_right, none]).mean()),
        "retrieved": float(np.concatenate([took, none]).mean()),
        "calibration error": ece(p, y),
        "auroc": float(auroc),
        "kept": float((p >= chosen.threshold).mean()),
    }
    summary: dict[str, JsonValue] = {
        **(fitted or {}),
        "examples": choices.cases + choices.empty,
        "empty": choices.empty,
        "options": len(choices.y),
        "folds": folds,
        "held_out": {
            name: round(value, 4) for name, value in held_out.items() if not math.isnan(value)
        },
        "kept": round(chosen.coverage, 4),
        "kept_precision": round(chosen.precision, 4),
    }
    stage = RetrievalStage(
        features=RETRIEVAL_FEATURES,
        weights=tuple(round(float(w), 6) for w in final.weights),
        intercept=round(float(final.intercept), 6),
        knots=tuple(round(float(k), 4) for k in final.knots),
        values=tuple(round(float(v), 4) for v in final.values),
        threshold=chosen.threshold,
        model=model,
        fitted=summary,
    )
    return FittedChoice(stage, held_out, chosen)


def choice_lines(fitted: FittedChoice) -> list[str]:
    """The aggregate summary of the retrieval stage."""
    h, stage = fitted.held_out, fitted.stage
    examples = stage.fitted.get("examples")
    out = [
        f"retrieval stage on {examples:,} escalated examples ({stage.fitted.get('options'):,} "
        f"codes): held-out accuracy@1 {h['accuracy']:.1%} (coder {h['coder accuracy']:.1%}), "
        f"retrieved code taken {h['retrieved']:.1%}, calibration error "
        f"{h['calibration error']:.3f}"
        + (f", AUROC {h['auroc']:.3f}" if "auroc" in h and not math.isnan(h["auroc"]) else ""),
        f"threshold {stage.threshold:.2f}: L2 keeps {h['kept']:.1%} of them; all kept "
        f"{fitted.chosen.coverage:.1%} at {fitted.chosen.precision:.1%}",
        "weights: "
        + ", ".join(
            f"{name} {weight:+.2f}"
            for name, weight in sorted(
                zip(stage.features, stage.weights, strict=True), key=lambda pair: -abs(pair[1])
            )
        ),
    ]
    return out


def _rows(
    choices: Choices, cases: npt.NDArray[np.int_]
) -> tuple[Matrix, Vector, npt.NDArray[np.int_]]:
    mask = np.isin(choices.case, cases)
    return choices.x[mask], choices.y[mask], choices.case[mask]


def _best(scores: Vector, case: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """The row of the best score of each case (rows of a case together; the first of equals)."""
    starts = np.flatnonzero(np.r_[True, case[1:] != case[:-1]])
    ends = np.r_[starts[1:], len(case)]
    return np.array(
        [
            start + int(np.argmax(scores[start:end]))
            for start, end in zip(starts, ends, strict=True)
        ],
        dtype=np.int_,
    )


def _fit_choice_once(x: Matrix, y: Vector, case: npt.NDArray[np.int_]) -> _Model:
    if len(set(y.tolist())) < 2:
        rate = np.array([float(y.mean())])
        return _Model(np.zeros(x.shape[1]), 0.0, np.zeros(1), rate)
    sk = _sklearn()
    scaler = sk["StandardScaler"]().fit(x)
    regression = sk["LogisticRegression"](C=1.0, max_iter=3000).fit(scaler.transform(x), y)
    weights = regression.coef_[0] / scaler.scale_
    intercept = float(regression.intercept_[0] - np.sum(weights * scaler.mean_))
    scores = x @ weights + intercept
    best = _best(scores, case)
    isotonic = sk["IsotonicRegression"](out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(scores[best], y[best])
    low, high = np.percentile(scores[best], _TAILS)
    knots = np.linspace(low, high, GRID)
    return _Model(weights, intercept, knots, np.clip(isotonic.predict(knots), 0.0, 1.0))
