"""Fit the coder's confidence calibration (tasks 7.2 and 7.3): `text2icd eval calibrate`.

The coder codes every example of a split; each gives its features (`resolve.confidence`) and
whether its primary code is right. A logistic regression over the features gives a score, and an
isotonic regression maps the score to a probability (scikit-learn, from the `eval` dependency
group; the table it writes needs neither). Folds grouped by gold code, so that the variants of
one code never sit on both sides, give the held-out calibration error and the selective-accuracy
curve. The escalation threshold is read off that curve: the lowest confidence at which the code
sets kept still reach the target precision, which keeps as many as possible.

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
from text2icd.resolve.confidence import FEATURES, Calibration

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
    uncoded = 0
    for example in examples:
        if progress is not None:
            progress(1)
        if any(gold.status == "unmappable" for gold in example.gold):
            continue
        code_set = coder.code(example.phrase)
        if not code_set.features:
            uncoded += 1
            continue
        rows.append([code_set.features[name] for name in FEATURES])
        labels.append(float(matches(code_set.codes[0], example.primary)))
        groups.append(example.primary.code)
    x = np.array(rows, dtype=np.float64).reshape(len(rows), len(FEATURES))
    return Sample(x, np.array(labels, dtype=np.float64), groups, uncoded)


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
