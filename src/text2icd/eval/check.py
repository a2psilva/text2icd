"""Regression bounds for evaluation runs (`text2icd eval check`).

An expectations file (TOML) names the dataset, split and release the bounds were measured on, and
for each system, by its label in the report tables, the bounds its numbers must meet:

    dataset = "cms-synthetic-sample"
    split = "all"
    release = "fy2027"
    sha256 = "0e54..."  # optional: the dataset file the bounds were measured on

    [systems.coder]
    "accuracy@1" = { min = 0.952 }
    "invalid-code rate" = { max = 0.0 }
    "kept/accuracy@1" = { min = 0.976 }  # a subset's metric
    "errors" = { max = 0 }               # a count

Point estimates are compared; the bootstrap intervals play no part. A run of another dataset,
split, release or sample size, or of a system the file does not list, fails the check.
"""

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from text2icd.eval.metrics import METRICS, Report
from text2icd.eval.report import system_label

COUNTS = ("examples", "scored", "unmappable", "mapped", "missing", "errors")


class ExpectationsError(ValueError):
    """An expectations file that cannot be read."""


class Bound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check(self) -> "Bound":
        if self.min is None and self.max is None:
            raise ValueError("a bound needs min, max or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min {self.min} is above max {self.max}")
        return self

    def holds(self, value: float) -> bool:
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)

    def __str__(self) -> str:
        if self.min is not None and self.max is not None:
            return f"{_number(self.min)} to {_number(self.max)}"
        if self.min is not None:
            return f">= {_number(self.min)}"
        return f"<= {_number(self.max or 0.0)}"


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    split: str
    release: str
    sha256: str | None = None
    systems: dict[str, dict[str, Bound]] = Field(min_length=1)

    @field_validator("systems")
    @classmethod
    def _names(cls, systems: dict[str, dict[str, Bound]]) -> dict[str, dict[str, Bound]]:
        for label, bounds in systems.items():
            if not bounds:
                raise ValueError(f"{label}: no bounds")
            for name in bounds:
                metric = name.rpartition("/")[2]
                if metric not in METRICS and metric not in COUNTS:
                    known = ", ".join((*METRICS, *COUNTS))
                    raise ValueError(f"{label}: unknown number {name!r} (known: {known})")
        return systems


def load_expectations(path: Path) -> Expectations:
    try:
        return Expectations.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ExpectationsError(f"{path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Outcome:
    system: str
    name: str
    measured: str
    bound: str
    ok: bool


def measure(report: Report, name: str) -> float | None:
    """A metric (`accuracy@1`), a count (`errors`) or a subset's (`kept/accuracy@1`); None when
    the report does not have it."""
    subset, _, number = name.rpartition("/")
    part = report.subsets.get(subset) if subset else report
    if part is None:
        return None
    if number in COUNTS:
        return float(getattr(part, number))
    estimate = part.metrics.get(number)
    return estimate.value if estimate is not None and estimate.defined else None


def check(runs: Sequence[tuple[dict[str, Any], Report]], expected: Expectations) -> list[Outcome]:
    """One outcome per bound of each run's system, or the reasons the run is not comparable."""
    outcomes = []
    for header, report in runs:
        label = system_label(header)
        bounds = expected.systems.get(label)
        if bounds is None:
            outcomes.append(Outcome(label, "system", label, "listed in the file", False))
            continue
        if mismatches := _mismatches(header, expected):
            outcomes += [Outcome(label, *m, False) for m in mismatches]
            continue
        for name, bound in bounds.items():
            value = measure(report, name)
            shown = "not measured" if value is None else _number(value)
            ok = value is not None and bound.holds(value)
            outcomes.append(Outcome(label, name, shown, str(bound), ok))
    return outcomes


def check_lines(outcomes: Sequence[Outcome]) -> list[str]:
    head = ["System", "Number", "Measured", "Bound", "Result"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for o in outcomes:
        cells = [o.system, o.name, o.measured, o.bound, "ok" if o.ok else "FAIL"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _mismatches(header: dict[str, Any], expected: Expectations) -> list[tuple[str, str, str]]:
    """(what, measured on, expected) for each way the run differs from the file's dataset."""
    dataset = header["dataset"]
    pairs = [
        ("dataset", dataset["name"], expected.dataset),
        ("split", dataset["split"], expected.split),
        ("release", header["release"], expected.release),
        ("limit", str(dataset.get("limit")), "None"),
    ]
    if expected.sha256 is not None:
        pairs.append(("dataset sha256", dataset["sha256"], expected.sha256))
    return [(what, str(got), want) for what, got, want in pairs if str(got) != want]


def _number(value: float) -> str:
    return f"{value:,.0f}" if value.is_integer() else f"{value:.4f}"
