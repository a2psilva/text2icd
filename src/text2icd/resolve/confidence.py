"""How sure the coder is of a code set (Phase 7, tasks 7.1 to 7.3).

`features` reads numbers off a code set and the Index walk behind its primary code: how much of
the phrase the path used, the score margin to the next code, whether a default or unspecified
code was taken, ties among Tabular children, typo corrections, cross-references followed, and
what the rules and notes changed.

A `Calibration` turns the features into the probability that the primary code is right: a
logistic model gives a score, and an isotonic map fitted on a dev split turns the score into a
probability (task 7.2). Both are a small JSON table, `resources/calibration.json`, written by
`text2icd eval calibrate`; the runtime reads it with the standard library alone. The table also
holds the escalation threshold chosen for a target precision (task 7.3): a code set whose
confidence is below it is marked `escalate`, for local retrieval or the LLM (Phases 7 to 9).
"""

import json
from bisect import bisect_right
from collections.abc import Collection, Mapping, Sequence
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from text2icd.resolve.profiles import MISSING
from text2icd.resolve.tabular import ASSUMED, REVIEW, TIE, UNSPECIFIED
from text2icd.resolve.walk import DEFAULT_CODE, DiagnosisWalk
from text2icd.text.model import ParsedPhrase

if TYPE_CHECKING:
    from text2icd.resolve.coder import CodedDiagnosis

CALIBRATION: Final = "calibration.json"
MARGIN_CAP = 3.0  # score margins beyond this many words tell nothing more
_STATUS = frozenset({"history", "family_history", "status_post", "resolved"})

# Every feature is a number in [0, 1] or a small count; the order is the table's.
FEATURES: Final = (
    "coverage",  # share of the primary diagnosis's words its path uses
    "phrase_coverage",  # share of the whole phrase's words the coded paths use
    "margin",  # score of the chosen path minus the best path to another code, capped
    "fallback",  # an earlier candidate did not complete to a billable code
    "undocumented",  # undocumented slots on the path (penalty / MISSING)
    "default_code",  # the Index default (I.A.18)
    "unspecified",  # the Tabular List took an unspecified child
    "seventh_assumed",  # no episode of care stated
    "tabular_tie",  # children tied and the first was taken
    "no_detail",  # no child matched and none is unspecified
    "corrections",  # typo corrections in the diagnosis
    "jumps",  # see and see also cross-references followed
    "undocumented_link",  # a "due to" link the phrase does not state
    "table",  # coded from the Neoplasm Table or the Table of Drugs
    "candidates",  # Index candidates of the diagnosis, as a share of the most kept
    "alternates",  # other completed codes of the diagnosis
    "words",  # content words of the diagnosis
    "diagnoses",  # diagnoses in the phrase beyond the first
    "uncoded",  # a diagnosis of the phrase got no code
    "rule_primary",  # a rule or note put another code first
    "unresolved",  # a mandatory instruction found no code
    "excludes1",  # an Excludes1 conflict left in the set
    "ambiguous_abbreviation",
    "unknown_abbreviation",
    "uncertain",  # an uncertainty cue
    "status",  # a history, family history or status-post cue
)
_CANDIDATES = 10  # the walk's TOP


def features(
    parsed: ParsedPhrase,
    walks: Sequence[DiagnosisWalk],
    coded: Sequence[tuple[DiagnosisWalk, "CodedDiagnosis"]],
    primary: str | None,
    flags: Collection[str],
) -> dict[str, float]:
    """The features of a code set: `walks`, every diagnosis walked; `coded`, the diagnoses that
    got a code with their walks; `primary`, the first code of the set; `flags`, the set's."""
    if not coded or primary is None:
        return {}
    walk, chosen = next(
        ((w, c) for w, c in coded if primary in (c.code, c.manifestation)), coded[0]
    )
    candidate = chosen.candidate
    words = set(walk.words)
    phrase_words = {word for w in walks for word in w.words}
    phrase_used = {word for _, c in coded for word in c.candidate.used} & phrase_words
    others = [c.score for c in walk.candidates if c.code != candidate.code]
    margin = candidate.score - max(others) if others else MARGIN_CAP
    rank = next((i for i, c in enumerate(walk.candidates) if c == candidate), 0)
    found = set(chosen.flags)
    values = {
        "coverage": len(set(candidate.used) & words) / len(words) if words else 1.0,
        "phrase_coverage": len(phrase_used) / len(phrase_words) if phrase_words else 1.0,
        "margin": max(-MARGIN_CAP, min(MARGIN_CAP, margin)),
        "fallback": float(rank > 0),
        "undocumented": candidate.penalty / MISSING,
        "default_code": float(DEFAULT_CODE in found),
        "unspecified": float(UNSPECIFIED in found),
        "seventh_assumed": float(ASSUMED in found),
        "tabular_tie": float(TIE in found),
        "no_detail": float(REVIEW in found),
        "corrections": float(sum(step.action == "spelling" for step in walk.trace)),
        "jumps": float(sum(step.action in ("see", "see_also") for step in candidate.trace)),
        "undocumented_link": float("undocumented_link" in candidate.flags),
        "table": float(candidate.book != "index"),
        "candidates": min(len(walk.candidates), _CANDIDATES) / _CANDIDATES,
        "alternates": float(len(chosen.alternates)),
        "words": float(len(words)),
        "diagnoses": float(max(len(walks) - 1, 0)),
        "uncoded": float(REVIEW in flags),
        "rule_primary": float(primary != chosen.code),
        "unresolved": float("required_unresolved" in flags),
        "excludes1": float("excludes1_conflict" in flags),
        "ambiguous_abbreviation": float("ambiguous_abbreviation" in parsed.flags),
        "unknown_abbreviation": float("unknown_abbreviation" in parsed.flags),
        "uncertain": float("uncertain" in parsed.flags),
        "status": float(bool(_STATUS & set(parsed.flags))),
    }
    return {name: round(values[name], 6) for name in FEATURES}


class CalibrationError(ValueError):
    """A calibration table that cannot be read or does not fit this version's features."""


class Calibration(BaseModel):
    """A logistic score over the features, then an isotonic map from score to probability, and
    the confidence below which a code set is escalated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    features: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    knots: tuple[float, ...]  # isotonic map: scores, increasing
    values: tuple[float, ...]  # and the probability at each
    threshold: float  # escalate below this confidence
    target: float  # the precision of the code sets kept that the threshold was chosen for
    fitted: dict[str, JsonValue] = {}  # dataset, split, release, examples, held-out results

    @model_validator(mode="after")
    def _consistent(self) -> "Calibration":
        if len(self.weights) != len(self.features):
            raise ValueError("one weight per feature")
        if not self.knots or len(self.knots) != len(self.values):
            raise ValueError("the isotonic map needs one value per knot")
        if any(b < a for a, b in zip(self.knots, self.knots[1:], strict=False)):
            raise ValueError("isotonic knots must increase")
        if any(not 0 <= v <= 1 for v in self.values):
            raise ValueError("isotonic values must be probabilities")
        return self

    def score(self, values: Mapping[str, float]) -> float:
        return self.intercept + sum(
            w * values.get(name, 0.0) for name, w in zip(self.features, self.weights, strict=True)
        )

    def probability(self, values: Mapping[str, float]) -> float:
        """The probability that the primary code is right; 0 for a set with no code."""
        if not values:
            return 0.0
        return round(interpolate(self.score(values), self.knots, self.values), 4)

    def escalate(self, confidence: float) -> bool:
        return confidence < self.threshold


def interpolate(x: float, knots: Sequence[float], values: Sequence[float]) -> float:
    """Piecewise-linear between knots, flat outside them (as scikit-learn's isotonic predict)."""
    if x <= knots[0]:
        return values[0]
    if x >= knots[-1]:
        return values[-1]
    i = bisect_right(knots, x)
    x0, x1, y0, y1 = knots[i - 1], knots[i], values[i - 1], values[i]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def read_calibration(path: Path | None = None) -> Calibration | None:
    """The table at `path`, or the one shipped with the package (None if there is none)."""
    if path is not None:
        return _parse(path.read_text(encoding="utf-8"), str(path))
    return _shipped()


@cache
def _shipped() -> Calibration | None:
    resource = resources.files("text2icd.resources").joinpath(CALIBRATION)
    if not resource.is_file():
        return None
    return _parse(resource.read_text(encoding="utf-8"), CALIBRATION)


def _parse(text: str, where: str) -> Calibration:
    try:
        calibration = Calibration.model_validate(json.loads(text))
    except ValueError as exc:
        raise CalibrationError(f"{where}: {exc}") from exc
    unknown = set(calibration.features) - set(FEATURES)
    if unknown:
        raise CalibrationError(f"{where}: unknown features {', '.join(sorted(unknown))}")
    return calibration


def write_calibration(path: Path, calibration: Calibration) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.model_dump(), indent=1) + "\n", encoding="utf-8")
    return path
