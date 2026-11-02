"""Status, negation and uncertainty: the words that say whether and how a condition is coded.

`cues(text)` finds them in a normalised phrase, before the condition they qualify
(`history of`, `possible`, `no`) or after it (`ruled out`, `resolved`, `?`). `judge` turns the
cues that apply to one condition into its status and certainty, and decides whether the condition
itself is reported in a setting:

- **status**: `history` (a personal history Z code, I.C.21.c.4), `family_history` (I.C.21.c.4),
  `status_post` (a status Z code after a procedure, I.C.21.c.3, or the history of an event), and
  `resolved`: no longer present, so a personal history rather than a current condition.
  "Resolved" is not "in remission" (I.C.4.a.1(b)); remission is a modifier.
- **certainty**: `negated` (absent or ruled out: not coded) and `uncertain` (the terms of
  Guideline II.H: probable, suspected, likely, questionable, possible, still to be ruled out,
  compatible with, consistent with). In the `inpatient` setting an uncertain diagnosis is coded
  as if established (II.H, III.C); in the `outpatient` setting it is not, and the symptoms are
  coded instead (IV.H). Negation wins over uncertainty (`possible PE, ruled out`).

Which conditions a cue covers is decided by the splitter: a cue before conditions covers those
after it in its list item, and a cue after them covers those before it.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from text2icd.text.model import Certainty, Cue, CueKind, Setting, Status
from text2icd.trace import TraceStep, text_step

_Before = tuple[CueKind, str]


def _words(pattern: str) -> str:
    return rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])"


_BEFORE: tuple[_Before, ...] = (
    ("family_history", r"(?:positive |significant )?family history(?: of)?|fhx(?: of)?"),
    (
        "history",
        r"(?:(?:personal|past|prior|previous|remote|distant|known) )?history of|"
        r"(?:past medical|past surgical) history of|prior|previous|remote|former",
    ),
    ("status_post", r"status post"),
    ("resolved", r"resolved"),
    (
        "negated",
        r"no (?:evidence|signs?|sign|history|findings|suspicion) (?:of|for)|no|not|denies|denied|"
        r"unlikely|"
        r"negative for|free of|absence of|never had|ruled out(?: for)?",
    ),
    (
        "uncertain",
        r"probable|probably|suspected|suspect|suspicion (?:of|for)|likely|most likely|"
        r"questionable|question of|possible|possibly|presumed|presumptive|presumably|"
        r"(?:still )?to be ruled out|rule out|cannot rule out|can't rule out|cannot exclude|"
        r"compatible with|consistent with|concern for|concerning for|suspicious (?:of|for)|"
        r"suggestive of|thought to be|felt to be|believed to be|query|working diagnosis(?: of)?|"
        r"differential (?:diagnosis )?(?:includes|of)|presumed to be",
    ),
)
_AFTER: tuple[_Before, ...] = (
    ("resolved", r"(?:now |has |had |since |which )?resolved|(?:now )?healed"),
    (
        "negated",
        r"(?:was |has been |been |now )?ruled out|(?:was |is )?excluded|negative|unlikely|"
        r"not (?:found|seen|present|confirmed)",
    ),
    (
        "uncertain",
        r"suspected|likely|possible|probable|presumed|questionable|not (?:ruled out|excluded)|"
        r"cannot be (?:ruled out|excluded)|(?:to be|still to be) ruled out",
    ),
)
_QUESTION = re.compile(r"\(\s*\?\s*\)|\?")


@dataclass(frozen=True, slots=True)
class _Rule:
    kind: CueKind
    pattern: re.Pattern[str]
    position: str  # "before" or "after"


# Cues after come first: where a word ends its segment ("pneumonia, likely") it qualifies what
# comes before it.
RULES: tuple[_Rule, ...] = (
    *(_Rule(kind, re.compile(_words(p) + r"(?=\s*(?:[,;)]|$))"), "after") for kind, p in _AFTER),
    *(_Rule(kind, re.compile(_words(p)), "before") for kind, p in _BEFORE),
)


def cues(text: str) -> tuple[Cue, ...]:
    """The cues of a normalised phrase, in order, without overlaps (longer matches win).

    A cue after a condition must end its segment (`PE ruled out`, `pneumonia, resolved`), so
    `possible` in `possible PE` is a cue before. A question mark is uncertain on either side:
    before the next words (`?PE`) or, at the end, after the previous ones (`PE?`).
    """
    candidates: list[tuple[int, int, int, CueKind, str]] = []
    for order, rule in enumerate(RULES):
        for match in rule.pattern.finditer(text):
            candidates.append((match.start(), match.end(), order, rule.kind, rule.position))
    for match in _QUESTION.finditer(text):
        after = not text[match.end() :].strip(" ,;)")
        candidates.append(
            (match.start(), match.end(), len(RULES), "uncertain", "after" if after else "before")
        )
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0], c[2]))
    taken: list[tuple[int, int, int, CueKind, str]] = []
    for candidate in candidates:
        if all(candidate[1] <= other[0] or candidate[0] >= other[1] for other in taken):
            taken.append(candidate)
    return tuple(
        Cue(
            kind=kind,
            text=text[start:end],
            start=start,
            end=end,
            position="after" if position == "after" else "before",
        )
        for start, end, _, kind, position in sorted(taken)
    )


@dataclass(frozen=True, slots=True)
class Judgement:
    status: Status
    certainty: Certainty
    reportable: bool  # whether the condition leads to a code (for a history, a history code)
    trace: tuple[TraceStep, ...]


_STATUS_OF: dict[CueKind, Status] = {
    "family_history": "family_history",
    "history": "history",
    "status_post": "status_post",
    "resolved": "resolved",
}


_STATUS_STEPS: dict[Status, tuple[str, str]] = {
    "history": ("personal history: a history Z code, not the condition", "I.C.21.c.4"),
    "family_history": ("family history: a family history Z code", "I.C.21.c.4"),
    "status_post": ("status post: a status Z code for a procedure or device", "I.C.21.c.3"),
    "resolved": (
        "resolved: a personal history, not a current condition, and not in remission",
        "I.C.21.c.4",
    ),
}


def judge(found: Iterable[Cue], setting: Setting = "inpatient") -> Judgement:
    """Status, certainty and reportability of a condition from the cues that cover it."""
    kinds = [cue.kind for cue in found]
    statuses = [_STATUS_OF[kind] for kind in kinds if kind in _STATUS_OF]
    status: Status = statuses[0] if statuses else "current"
    certainty: Certainty = (
        "negated" if "negated" in kinds else "uncertain" if "uncertain" in kinds else "affirmed"
    )
    steps: list[TraceStep] = []
    if status != "current":
        detail, guideline = _STATUS_STEPS[status]
        steps.append(text_step(status, detail, guideline))
    reportable = True
    if certainty == "negated":
        reportable = False
        steps.append(text_step("negated", "absent or ruled out: not coded"))
    elif certainty == "uncertain" and setting == "inpatient":
        steps.append(
            text_step(
                "uncertain",
                "uncertain diagnosis, inpatient: coded as if established",
                "II.H, III.C",
            )
        )
    elif certainty == "uncertain":
        reportable = False
        steps.append(
            text_step(
                "uncertain",
                "uncertain diagnosis, outpatient: not coded; code the symptoms instead",
                "IV.H",
            )
        )
    return Judgement(status, certainty, reportable, tuple(steps))
