"""The Index router: which book a coder opens for a diagnosis, and which column (Guidelines
I.A.1, I.C.2, I.C.19).

- **Neoplasm Table.** A phrase that states a neoplasm's behaviour ("malignant neoplasm of",
  "secondary malignant neoplasm of", "carcinoma in situ of", "benign neoplasm of", "neoplasm of
  uncertain behavior", "cancer of") is looked up by site in the Neoplasm Table, in the column of
  that behaviour; "neoplasm of" with no behaviour reads Unspecified Behavior. Morphology words
  ("adenocarcinoma") are not routed: their Index entries send the coder to the right column
  (`walk`, cross-references).
- **Table of Drugs and Chemicals.** Poisoning, overdose, adverse effect, underdosing and toxicity
  wording is looked up by substance, in the column of its intent (I.C.19.e): self-harm, assault
  and undetermined intent only when written; a poisoning of unknown intent is accidental
  (I.C.19.e.5(b)); "toxicity" of a drug is read as an adverse effect, flagged, since a drug
  correctly prescribed and taken is an adverse effect (I.C.19.e.5(a)).
- **External Cause Index.** A stated mechanism of injury (a fall, a collision, a bite) also gets
  an external cause code, with role `external_cause` (I.C.20); none is added otherwise.
- Everything else, and the words left over, go to the main Index.

The cue words a route reads count as used by the walk, so a routed walk competes with the main
Index walk on equal terms.
"""

import re
from dataclasses import dataclass
from typing import Literal

Book = Literal["neoplasm", "drug", "eindex"]

_OF = r"(?:\s+(?:of|in|to|on|with|by|from))?"
# Neoplasm behaviour, most specific first; the column numbers are those of the table.
_NEOPLASM: tuple[tuple[int, str], ...] = (
    (
        3,
        r"secondary malignant (?:neoplasm|tumou?r|cancer)s?|"
        r"metastatic (?:\w+ )?(?:cancer|carcinoma|neoplasm|tumou?r|disease)s?|metastas[ie]s|"
        r"secondary (?:neoplasm|cancer|tumou?r)s?",
    ),
    (4, r"carcinoma in situ|in situ (?:neoplasm|carcinoma)s?|neoplasms? in situ"),
    (
        6,
        r"(?:neoplasm|tumou?r)s? of uncertain (?:behaviou?r|malignant potential)|"
        r"uncertain behaviou?r(?: neoplasms?)?",
    ),
    (
        7,
        r"(?:neoplasm|tumou?r)s? of unspecified behaviou?r|unspecified behaviou?r(?: neoplasms?)?",
    ),
    (5, r"benign (?:neoplasm|tumou?r)s?"),
    (2, r"(?:primary )?malignant (?:neoplasm|tumou?r)s?|cancer|malignancy|malignancies"),
    (7, r"neoplasms?"),
)
# Drug Table intent. Columns: 2 accidental, 3 self-harm, 4 assault, 5 undetermined,
# 6 adverse effect, 7 underdosing.
_INTENT: tuple[tuple[int, str], ...] = (
    (
        3,
        r"intentional self[- ]poisoning|intentional (?:overdose|poisoning)|self[- ]harm|"
        r"suicid(?:e|al)(?: attempt)?(?: by)?(?: overdose| poisoning)?",
    ),
    (4, r"assault by poisoning|homicidal poisoning|poisoning by another person"),
    (5, r"undetermined intent|intent undetermined|undetermined"),
    (6, r"adverse (?:effect|reaction)s?|side effects?"),
    (7, r"underdos(?:ing|ed|e)"),
)
_POISONING = r"(?:accidental )?(?:poisoning|overdose|ingestion|toxic effects?)|accidental"
_TOXICITY = r"toxicity|toxic"
# Mechanisms of injury that call for an external cause code.
_EXTERNAL = re.compile(
    r"\b(?:fall|falls|fell|falling|slipped|tripped|struck|hit by|collision|motor vehicle|"
    r"mva|mvc|traffic accident|pedestrian|cyclist|bitten|bite|stung|sting|gunshot|stabbed|"
    r"assaulted|drowning|explosion|crushed)\b"
)


@dataclass(frozen=True, slots=True)
class Route:
    """A book to look a diagnosis up in, and what to look up."""

    book: Book
    column: int | None  # the table column; None for the External Cause Index
    cue: str  # the words that chose the route, as written
    text: str  # what is looked up: the site, the substance, or the whole phrase
    guideline: str
    flags: tuple[str, ...] = ()


def route(text: str) -> list[Route]:
    """The routes a diagnosis takes besides the main Index, best first."""
    lowered = text.casefold()
    routes: list[Route] = []
    drug = _drug(lowered)
    if drug is not None:
        routes.append(drug)
    else:
        neoplasm = _neoplasm(lowered)
        if neoplasm is not None:
            routes.append(neoplasm)
    if drug is None and (found := _EXTERNAL.search(lowered)) is not None:
        routes.append(Route("eindex", None, found.group(0), lowered, "I.C.20"))
    return routes


def _neoplasm(text: str) -> Route | None:
    for column, pattern in _NEOPLASM:
        found = re.search(rf"\b(?:{pattern}){_OF}\b", text)
        if found is not None:
            return Route("neoplasm", column, found.group(0), _cut(text, [found.span()]), "I.C.2")
    return None


def _drug(text: str) -> Route | None:
    spans: list[tuple[int, int]] = []
    column: int | None = None
    for intent, pattern in _INTENT:
        found = re.search(rf"\b(?:{pattern}){_OF}\b", text)
        if found is not None:
            spans.append(found.span())
            column = column or intent
    poisoning = re.search(rf"\b(?:{_POISONING}){_OF}\b", text)
    if poisoning is not None:
        spans.append(poisoning.span())
    flags: tuple[str, ...] = ()
    guideline = "I.C.19.e"
    if column is None and poisoning is not None:
        column = 2  # intent unknown or unspecified: accidental
        if "accidental" not in text:
            flags, guideline = ("intent_unknown_accidental",), "I.C.19.e.5(b)"
    if column is None:
        toxicity = re.search(rf"\b(?:{_TOXICITY}){_OF}\b", text)
        if toxicity is None:
            return None
        spans.append(toxicity.span())
        column, flags, guideline = 6, ("toxicity_as_adverse_effect",), "I.C.19.e.5(a)"
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    cue = " ".join(text[start:end].strip() for start, end in merged)
    return Route("drug", column, cue, _cut(text, spans), guideline, flags)


def _cut(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return " ".join("".join(chars).replace(",", " ").split())
