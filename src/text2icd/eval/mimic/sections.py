"""The "Discharge Diagnosis" section of a MIMIC-IV-Note discharge summary, split into items.

Discharge summaries follow one template: labelled sections such as `Discharge Medications:`,
`Discharge Diagnosis:`, `Discharge Condition:`, each label alone on its line. The diagnosis section
runs from its label to the next main label. Inside it, diagnoses are listed one per line, often
under group labels (`Primary:`, `Secondary Diagnoses:`, `PRIMARY DIAGNOSIS` underlined with
`=====`, `Active issues:`, `PMH:`), with bullets or numbers, and sometimes several on one line
separated by semicolons.

`diagnosis_items(text)` returns the items in order, each with its group (`primary`, `secondary`
or None) when a group label says so. A section with more than `MAX_ITEMS` items is left out.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

Group = Literal["primary", "secondary"]

_START = re.compile(
    r"^[ \t]*(?:final[ \t]+)?discharge[ \t]+diagnos[ie]s(?:/es)?[ \t]*:?[ \t]*(?P<rest>[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)
# A label alone on its line: "Discharge Condition:", "Primary:", "SECONDARY DIAGNOSES".
_LABEL = re.compile(r"^[ \t]*(?P<label>[A-Za-z][A-Za-z /&()-]{0,40}?)[ \t]*:[ \t]*$")
# Template sections that can follow the diagnosis section; any of them ends it.
END_LABELS = frozenset(
    {
        "allergies",
        "attending",
        "brief hospital course",
        "chief complaint",
        "discharge condition",
        "discharge disposition",
        "discharge instructions",
        "discharge medications",
        "facility",
        "family history",
        "followup instructions",
        "history of present illness",
        "major surgical or invasive procedure",
        "medications on admission",
        "pertinent results",
        "physical exam",
        "social history",
        "condition at discharge",
        "mental status",
        "level of consciousness",
        "activity status",
    }
)
# Group labels inside the section, by the group they open.
_PRIMARY = re.compile(r"^(?:primary|principal|main|major|acute|active|final)\b", re.IGNORECASE)
_SECONDARY = re.compile(
    r"^(?:secondary|seconday|other|chronic|resolved|inactive|stable|past medical history|pmh|"
    r"history of|hx\b|comorbidit)",
    re.IGNORECASE,
)
_UNDERLINE = re.compile(r"^[ \t]*[-=_*~#.]{3,}[ \t]*$")
_BULLET = re.compile(r"^[ \t]*(?:[-*•·+>o]|#?\d{1,2}[.)]|\(\d{1,2}\)|[a-zA-Z][.)])[ \t]+")
_HASH = re.compile(r"^[ \t]*#[ \t]*")
_GROUP_WORDS = (
    r"(?:primary|secondary|seconday|principal|final|other|chronic|active|acute|resolved|"
    r"inactive|stable)(?:[ \t]+(?:discharge[ \t]+)?(?:diagnos[ie]s|diagnosis/es|dx|issues?|"
    r"problems?|conditions?))?"
)
# A group label alone on its line, with or without a colon: "Primary", "SECONDARY DIAGNOSES:".
_GROUP_ALONE = re.compile(rf"^[ \t]*(?P<label>{_GROUP_WORDS})[ \t]*:?[ \t]*$", re.IGNORECASE)
# A group label followed by a diagnosis: "Primary: sepsis", "Secondary diagnoses - anemia". The
# colon or dash is required, so that "Acute renal failure" stays a diagnosis.
_GROUP_INLINE = re.compile(
    rf"^[ \t]*(?P<label>{_GROUP_WORDS})[ \t]*(?::|[ \t]-)[ \t]*(?P<rest>.+)$", re.IGNORECASE
)
# Any other short label before a diagnosis that names a list: "Dx: sepsis", "PMH: asthma",
# "Chronic medical problems: CKD".
_ANY_LABEL = re.compile(
    r"^[ \t]*(?P<label>(?:[A-Za-z][A-Za-z /]{0,30}?[ /])?(?:diagnos[ie]s|diagnosis|dx|issues?|"
    r"problems?|conditions?|history|pmh|hx))[ \t]*:[ \t]*(?P<rest>.+)$",
    re.IGNORECASE,
)
MAX_ITEM = 200
# A section with more items than this is not a diagnosis list (0.02% of notes): left out.
MAX_ITEMS = 40


@dataclass(frozen=True, slots=True)
class DiagnosisItem:
    text: str
    position: int  # 0-based order within the section
    group: Group | None


def section(text: str) -> str | None:
    """The text of the first discharge diagnosis section, or None if the note has none."""
    start = _START.search(text)
    if start is None:
        return None
    lines = [start.group("rest")] if start.group("rest").strip() else []
    for line in text[start.end() :].split("\n")[1:]:
        label = _LABEL.match(line)
        if label and label.group("label").strip().casefold() in END_LABELS:
            break
        lines.append(line)
    return "\n".join(lines).strip("\n")


def _group(label: str) -> Group | None:
    label = label.strip()
    if _PRIMARY.match(label):
        return "primary"
    if _SECONDARY.match(label):
        return "secondary"
    return None


def diagnosis_items(text: str) -> list[DiagnosisItem]:
    """The diagnoses listed in a note's discharge diagnosis section, in order."""
    body = section(text)
    if body is None:
        return []
    items: list[DiagnosisItem] = []
    group: Group | None = None
    for raw in body.splitlines():
        if not raw.strip() or _UNDERLINE.match(raw):
            continue
        line = _HASH.sub("", _BULLET.sub("", raw.rstrip()))
        alone = _GROUP_ALONE.match(line)
        if alone:
            group = _group(alone.group("label"))
            continue
        label = _LABEL.match(line)
        if label:
            group = _group(label.group("label")) or group
            continue
        inline = _GROUP_INLINE.match(line) or _ANY_LABEL.match(line)
        if inline:
            group = _group(inline.group("label")) or group
            line = inline.group("rest")
        for part in line.split(";"):
            item = " ".join(part.split()).strip(" ,.-:")
            if _GROUP_ALONE.match(item):  # a group label with other punctuation ("Primary.")
                group = _group(item)
            elif 2 <= len(item) <= MAX_ITEM and any(c.isalpha() for c in item):
                items.append(DiagnosisItem(item, len(items), group))
    return items if len(items) <= MAX_ITEMS else []


@dataclass
class SectionStats:
    """Aggregate counts only: how many notes have a section, and how many items they give."""

    notes: int = 0
    with_section: int = 0
    with_items: int = 0
    items: int = 0
    groups: dict[str, int] = field(default_factory=dict)
    per_note: dict[str, int] = field(default_factory=dict)

    def add(self, text: str) -> list[DiagnosisItem]:
        self.notes += 1
        if section(text) is None:
            return []
        self.with_section += 1
        items = diagnosis_items(text)
        if items:
            self.with_items += 1
        self.items += len(items)
        for item in items:
            key = item.group or "none"
            self.groups[key] = self.groups.get(key, 0) + 1
        band = _band(len(items))
        self.per_note[band] = self.per_note.get(band, 0) + 1
        return items

    def lines(self) -> list[str]:
        def share(n: int, total: int) -> str:
            return f"{n / total:.1%}" if total else "n/a"

        groups = ", ".join(
            f"{name} {share(self.groups.get(name, 0), self.items)}"
            for name in ("primary", "secondary", "none")
        )
        bands = ", ".join(f"{band}: {self.per_note.get(band, 0):,}" for band in _BANDS)
        return [
            f"notes {self.notes:,}; with a discharge diagnosis section {self.with_section:,} "
            f"({share(self.with_section, self.notes)}); with items {self.with_items:,}",
            f"items {self.items:,}, {self.items / max(self.with_items, 1):.2f} per note with items",
            f"groups: {groups}",
            f"items per note: {bands}",
        ]


_BANDS = ("0", "1-5", "6-10", "11-20", "21-40")


def _band(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 10:
        return "6-10"
    return "11-20" if n <= 20 else "21-40"
