"""The billable code set of a release, generated from the Tabular List.

A leaf code is billable as it is, unless a 7th-character definition applies to it (its own or the
nearest ancestor's). Then each 7th character makes one billable code, padded with the `X`
placeholder (Guidelines I.A.4, I.A.5). Exceptions stated in instructional notes are listed in
`SEVENTH_CHARACTER_EXCEPTIONS`; each must still be present in the notes, or generation fails so the
rule is reviewed rather than silently applied to a release that changed it.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from text2icd.ontology.codes import (
    CODE_LENGTH_BEFORE_SEVENTH,
    PLACEHOLDER,
    undot,
    with_seventh_character,
)
from text2icd.ontology.tabular import Tabular


@dataclass(frozen=True, slots=True)
class SeventhCharacterException:
    """7th characters that do not apply to some codes of a category, as a Tabular note says."""

    category: str
    sixth_characters: frozenset[str]
    excluded: frozenset[str]
    note: str


SEVENTH_CHARACTER_EXCEPTIONS = (
    SeventhCharacterException(
        category="S06",
        sixth_characters=frozenset("78"),
        excluded=frozenset("DS"),
        note="7th characters D and S do not apply to codes in category S06 with 6th character 7",
    ),
)


@dataclass(frozen=True, slots=True)
class BillableCode:
    """One billable code and where it comes from in the Tabular List.

    `base` is the Tabular leaf; `seventh` is the 7th character added to it (None when the leaf is
    billable as it is) and `owner` the node whose 7th-character definition supplied it.
    """

    code: str
    base: str
    seventh: str | None = None
    owner: str | None = None


def generate_billable(tabular: Tabular) -> frozenset[str]:
    return frozenset(entry.code for entry in iter_billable(tabular))


def iter_billable(tabular: Tabular) -> Iterator[BillableCode]:
    """Every billable code in Tabular List order; 7th-character codes follow their leaf."""
    exceptions = [e for e in SEVENTH_CHARACTER_EXCEPTIONS if e.category in tabular]
    for exception in exceptions:
        notes = (note.text for note in tabular[exception.category].notes)
        if not any(note.startswith(exception.note) for note in notes):
            raise ValueError(
                f"the {exception.category} note behind a 7th-character exception has changed; "
                "review SEVENTH_CHARACTER_EXCEPTIONS"
            )

    for node in tabular.diags():
        if node.children:
            continue
        owner = tabular.seventh_character_owner(node.key)
        if owner is None:
            yield BillableCode(node.key, node.key)
            continue
        padded = undot(node.key).ljust(CODE_LENGTH_BEFORE_SEVENTH, PLACEHOLDER)
        excluded: set[str] = set()
        for exception in exceptions:
            if padded.startswith(exception.category) and padded[5] in exception.sixth_characters:
                excluded |= exception.excluded
        for char, _ in tabular[owner].seventh_characters:
            if char not in excluded:
                yield BillableCode(with_seventh_character(node.key, char), node.key, char, owner)
