"""Code references in Tabular List notes: `(N18.1-N18.6)`, `(E10.-)`, `category T51`, `J09-J18`.

Instructional notes name codes in free text. The edges of the graph (Excludes1, Excludes2, code
first, use additional code, code also) come from the references found here; the note text is kept
next to every edge, because it often qualifies the reference ("with fifth or sixth character 5").

A range resolves when at least one Tabular code lies inside it: some ranges end at a category that
only exists in the WHO ICD-10 (`K00-K93`). A bare three-character token that matches no Tabular
code is not a reference but a word that looks like one (`vitamin B12`, `GM2 gangliosidosis`). Any
other reference with no code inside is kept as unresolved: it is an error in the notes (the FY2026
F02 note names `G31.90`, a code that does not exist).
"""

import re
from bisect import bisect_left
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from text2icd.ontology.codes import CodeReference, parse_reference, undot

_CATEGORY = r"[A-Z](?:[0-9][0-9A-Z]|[A-Z][0-9])"
# A code as notes write it, allowing the stray dots found in the files (`M51.1.-`, `O33.3.`).
_CODE = rf"{_CATEGORY}(?:\.[0-9A-Z]{{1,4}})?(?:\.?-)?\.?"
_DASH = "[-\N{EN DASH}]"
_REFERENCE = re.compile(rf"(?<![A-Za-z0-9.])({_CODE})(?:\s*{_DASH}\s*({_CODE}))?(?![A-Za-z0-9])")


@dataclass(frozen=True, slots=True)
class NoteReference:
    """A reference as written in a note, and the code range it stands for."""

    written: str
    reference: CodeReference

    @property
    def is_range(self) -> bool:
        return self.reference.start != self.reference.end

    @property
    def is_bare(self) -> bool:
        """A lone category with no dot or dash (`B12`, `GM2`): a word if no such code exists."""
        return not self.is_range and "." not in self.written and "-" not in self.written


def find_references(text: str) -> list[NoteReference]:
    """Every code or range written in `text`, in order of appearance."""
    found = []
    for match in _REFERENCE.finditer(text):
        first, last = match.group(1).rstrip("."), (match.group(2) or "").rstrip(".")
        start = parse_reference(first)
        end = parse_reference(last) if last else start
        if start.start > end.end:
            continue
        found.append(NoteReference(match.group(0), CodeReference(start.start, end.end)))
    return found


class CodeIndex:
    """The codes of a release, sorted, to test whether any of them lies in a reference."""

    def __init__(self, codes: Iterable[str]) -> None:
        self._sorted = sorted({undot(code) for code in codes})

    def any_in(self, reference: CodeReference) -> bool:
        # The first code at or after `start` has the smallest prefix that is >= start. Prefixes
        # only grow from there, so if this code is past `end`, every later one is too.
        i = bisect_left(self._sorted, reference.start)
        return i < len(self._sorted) and self._sorted[i][: len(reference.end)] <= reference.end

    def __iter__(self) -> Iterator[str]:
        return iter(self._sorted)
