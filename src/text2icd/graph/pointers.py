"""Index-to-Tabular pointers: every code an Index term gives, resolved to a Tabular node.

A term can give a code (`<code>`), a manifestation code (`<manif>`), a subcategory or category to
consult (`<subcat>`, `<seecat>`) and, in the Neoplasm and Drug tables, one code per column. A code
ending in `-` is incomplete (Guideline I.B.1) and points to the Tabular node it names, from which
the coder continues to a billable code.

Known data errors are listed in `resources/index_exceptions.toml`. An entry with a correction
(`code`, `manifestation`) replaces the written value; an entry without one keeps the pointer out of
the graph with a warning. Anything else that does not resolve is marked `unresolved`, which the
integrity checks reject.
"""

from collections import Counter
from collections.abc import Container, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from text2icd.ontology.codes import dot, is_code, is_partial
from text2icd.ontology.index import NOT_APPLICABLE, IndexBook, IndexSource, IndexTerm
from text2icd.ontology.validate import IndexReference, KnownException

PointerField = Literal["code", "manifestation", "subcategory", "see_category", "cell"]
PointerStatus = Literal["ok", "corrected", "excluded", "unresolved"]

_SINGLE_FIELDS: tuple[PointerField, ...] = ("code", "manifestation", "subcategory", "see_category")


@dataclass(frozen=True, slots=True)
class Pointer:
    """One code given by an Index term. `column` is the table column of a cell, otherwise 0."""

    term: int
    field: PointerField
    column: int
    written: str
    code: str | None
    partial: bool
    target: str | None
    status: PointerStatus


@dataclass(frozen=True, slots=True)
class PointerResult:
    pointers: tuple[Pointer, ...]
    warnings: tuple[str, ...]


def parse_pointer(value: str) -> tuple[str | None, bool]:
    """`C79.8-` -> ("C79.8", True); `E85.4` -> ("E85.4", False); text -> (None, False)."""
    raw = value.strip()
    partial = is_partial(raw)
    bare = raw.rstrip("-").rstrip(".") if partial else raw
    if not is_code(bare):
        return None, False
    return dot(bare), partial


def index_pointers(
    books: Mapping[IndexSource, IndexBook],
    offsets: Mapping[IndexSource, int],
    nodes: Container[str],
    exceptions: Sequence[KnownException],
) -> PointerResult:
    """Resolve every code of every term. `exceptions` must already be those of the release."""
    known = {exception.reference: exception for exception in exceptions}
    seen: set[IndexReference] = set()
    pointers: list[Pointer] = []
    warnings: list[str] = []
    for source, book in books.items():
        offset = offsets[source]
        paths = book.paths()
        for term in book.terms:
            for field, column, written in _fields(term):
                name = f"cell {column}" if field == "cell" else field
                reference = IndexReference(source, paths[term.id], name, written)
                exception = known.get(reference)
                if exception is None:
                    pointers.append(_resolve(offset + term.id, field, column, written, nodes))
                    continue
                seen.add(reference)
                corrected = _apply(offset + term.id, exception, nodes)
                if not corrected:
                    warnings.append(
                        f"excluded {source}: {reference.path} [{name}] {written!r}: "
                        f"{exception.reason}"
                    )
                    pointers.append(
                        Pointer(
                            offset + term.id, field, column, written, None, False, None, "excluded"
                        )
                    )
                pointers.extend(corrected)
    for reference in sorted(set(known) - seen, key=lambda r: (r.book, r.path, r.field)):
        warnings.append(
            f"exception not found in the Index: {reference.book}: {reference.path} "
            f"[{reference.field}] {reference.value!r}"
        )
    return PointerResult(_check_unique(pointers), tuple(warnings))


def _fields(term: IndexTerm) -> Iterator[tuple[PointerField, int, str]]:
    for field in _SINGLE_FIELDS:
        value = getattr(term, field)
        if value:
            yield field, 0, value
    for column, value in term.cells:
        if value not in NOT_APPLICABLE:
            yield "cell", column, value


def _resolve(
    term: int, field: PointerField, column: int, written: str, nodes: Container[str]
) -> Pointer:
    code, partial = parse_pointer(written)
    target = code if code is not None and code in nodes else None
    status: PointerStatus = "ok" if target is not None else "unresolved"
    return Pointer(term, field, column, written, code, partial, target, status)


def _apply(term: int, exception: KnownException, nodes: Container[str]) -> list[Pointer]:
    """The pointers an exception's correction gives, or [] when it has none."""
    written = exception.reference.value
    corrections: tuple[tuple[PointerField, str | None], ...] = (
        ("code", exception.code),
        ("manifestation", exception.manifestation),
    )
    pointers = []
    for field, value in corrections:
        if not value:
            continue
        code, partial = parse_pointer(value)
        if code is None or code not in nodes:
            raise ValueError(
                f"index_exceptions.toml corrects {exception.reference.path} to {value!r}, "
                "which is not a Tabular code of this release"
            )
        pointers.append(Pointer(term, field, 0, written, code, partial, code, "corrected"))
    return pointers


def _check_unique(pointers: list[Pointer]) -> tuple[Pointer, ...]:
    counts = Counter((p.term, p.field, p.column) for p in pointers)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"an exception correction duplicates an existing pointer: {duplicates}")
    return tuple(pointers)
