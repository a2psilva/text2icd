"""Consistency checks between the Alphabetic Index and the Tabular List of a release."""

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from text2icd.ontology.codes import dot, is_code, is_partial, strip_partial
from text2icd.ontology.index import NOT_APPLICABLE, IndexBook, IndexSource, IndexTerm
from text2icd.ontology.tabular import Tabular

_REFERENCE_FIELDS = ("code", "manifestation", "subcategory", "see_category")


@dataclass(frozen=True, slots=True)
class IndexReference:
    """One code an Index term points to, located by its book, path and field."""

    book: IndexSource
    path: str
    field: str
    value: str


@dataclass(frozen=True, slots=True)
class KnownException:
    reference: IndexReference
    releases: frozenset[str]
    reason: str
    code: str | None = None
    manifestation: str | None = None


def references(book: IndexBook) -> Iterable[IndexReference]:
    for term in book.terms:
        for field, value in _values(term):
            yield IndexReference(book.source, " > ".join(book.path(term.id)), field, value)


def unresolved_references(
    books: Mapping[IndexSource, IndexBook], tabular: Tabular, billable: frozenset[str]
) -> list[IndexReference]:
    """Index references whose code is not in the Tabular List or the billable set."""
    return [
        ref
        for book in books.values()
        for ref in references(book)
        if not _resolves(ref.value, tabular, billable)
    ]


def load_exceptions(path: Path | None = None) -> list[KnownException]:
    if path is None:
        raw = resources.files("text2icd.resources").joinpath("index_exceptions.toml").read_bytes()
    else:
        raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    if data.get("schema") != 1:
        raise ValueError(f"unsupported exceptions schema {data.get('schema')!r}")
    exceptions = []
    for entry in data.get("exception", []):
        reference = IndexReference(entry["book"], entry["path"], entry["field"], entry["value"])
        exceptions.append(
            KnownException(
                reference=reference,
                releases=frozenset(entry["releases"]),
                reason=entry["reason"],
                code=entry.get("code"),
                manifestation=entry.get("manifestation"),
            )
        )
    return exceptions


def _values(term: IndexTerm) -> Iterable[tuple[str, str]]:
    for field in _REFERENCE_FIELDS:
        value = getattr(term, field)
        if value:
            yield field, value
    for column, value in term.cells:
        if value not in NOT_APPLICABLE:
            yield f"cell {column}", value


def _resolves(value: str, tabular: Tabular, billable: frozenset[str]) -> bool:
    raw = value.strip()
    if not is_code(raw.rstrip("-").rstrip(".")):
        return False
    code = strip_partial(raw) if is_partial(raw) else dot(raw)
    return code in tabular or code in billable
