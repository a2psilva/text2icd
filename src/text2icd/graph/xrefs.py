"""Index cross-references: `see` and `see also` text resolved to the Index terms it names.

A coder reads a cross-reference as a path of comma-separated segments: `Complication, prosthetic
device, mechanical` is the main term Complication, its subterm "prosthetic device", then
"mechanical". Main-term titles contain commas themselves (`Abdomen, abdominal`), so at each level
the longest run of segments is tried first, and a segment may also hold several levels written
without commas ("Thyrotoxicosis with thyroid storm"). Titles and segments are compared in lenient
forms that ignore case, parenthesized words, NEC and NOS, hyphens, apostrophes, plurals, a trailing
connector (`Carrier of`), and the alternatives of "or" and of comma-separated titles.

Part of a reference can be an instruction rather than a path:

- `by site`, `by type`, `by drug`: the coder picks the subterm from the record;
- a column of the Neoplasm Table or the Table of Drugs and Chemicals (`Neoplasm, benign, by site`),
  matched against the column headings of the release;
- a book (`Table of Drugs and Chemicals`, `Index to Diseases and Injuries`);
- a code instruction (`category M01`, `N00-N07 with fourth character .8`) or a generic one
  (`condition`, `specific type of injury by site`), which names no term at all.

Each reference gets a status: `resolved` (every segment used), `partial` (a term was reached and the
words left over are kept in `remainder`, from which the coder continues), `instruction` (no term,
only an instruction) or `unresolved`. The unresolved references known to be errors in the Index are
listed, with a reason, in `resources/xref_exceptions.toml`; any other one is a build warning.
"""

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal, TypeVar

from text2icd.ontology.index import IndexBook, IndexSource, IndexTerm
from text2icd.ontology.validate import IndexReference, KnownException

XrefKind = Literal["see", "see_also"]
XrefStatus = Literal["resolved", "partial", "instruction", "unresolved"]
XREF_EXCEPTIONS = "xref_exceptions.toml"
_K = TypeVar("_K")

# Book names written at the start of a reference, followed by a comma or " - ".
_BOOK_NAMES: dict[str, IndexSource] = {
    "index to diseases and injuries": "index",
    "index to disease": "index",
    "alphabetic index": "index",
    "external cause of injuries index": "eindex",
    "index to external causes of injury": "eindex",
    "index to external causes": "eindex",
    "external cause index": "eindex",
    "table of neoplasms": "neoplasm",
    "neoplasm table": "neoplasm",
    "table of drugs and chemicals": "drug",
    "table of drugs and biologicals": "drug",
}
_BOOK_ALTERNATIVES = "|".join(map(re.escape, sorted(_BOOK_NAMES, key=len, reverse=True)))
_BOOK_PREFIX = re.compile(rf"^({_BOOK_ALTERNATIVES})(?:\s*,\s*|\s+-\s+|$)", re.IGNORECASE)
# `by site`, `by type of vehicle`: the coder chooses among the subterms. Other words after "by"
# start a subterm title ("by cesarean").
_BY_WORDS = (
    "site|type|drug|substance|animal|region|agent|degree|cause|histology|chromosome|specific"
)
_BY = rf"by\s+(?:{_BY_WORDS})\b"
_BY_SEGMENT = re.compile(rf"^{_BY}")
_BY_INSIDE = re.compile(rf"\s+({_BY}.*)$")
_GENERIC = re.compile(r"^(?:condition|specific|specified|nature of|substance specified)\b")
_CODE_INSTRUCTION = re.compile(
    r"^(?i:(?:sub)?categor(?:y|ies)|codes?)\b|^[A-Z](?:[0-9][0-9A-Z]|[A-Z][0-9])(?![A-Za-z0-9])"
)
# A stray "also" from a `<see>also ...</see>` element, or a leading dash.
_LEAD = re.compile(r"^(?:-\s*|also\b\s*)+")
# "see Levo-": every substance whose name starts with the prefix.
_NAME_PREFIX = re.compile(r"^[A-Za-z]+-$")
# `Carrier (suspected) of`, written `Carrier, hepatitis` in references.
_TRAILING_CONNECTOR = re.compile(r"\s+(?:of|to|with|in|from|for|by|on)$")
_PARENS = re.compile(r"\s*\([^()]*\)")
_UNWRAP = re.compile(r"\s*\((-?)([^()]*)\)")
_QUALIFIER = re.compile(r"\s+(?:nec|nos)$")
# Words that cannot name a table column on their own (`in` of "Ca in situ").
_COLUMN_STOPWORDS = frozenset({"in", "of", "and", "or", "with"})


@dataclass(frozen=True, slots=True)
class CrossReference:
    """A `see` or `see also` of one Index term. Term ids are the global ids of `index_term`."""

    term: int
    kind: XrefKind
    written: str
    status: XrefStatus
    book: IndexSource | None = None
    target: int | None = None
    column: int | None = None
    instruction: str | None = None
    remainder: str | None = None
    reason: str | None = None  # why an unresolved reference is a known error


@dataclass(frozen=True, slots=True)
class XrefResult:
    xrefs: tuple[CrossReference, ...]
    warnings: tuple[str, ...]


def cross_references(
    books: Mapping[IndexSource, IndexBook],
    offsets: Mapping[IndexSource, int],
    exceptions: Sequence[KnownException] = (),
) -> XrefResult:
    """Resolve the `see` and `see also` of every term, in book and term order. `exceptions` are
    the known unresolved references of the release; each one gives its reason to the reference."""
    known = {exception.reference: exception for exception in exceptions}
    seen: set[IndexReference] = set()
    resolver = XrefResolver(books, offsets)
    found = []
    warnings = []
    for source, book in books.items():
        paths = book.paths()
        for term in book.terms:
            references: tuple[tuple[XrefKind, str | None], ...] = (
                ("see", term.see),
                ("see_also", term.see_also),
            )
            for kind, written in references:
                if not written:
                    continue
                xref = resolver.resolve(source, offsets[source] + term.id, kind, written)
                if xref.status == "unresolved":
                    reference = IndexReference(source, paths[term.id], kind, written)
                    exception = known.get(reference)
                    if exception is None:
                        warnings.append(
                            f"unresolved {kind} of {source}: {reference.path}: {written!r}"
                        )
                    else:
                        seen.add(reference)
                        xref = replace(xref, reason=exception.reason)
                found.append(xref)
    for reference in sorted(set(known) - seen, key=lambda r: (r.book, r.path, r.field)):
        warnings.append(
            f"cross-reference exception not observed: {reference.book}: {reference.path} "
            f"[{reference.field}] {reference.value!r}"
        )
    return XrefResult(tuple(found), tuple(warnings))


@dataclass(frozen=True, slots=True)
class _Match:
    """A term reached by a reference, how well, and the segments it left over."""

    left: int  # words left over, after columns
    rank: int  # 0 for exact forms; one more for each step taken through an alternate form
    book_order: int
    breadth: int  # subterms of the main term the match starts from
    target: int
    book: IndexSource
    column: int | None
    remainder: tuple[str, ...]

    def key(self) -> tuple[int, int, int, int, int]:
        # Fewest words left, then exact titles, then the term's own book. Between main terms with
        # the same head ("Fracture, burst", "Fracture, traumatic"), the one with the most subterms
        # is the main entry; then Index order.
        return (self.left, self.rank, self.book_order, -self.breadth, self.target)


class XrefResolver:
    """Resolves cross-reference text against the Index books of one release."""

    def __init__(
        self, books: Mapping[IndexSource, IndexBook], offsets: Mapping[IndexSource, int]
    ) -> None:
        self._books = books
        self._offsets = offsets
        self._main: dict[str, list[tuple[tuple[IndexSource, int], int]]] = {}
        for source, book in books.items():
            for term_id in book.main_terms:
                for form, rank in _term_forms(book[term_id]).items():
                    self._main.setdefault(form, []).append(((source, term_id), rank))
        self._children: dict[tuple[IndexSource, int], dict[str, list[tuple[int, int]]]] = {}
        self._columns = {
            source: [
                (col, frozenset(re.findall(r"[a-z-]+", heading.casefold())))
                for col, heading in sorted(book.columns.items())
                if col > 1  # column 1 heads the term titles; the codes start in column 2
            ]
            for source, book in books.items()
        }

    def resolve(
        self, source: IndexSource, term: int, kind: XrefKind, written: str
    ) -> CrossReference:
        text = _LEAD.sub("", " ".join(written.split()))
        if (
            _CODE_INSTRUCTION.match(text)
            or _GENERIC.match(text.casefold())
            or _NAME_PREFIX.match(text)
        ):
            return CrossReference(term, kind, written, "instruction", instruction=text)
        books: Sequence[IndexSource] = [source, *(b for b in self._books if b != source)]
        book: IndexSource | None = None
        prefix = _BOOK_PREFIX.match(text)
        if prefix:
            book = _BOOK_NAMES[prefix.group(1).casefold()]
            books, text = [book], text[prefix.end() :]
        path, instructions, after = _segments(text)
        instruction = ", ".join(instructions) or None
        best = min(self._matches(path, after, books, book), key=_Match.key, default=None)
        if best is not None:
            remainder = ", ".join(best.remainder) or None
            return CrossReference(
                term,
                kind,
                written,
                "partial" if remainder else "resolved",
                best.book,
                self._offsets[best.book] + best.target,
                best.column,
                instruction,
                remainder,
            )
        column, left = self._tail(book, path + after) if book else (None, path + after)
        if not left and (book or instruction):
            return CrossReference(
                term, kind, written, "instruction", book, None, column, instruction
            )
        remainder = ", ".join(path + after) or None
        return CrossReference(
            term, kind, written, "unresolved", book, instruction=instruction, remainder=remainder
        )

    def _matches(
        self,
        path: tuple[str, ...],
        after: tuple[str, ...],
        books: Sequence[IndexSource],
        within: IndexSource | None,
    ) -> Iterator[_Match]:
        starts = [
            (source, term_id, rank, rest)
            for lead, rest in _leads(path)
            for (source, term_id), rank in _lookup(self._main, lead).items()
            if source in books
        ]
        if within is not None and len(self._books[within].main_terms) == 1:
            # "Table of Neoplasms, small intestine": the rows of the table are subterms of its one
            # main term, `Neoplasm, neoplastic`.
            starts.append((within, self._books[within].main_terms[0], 0, path))
        for source, term_id, rank, rest in starts:
            breadth = len(self._books[source][term_id].children)
            for target, rank_below, left in self._descend(source, term_id, rest):
                column, remainder = self._tail(source, left + after)
                yield _Match(
                    sum(len(segment.split()) for segment in remainder),
                    rank + rank_below,
                    books.index(source),
                    breadth,
                    target,
                    source,
                    column,
                    remainder,
                )

    def _descend(
        self, source: IndexSource, term_id: int, segments: tuple[str, ...]
    ) -> Iterator[tuple[int, int, tuple[str, ...]]]:
        """Every term reachable below `term_id` by leading segments, with the segments left."""
        yield term_id, 0, segments
        children = self._child_forms(source, term_id)
        for lead, rest in _leads(segments):
            for child, rank in _lookup(children, lead).items():
                for target, rank_below, left in self._descend(source, child, rest):
                    yield target, rank + rank_below, left

    def _child_forms(self, source: IndexSource, term_id: int) -> dict[str, list[tuple[int, int]]]:
        key = (source, term_id)
        if key not in self._children:
            book = self._books[source]
            forms: dict[str, list[tuple[int, int]]] = {}
            for child in book[term_id].children:
                for form, rank in _term_forms(book[child]).items():
                    forms.setdefault(form, []).append((child, rank))
            self._children[key] = forms
        return self._children[key]

    def _tail(
        self, source: IndexSource, segments: tuple[str, ...]
    ) -> tuple[int | None, tuple[str, ...]]:
        """Take a table column out of the segments left: `malignant`, `adverse effect`."""
        column = None
        left = []
        for segment in segments:
            found = self._column(source, segment) if column is None else None
            if found is None:
                left.append(segment)
            column = column or found
        return column, tuple(left)

    def _column(self, source: IndexSource, segment: str) -> int | None:
        words = set(segment.split())
        if not words - _COLUMN_STOPWORDS:
            return None
        for col, heading in self._columns.get(source, ()):
            if words <= heading:
                return col
        return None


def _segments(text: str) -> tuple[tuple[str, ...], list[str], tuple[str, ...]]:
    """Split a reference into the path to follow, its instructions, and the segments after the
    first instruction: in `Diabetes, by type, with coma` the coder goes to Diabetes, picks the type
    from the record, and only then looks for "with coma"."""
    path: list[str] = []
    instructions: list[str] = []
    after: list[str] = []
    parts = [part.strip() for part in text.split(",")]
    for i, part in enumerate(parts):
        if _CODE_INSTRUCTION.match(part):
            # A code instruction runs to the end: "categories T36-T50, with final character 6".
            instructions.append(", ".join(parts[i:]))
            break
        segment = _normalize(part)
        if not segment:
            continue
        inside = _BY_INSIDE.search(segment)
        if _BY_SEGMENT.match(segment) or (inside and inside.start() > 0):
            if inside and not _BY_SEGMENT.match(segment):
                (after if instructions else path).append(segment[: inside.start()])
                segment = inside.group(1)
            instructions.append(segment)
            continue
        (after if instructions else path).append(segment)
    return tuple(path), instructions, tuple(after)


def _leads(segments: tuple[str, ...]) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Ways to take the next step: a run of whole segments, or the first words of one segment."""
    if not segments:
        return
    for used in range(len(segments), 0, -1):
        yield ", ".join(segments[:used]), segments[used:]
    words = segments[0].split()
    for used in range(len(words) - 1, 0, -1):
        yield " ".join(words[:used]), (" ".join(words[used:]), *segments[1:])


def _lookup(table: Mapping[str, list[tuple[_K, int]]], text: str) -> dict[_K, int]:
    """Entries of `table` under any lenient form of `text`, each with its best rank."""
    hits: dict[_K, int] = {}
    for form in _forms(text):
        for key, rank in table.get(form, ()):
            hits[key] = min(rank, hits.get(key, rank))
    return hits


def _normalize(text: str) -> str:
    text = text.casefold().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    return " ".join(text.split()).strip(" .,;:")


@lru_cache(maxsize=65536)
def _forms(text: str) -> frozenset[str]:
    """Lenient forms of a title or segment: `Light-for-dates (infant)` -> `light for dates`, ..."""
    base = _normalize(text)
    forms = {base, _normalize(_PARENS.sub("", base))}
    forms.add(_normalize(_UNWRAP.sub(lambda m: (m[1] or " ") + m[2], base)))
    forms |= {_QUALIFIER.sub("", form) for form in forms}
    forms |= {form.replace("-", " ") for form in forms}
    forms |= {form.replace("'", "") for form in forms}  # Chagas' disease
    forms |= {" ".join(form.replace(",", " ").split()) for form in forms}
    forms |= {form[:-1] for form in forms if form.endswith("s") and len(form) > 3}
    return frozenset(form for form in forms if form)


def _term_forms(term: IndexTerm) -> dict[str, int]:
    """Lenient forms of a term's title with their rank: 0 for the title and for its head, the part
    before the first comma (`Fracture, traumatic`, `Disease, diseased`); 1 for the other parts and
    the alternatives of "or" (`Abortion or miscarriage`)."""
    exact = _forms(term.title) | _forms(term.display_title)
    if "s" in term.nonessential:  # Complication(s)
        exact |= _forms(term.title + "s")
    heads = {form for title in exact for form in _forms(title.split(",")[0])}
    alternates = {
        form for title in exact for part in re.split(r",| or ", title) for form in _forms(part)
    }
    alternates |= {_TRAILING_CONNECTOR.sub("", form) for form in exact | heads | alternates}
    ranked = dict.fromkeys(alternates, 1)
    ranked.update(dict.fromkeys(exact | heads, 0))
    return ranked
