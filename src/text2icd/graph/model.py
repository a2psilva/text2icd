"""The built graph of a release, loaded into memory: `load("fy2027") -> Graph`.

Every node and edge is read once into dicts and named tuples, so traversal never touches the disk.
Search runs on the FTS5 tables through a read-only connection to the same file.

Tabular codes are addressed by their dotted key (`E11.22`, also given as `E1122`) and Index terms by
their global id or by book and path (`Pneumonia > aspiration`).
"""

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing
from functools import cached_property
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from types import TracebackType
from typing import Literal, NamedTuple, Self

from text2icd.config import Settings
from text2icd.graph.build import PATH_SEPARATOR, SCHEMA_VERSION
from text2icd.graph.search import IndexHit, TabularHit, search_index, search_tabular
from text2icd.ontology.codes import dot, is_code
from text2icd.releases import load_registry

NodeKind = Literal["chapter", "section", "diag", "generated"]


class Code(NamedTuple):
    """A Tabular List node: chapter, section, code, or a code generated from a 7th character."""

    key: str
    kind: NodeKind
    parent: str | None
    depth: int
    description: str
    billable: bool
    placeholder: bool
    seventh_from: str | None  # the node whose 7th-character definition applies


class Note(NamedTuple):
    node: str
    kind: str
    text: str
    id: int


class Edge(NamedTuple):
    """A code or range named in a note of `source`."""

    source: str
    kind: str
    written: str
    start: str
    end: str
    target: str | None
    status: str
    note: str  # the note text, which often qualifies the reference
    note_id: int


class Term(NamedTuple):
    """An Index term. `id` is global across the four books."""

    id: int
    book: str
    parent: int | None
    level: int
    title: str
    printed_title: str
    nonessential: tuple[str, ...]
    path: str


class Pointer(NamedTuple):
    """A code an Index term gives; `column` is the table column of a cell, otherwise 0."""

    term: int
    field: str
    column: int
    written: str
    code: str | None
    partial: bool
    target: str | None
    status: str


class CrossRef(NamedTuple):
    term: int
    kind: str
    written: str
    status: str
    book: str | None
    target: int | None
    column: int | None
    instruction: str | None
    remainder: str | None
    reason: str | None


class TabularText(NamedTuple):
    """A description, inclusion term or `includes` note of a Tabular node, as searched."""

    id: int
    node: str
    kind: Literal["description", "inclusion", "includes"]
    text: str


class GraphError(Exception):
    """The graph file is missing or was built by an incompatible version."""


class Graph:
    """Read-only accessors over one release. Use `load()` to build one."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._uri = f"{path.resolve().as_uri()}?mode=ro"
        self._local = threading.local()
        self._opened: list[tuple[int, sqlite3.Connection]] = []  # (process id, connection)
        self._opening = threading.Lock()
        self._closed = False
        try:
            self.meta: dict[str, str] = dict(self._conn.execute("SELECT key, value FROM meta"))
        except sqlite3.DatabaseError as exc:
            self.close()
            raise GraphError(f"{path} is not a readable graph file: {exc}") from exc
        if self.meta.get("schema_version") != str(SCHEMA_VERSION):
            self.close()
            raise GraphError(
                f"{path} has schema version {self.meta.get('schema_version')}, this text2icd "
                f"reads {SCHEMA_VERSION}; rebuild it with `text2icd build-graph`"
            )
        self.release = self.meta["release"]
        self._load_tabular()
        self._load_index()

    # -- Tabular List -------------------------------------------------------------------------

    def code(self, key: str) -> Code:
        """The Tabular node `key` (`E11.22` or `E1122`); KeyError if the release lacks it."""
        return self._codes[self._key(key)]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._key(key) in self._codes

    def codes(self) -> Iterator[Code]:
        """Every Tabular node in Tabular order."""
        return iter(self._codes.values())

    def parent(self, key: str) -> Code | None:
        parent = self.code(key).parent
        return None if parent is None else self._codes[parent]

    def children(self, key: str) -> tuple[Code, ...]:
        return tuple(self._codes[child] for child in self._children.get(self._key(key), ()))

    def ancestors(self, key: str) -> tuple[Code, ...]:
        """From the parent up to the chapter."""
        found = []
        parent = self.code(key).parent
        while parent is not None:
            node = self._codes[parent]
            found.append(node)
            parent = node.parent
        return tuple(found)

    def tabular_texts(self) -> Iterator[TabularText]:
        """Every description, inclusion term and `includes` note, in id order (read from disk)."""
        rows = self._conn.execute("SELECT id, node, kind, text FROM tabular_term ORDER BY id")
        return map(TabularText._make, rows)

    def seven_char_def(self, key: str) -> tuple[tuple[str, str], ...]:
        """The 7th characters and their meanings that apply to `key`, or () if none do."""
        owner = self.code(key).seventh_from
        return self._seventh.get(owner, ()) if owner else ()

    def notes(
        self, key: str, kind: str | None = None, *, inherited: bool = False
    ) -> tuple[Note, ...]:
        """The notes of `key` in document order; with `inherited`, then those of its ancestors."""
        return tuple(
            note
            for node in self._lineage(key, inherited)
            for note in self._notes.get(node, ())
            if kind is None or note.kind == kind
        )

    def edges(
        self, key: str, kind: str | None = None, *, inherited: bool = False
    ) -> tuple[Edge, ...]:
        """The code references in the notes of `key`; with `inherited`, then its ancestors'."""
        return tuple(
            edge
            for node in self._lineage(key, inherited)
            for edge in self._edges.get(node, ())
            if kind is None or edge.kind == kind
        )

    def edges_to(self, key: str) -> tuple[Edge, ...]:
        """Notes elsewhere that name `key` as a single code."""
        return self._edges_to.get(self._key(key), ())

    # -- Alphabetic Index ---------------------------------------------------------------------

    def term(self, term_id: int) -> Term:
        return self._terms[term_id]

    def term_at(self, path: str, book: str = "index") -> Term | None:
        """The term at `path` (`Pneumonia > aspiration`, case-insensitive) in `book`."""
        parts = [part.strip() for part in path.split(">")]
        found = self._paths.get((book, PATH_SEPARATOR.join(parts).casefold()))
        return None if found is None else self._terms[found]

    @cached_property
    def _paths(self) -> dict[tuple[str, str], int]:
        paths: dict[tuple[str, str], int] = {}
        for term in self._terms.values():
            paths.setdefault((term.book, term.path.casefold()), term.id)
        return paths

    def terms(self, book: str | None = None) -> Iterator[Term]:
        """Every Index term (of one book), in id order: each main term before its subterms."""
        return (term for term in self._terms.values() if book is None or term.book == book)

    def main_terms(self, book: str = "index") -> tuple[Term, ...]:
        return tuple(self._terms[term_id] for term_id in self._main.get(book, ()))

    def subterms(self, term_id: int) -> tuple[Term, ...]:
        return tuple(self._terms[child] for child in self._subterms.get(term_id, ()))

    def term_parent(self, term_id: int) -> Term | None:
        parent = self._terms[term_id].parent
        return None if parent is None else self._terms[parent]

    def pointers(self, term_id: int) -> tuple[Pointer, ...]:
        """The codes a term gives: `code`, `manifestation`, `subcategory`, `see_category`, cells."""
        return self._pointers.get(term_id, ())

    def pointers_to(self, key: str) -> tuple[Pointer, ...]:
        """The Index pointers that resolve to `key`."""
        return self._pointers_to.get(self._key(key), ())

    def xrefs(self, term_id: int) -> tuple[CrossRef, ...]:
        """The `see` and `see also` of a term."""
        return self._xrefs.get(term_id, ())

    def xrefs_to(self, term_id: int) -> tuple[CrossRef, ...]:
        """The `see` and `see also` that lead to a term, for the graph explorer (task 9.11)."""
        return self._xrefs_to.get(term_id, ())

    def column_heading(self, book: str, column: int) -> str | None:
        """The heading of a Neoplasm Table or Table of Drugs column: `Malignant Primary`."""
        return self._columns.get((book, column))

    # -- Search -------------------------------------------------------------------------------

    def search_index(
        self, query: str, k: int = 10, *, book: str | None = None, main_terms: bool = False
    ) -> list[IndexHit]:
        return search_index(self._conn, query, k, book=book, main_terms=main_terms)

    def search_tabular(self, query: str, k: int = 10) -> list[TabularHit]:
        return search_tabular(self._conn, query, k)

    # -- Lifecycle ----------------------------------------------------------------------------

    @property
    def _conn(self) -> sqlite3.Connection:
        """This thread's read-only connection (task 9.10). SQLite connections are not shared
        between threads, so each thread that reads the file (search, `tabular_texts`) opens its
        own on first use, and searches in different threads run in parallel. A forked process
        opens new ones."""
        if self._closed:
            raise sqlite3.ProgrammingError(f"the graph {self.path} is closed")
        pid = os.getpid()
        held: tuple[int, sqlite3.Connection] | None = getattr(self._local, "held", None)
        if held is None or held[0] != pid:
            # check_same_thread=False only so that `close` may close it from another thread.
            conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
            held = (pid, conn)
            with self._opening:
                self._opened.append(held)
            self._local.held = held
        return held[1]

    def close(self) -> None:
        """Close the connections this process opened, in every thread."""
        pid = os.getpid()
        with self._opening:
            self._closed = True
            mine = [conn for owner, conn in self._opened if owner == pid]
            self._opened = [held for held in self._opened if held[0] != pid]
        for conn in mine:
            conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Graph({self.release!r}, {len(self._codes):,} codes, {len(self._terms):,} terms)"

    # -- Loading ------------------------------------------------------------------------------

    def _key(self, key: str) -> str:
        bare = key.strip()
        return dot(bare) if is_code(bare) else bare

    def _lineage(self, key: str, inherited: bool) -> list[str]:
        node = self.code(key)
        lineage = [node.key]
        if inherited:
            lineage += [ancestor.key for ancestor in self.ancestors(node.key)]
        return lineage

    def _load_tabular(self) -> None:
        self._codes: dict[str, Code] = {}
        self._children: dict[str, list[str]] = {}
        for (
            key,
            kind,
            parent,
            depth,
            description,
            billable,
            placeholder,
            owner,
        ) in self._conn.execute(
            "SELECT key, kind, parent, depth, description, billable, placeholder, seventh_from "
            "FROM tabular_node ORDER BY ordinal"
        ):
            self._codes[key] = Code(
                key, kind, parent, depth, description, billable == 1, placeholder == 1, owner
            )
            if parent is not None:
                self._children.setdefault(parent, []).append(key)
        self._seventh = {
            owner: tuple((char, meaning) for _, char, meaning in rows)
            for owner, rows in groupby(
                self._conn.execute(
                    "SELECT owner, char, meaning FROM seventh_char ORDER BY owner, ordinal"
                ),
                key=itemgetter(0),
            )
        }
        # Note ids follow document order, so the notes of a node are consecutive.
        notes = self._conn.execute("SELECT node, kind, text, id FROM tabular_note ORDER BY id")
        texts: dict[int, str] = {}
        self._notes: dict[str, tuple[Note, ...]] = {}
        for node, rows in groupby(notes, key=itemgetter(0)):
            found = []
            for _, kind, text, note_id in rows:
                found.append(Note(node, kind, text, note_id))
                texts[note_id] = text
            self._notes[node] = tuple(found)
        edges = self._conn.execute(
            "SELECT source, kind, written, range_start, range_end, target, status, note "
            "FROM tabular_edge ORDER BY source, note, ordinal"
        )
        self._edges = {
            source: tuple(
                Edge(source, kind, written, start, end, target, status, texts[note_id], note_id)
                for _, kind, written, start, end, target, status, note_id in rows
            )
            for source, rows in groupby(edges, key=itemgetter(0))
        }

    def _load_index(self) -> None:
        self._terms: dict[int, Term] = {}
        self._subterms: dict[int, list[int]] = {}
        self._main: dict[str, list[int]] = {}
        for term_id, book, parent, level, title, printed, nonessential, path in self._conn.execute(
            "SELECT id, book, parent, level, title, printed_title, nonessential, path "
            "FROM index_term ORDER BY id"
        ):
            modifiers = () if nonessential == "[]" else tuple(json.loads(nonessential))
            self._terms[term_id] = Term(
                term_id, book, parent, level, title, printed, modifiers, path
            )
            if parent is None:
                self._main.setdefault(book, []).append(term_id)
            else:
                self._subterms.setdefault(parent, []).append(term_id)
        pointers = self._conn.execute(
            "SELECT term, field, col, written, code, partial, target, status FROM index_pointer "
            "ORDER BY term, field, col"
        )
        self._pointers = {
            term: tuple(
                Pointer(term, field, col, written, code, partial == 1, target, status)
                for _, field, col, written, code, partial, target, status in rows
            )
            for term, rows in groupby(pointers, key=itemgetter(0))
        }
        xrefs = self._conn.execute(
            "SELECT term, kind, written, status, book, target, col, instruction, remainder, reason "
            "FROM index_xref ORDER BY term, kind"
        )
        self._xrefs = {
            term: tuple(map(CrossRef._make, rows))
            for term, rows in groupby(xrefs, key=itemgetter(0))
        }
        self._columns: dict[tuple[str, int], str] = {
            (book, col): heading
            for book, col, heading in self._conn.execute(
                "SELECT book, col, heading FROM table_column"
            )
        }

    # Reverse lookups, built from memory on first use.

    @cached_property
    def _edges_to(self) -> dict[str, tuple[Edge, ...]]:
        found: dict[str, list[Edge]] = {}
        for edges in self._edges.values():
            for edge in edges:
                if edge.target is not None:
                    found.setdefault(edge.target, []).append(edge)
        return {target: tuple(edges) for target, edges in found.items()}

    @cached_property
    def _pointers_to(self) -> dict[str, tuple[Pointer, ...]]:
        found: dict[str, list[Pointer]] = {}
        for pointers in self._pointers.values():
            for pointer in pointers:
                if pointer.target is not None:
                    found.setdefault(pointer.target, []).append(pointer)
        return {target: tuple(pointers) for target, pointers in found.items()}

    @cached_property
    def _xrefs_to(self) -> dict[int, tuple[CrossRef, ...]]:
        found: dict[int, list[CrossRef]] = {}
        for xrefs in self._xrefs.values():
            for xref in xrefs:
                if xref.target is not None:
                    found.setdefault(xref.target, []).append(xref)
        return {target: tuple(xrefs) for target, xrefs in found.items()}


def load(
    release: str | None = None, *, path: Path | None = None, settings: Settings | None = None
) -> Graph:
    """Load the graph of `release` (default: TEXT2ICD_RELEASE) from the data directory, or the
    file at `path`."""
    return Graph(graph_file(release, path=path, settings=settings))


def read_meta(path: Path) -> dict[str, str]:
    """The `meta` table of a graph file (release, hashes, versions), without loading the graph."""
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            return dict(conn.execute("SELECT key, value FROM meta"))
    except sqlite3.DatabaseError as exc:
        raise GraphError(f"{path} is not a readable graph file: {exc}") from exc


def graph_file(
    release: str | None = None, *, path: Path | None = None, settings: Settings | None = None
) -> Path:
    """The graph file of `release` in the data directory, or `path`; GraphError if it is missing."""
    if path is None:
        settings = settings or Settings()
        release = load_registry().get(release or settings.release).id
        path = settings.graph_path(release)
    if not path.is_file():
        hint = f" -r {release}" if release else ""
        message = f"{path} does not exist; run `text2icd setup{hint}`"
        here = Path("data") / "processed" / path.name  # a checkout's graph, without its .env
        if here.is_file() and here.resolve() != path.resolve():
            message += f", or use {here}: set TEXT2ICD_DATA_DIR=data or pass --data-dir data"
        raise GraphError(message)
    return path
