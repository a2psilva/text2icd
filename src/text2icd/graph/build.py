"""Build the knowledge-graph SQLite file of one release from its CMS files.

The builder copies what the parsers read into the tables of `schema.sql`, adds what can be derived
without interpretation (the codes generated from 7th-character definitions, full Index paths), and
records the source file hashes and a content hash in `meta`. It never fails on a data problem it
can record: the integrity checks decide whether a graph is usable.
"""

import hashlib
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Container, Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from text2icd import __version__
from text2icd.fetch import sha256_of
from text2icd.graph.notes import CodeIndex, find_references
from text2icd.graph.pointers import Pointer, index_pointers
from text2icd.ontology.billable import BillableCode, iter_billable
from text2icd.ontology.codes import CODE_LENGTH_BEFORE_SEVENTH, dot, undot
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.index import IndexBook, IndexSource, parse_index
from text2icd.ontology.order import OrderEntry, parse_order_file
from text2icd.ontology.tabular import Tabular, parse_tabular
from text2icd.ontology.validate import KnownException, load_exceptions

SCHEMA_VERSION = 1
BOOKS: tuple[IndexSource, ...] = ("index", "eindex", "neoplasm", "drug")
PATH_SEPARATOR = " > "
# Tables left out of the content hash: `meta` stores the hash, and FTS5 tables (named fts_*) are
# derived from the hashed tables.
_UNHASHED_PREFIXES = ("meta", "fts_", "sqlite_")

Row = tuple[object, ...]
# Note kinds whose references are the edges of the graph, reported in the build log.
EDGE_NOTE_KINDS = ("excludes1", "excludes2", "codeFirst", "useAdditionalCode", "codeAlso")


@dataclass(frozen=True, slots=True)
class Sources:
    """Everything the builder reads for one release, parsed."""

    release: str
    tabular: Tabular
    books: dict[IndexSource, IndexBook]
    order: tuple[OrderEntry, ...]
    billable: tuple[BillableCode, ...]
    file_hashes: dict[str, str]
    exceptions: tuple[KnownException, ...] = ()


@dataclass(frozen=True, slots=True)
class BuildReport:
    release: str
    path: Path
    rows: dict[str, int]
    content_sha256: str
    seconds: float
    details: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        width = max(len(name) for name in self.rows)
        out = [f"{self.release}: {self.path}  (built in {self.seconds:.1f} s)"]
        out += [f"  {name:<{width}}  {count:>8,}" for name, count in self.rows.items()]
        out.append(f"  content sha256  {self.content_sha256}")
        out += [f"  {detail}" for detail in self.details]
        out += [f"  warning: {warning}" for warning in self.warnings]
        return out


@dataclass(slots=True)
class BuildLog:
    """What the build reports besides row counts: facts about the data, and warnings."""

    details: list[str]
    warnings: list[str]


def load_sources(files: ReleaseFiles, release_id: str, exceptions: Path | None = None) -> Sources:
    """Parse a release. `exceptions` replaces the packaged `index_exceptions.toml`."""
    paths = {
        "tabular": files.tabular,
        "index": files.index,
        "eindex": files.eindex,
        "neoplasm": files.neoplasm,
        "drug": files.drug,
        "order": files.order,
    }
    tabular = parse_tabular(paths["tabular"])
    return Sources(
        release=release_id,
        tabular=tabular,
        books={book: parse_index(paths[book], book) for book in BOOKS},
        order=parse_order_file(paths["order"]),
        billable=tuple(iter_billable(tabular)),
        file_hashes={path.name: sha256_of(path) for path in paths.values()},
        exceptions=tuple(e for e in load_exceptions(exceptions) if release_id in e.releases),
    )


def build_graph(
    files: ReleaseFiles, release_id: str, output: Path, *, exceptions: Path | None = None
) -> BuildReport:
    """Write the graph of one release to `output`, replacing it only once the build succeeds."""
    start = time.perf_counter()
    sources = load_sources(files, release_id, exceptions)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(partial)) as conn:
            conn.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(schema_sql())
            with conn:
                log = write_graph(conn, sources)
                digest = content_hash(conn)
                _write_meta(conn, sources, digest)
            rows = row_counts(conn)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(output)
    seconds = time.perf_counter() - start
    return BuildReport(
        release_id, output, rows, digest, seconds, tuple(log.details), tuple(log.warnings)
    )


def write_graph(conn: sqlite3.Connection, sources: Sources) -> BuildLog:
    """Insert every row and return what the build log should show."""
    log = BuildLog([], [])
    _insert(conn, "order_code", _order_rows(sources.order))
    _insert(conn, "tabular_node", _tabular_rows(sources))
    notes = list(_note_rows(sources.tabular))
    _insert(conn, "tabular_note", notes)
    _insert(conn, "seventh_char", _seventh_rows(sources.tabular))
    _insert(conn, "index_term", _index_rows(sources.books))
    _insert(conn, "table_column", _column_rows(sources.books))

    nodes = set(sources.tabular.nodes) | {entry.code for entry in sources.billable}
    codes = CodeIndex(key for key, node in sources.tabular.nodes.items() if node.kind == "diag")
    _insert(conn, "tabular_edge", _edge_rows(notes, codes, nodes, log))
    pointers = index_pointers(sources.books, term_offsets(sources.books), nodes, sources.exceptions)
    _insert(conn, "index_pointer", map(_pointer_row, pointers.pointers))
    log.warnings.extend(pointers.warnings)
    return log


def schema_sql() -> str:
    return resources.files("text2icd.graph").joinpath("schema.sql").read_text(encoding="utf-8")


def term_offsets(books: dict[IndexSource, IndexBook]) -> dict[IndexSource, int]:
    """Global id of the first term of each book: ids run through the books in `BOOKS` order."""
    offsets: dict[IndexSource, int] = {}
    total = 0
    for book in BOOKS:
        offsets[book] = total
        total += len(books[book])
    return offsets


def content_hash(conn: sqlite3.Connection) -> str:
    """SHA-256 over every hashed table: its name, its columns, then its rows in sorted order."""
    digest = hashlib.sha256()
    for table in hashed_tables(conn):
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        digest.update(f"{table}\t{','.join(columns)}\n".encode())
        order = ", ".join(str(i) for i in range(1, len(columns) + 1))
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def hashed_tables(conn: sqlite3.Connection) -> list[str]:
    names = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%' "
        "ORDER BY name"
    )
    return [name for (name,) in names if not name.startswith(_UNHASHED_PREFIXES)]


def row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in hashed_tables(conn)
    }


def _insert(conn: sqlite3.Connection, table: str, rows: Iterable[Row]) -> None:
    width = len(conn.execute(f"PRAGMA table_info({table})").fetchall())
    conn.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * width)})", rows)


def _order_rows(order: tuple[OrderEntry, ...]) -> Iterator[Row]:
    for entry in order:
        yield (
            entry.code,
            entry.order,
            int(entry.billable),
            entry.short_description,
            entry.long_description,
        )


def _tabular_rows(sources: Sources) -> Iterator[Row]:
    tabular = sources.tabular
    leaves_as_is = {entry.base for entry in sources.billable if entry.seventh is None}
    generated: dict[str, list[BillableCode]] = {}
    for entry in sources.billable:
        if entry.seventh is not None:
            generated.setdefault(entry.base, []).append(entry)
    official = {entry.code: entry.long_description for entry in sources.order}

    depth: dict[str, int] = {}
    ordinal = 0
    for node in tabular.nodes.values():
        depth[node.key] = 0 if node.parent is None else depth[node.parent] + 1
        owner = tabular.seventh_character_owner(node.key) if node.kind == "diag" else None
        yield (
            node.key,
            node.kind,
            node.parent,
            ordinal,
            depth[node.key],
            node.description,
            int(node.key in leaves_as_is),
            int(node.placeholder),
            owner,
            None,
        )
        ordinal += 1
        for entry in generated.get(node.key, ()):
            assert entry.owner is not None
            assert entry.seventh is not None
            meaning = dict(tabular[entry.owner].seventh_characters)[entry.seventh]
            padded = len(undot(entry.base)) < CODE_LENGTH_BEFORE_SEVENTH
            yield (
                entry.code,
                "generated",
                node.key,
                ordinal,
                depth[node.key] + 1,
                official.get(entry.code) or f"{node.description}, {meaning}",
                1,
                int(padded),
                entry.owner,
                entry.seventh,
            )
            ordinal += 1


def _pointer_row(pointer: Pointer) -> Row:
    return (
        pointer.term,
        pointer.field,
        pointer.column,
        pointer.written,
        pointer.code,
        int(pointer.partial),
        pointer.target,
        pointer.status,
    )


def _edge_rows(
    notes: list[Row], codes: CodeIndex, nodes: Container[str], log: BuildLog
) -> Iterator[Row]:
    """One row per code reference in a note, in the order written.

    Words that look like codes are counted in the build log, not kept, and so is a note naming its
    own code ("to be assigned to codes in subcategory E08.32"), which is not an edge.
    """
    words: Counter[str] = Counter()
    notes_of: Counter[str] = Counter()
    notes_with: Counter[str] = Counter()
    count = 0
    for note_id, node, kind, _, text in notes:
        ordinal = 0
        for found in find_references(str(text)):
            resolves = codes.any_in(found.reference)
            if not resolves and found.is_bare:
                words[found.written] += 1
                continue
            start, end = dot(found.reference.start), dot(found.reference.end)
            target = start if start == end and start in nodes else None
            if target == node:
                continue
            if not resolves:
                log.warnings.append(f"no code for {found.written!r} in a {kind} note of {node}")
            status = "ok" if resolves else "unresolved"
            yield note_id, ordinal, node, kind, found.written, start, end, target, status
            ordinal += 1
        count += ordinal
        notes_of[str(kind)] += 1
        notes_with[str(kind)] += ordinal > 0
    shares = ", ".join(
        f"{kind} {notes_with[kind] / notes_of[kind]:.0%}"
        for kind in EDGE_NOTE_KINDS
        if notes_of[kind]
    )
    log.details.append(f"note references: {count:,}; notes with one: {shares}")
    if words:
        listed = ", ".join(f"{word} ({n})" for word, n in sorted(words.items()))
        log.details.append(f"not taken as codes: {listed}")


def _note_rows(tabular: Tabular) -> Iterator[Row]:
    note_id = 0
    for node in tabular.nodes.values():
        for ordinal, note in enumerate(node.notes):
            note_id += 1
            yield (note_id, node.key, note.kind, ordinal, note.text)


def _seventh_rows(tabular: Tabular) -> Iterator[Row]:
    for node in tabular.nodes.values():
        for ordinal, (char, meaning) in enumerate(node.seventh_characters):
            yield (node.key, char, ordinal, meaning)


def _index_rows(books: dict[IndexSource, IndexBook]) -> Iterator[Row]:
    offsets = term_offsets(books)
    for source in BOOKS:
        book, offset = books[source], offsets[source]
        ordinals = {term_id: i for i, term_id in enumerate(book.main_terms)}
        paths = book.paths(PATH_SEPARATOR)
        for term in book.terms:
            ordinals.update((child, i) for i, child in enumerate(term.children))
            yield (
                offset + term.id,
                source,
                None if term.parent is None else offset + term.parent,
                ordinals[term.id],
                term.level,
                term.letter,
                term.title,
                term.display_title,
                json.dumps(list(term.nonessential), ensure_ascii=False),
                paths[term.id],
            )


def _column_rows(books: dict[IndexSource, IndexBook]) -> Iterator[Row]:
    for source in ("neoplasm", "drug"):
        for col, heading in sorted(books[source].columns.items()):
            yield (source, col, heading)


def _write_meta(conn: sqlite3.Connection, sources: Sources, digest: str) -> None:
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "release": sources.release,
        "builder_version": __version__,
        "tabular_version": sources.tabular.version,
        "content_sha256": digest,
    }
    meta.update({f"sha256:{name}": value for name, value in sorted(sources.file_hashes.items())})
    conn.executemany("INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
