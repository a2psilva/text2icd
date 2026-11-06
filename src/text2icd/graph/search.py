"""Full-text search over the Index and the Tabular List: SQLite FTS5 with BM25 ranking.

Two contentless FTS5 tables are filled when the graph is built:

- `fts_index_term`, one row per Index term (rowid = `index_term.id`): the essential title, the
  nonessential modifiers, and the full path, weighted in that order;
- `fts_tabular`, one row per `tabular_term` (rowid = `tabular_term.id`): code descriptions and
  inclusion terms.

The tokenizer is chosen at build time and recorded in `meta.fts_tokenizer`: `porter` (stems
"fractures" to "fractur") or `unicode61` (exact words). Both ignore case and diacritics.

Queries are free text. Every word is quoted, so FTS5 syntax in the input has no effect, and the
words are OR-ed, so BM25 ranks rows by how many of the words they hold and how rare those are.
"""

import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

Tokenizer = Literal["porter", "unicode61"]
TOKENIZERS: dict[Tokenizer, str] = {
    "porter": "porter unicode61 remove_diacritics 2",
    "unicode61": "unicode61 remove_diacritics 2",
}
DEFAULT_TOKENIZER: Tokenizer = "porter"
SEARCH_TABLES = ("fts_index_term", "fts_tabular")

# BM25 weights of the fts_index_term columns: title, modifiers, path.
INDEX_WEIGHTS = (10.0, 2.0, 1.0)
_WORD = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class IndexHit:
    term: int
    score: float  # BM25: lower is better


@dataclass(frozen=True, slots=True)
class TabularHit:
    node: str
    kind: str
    text: str
    score: float


def create_search_tables(conn: sqlite3.Connection, tokenizer: Tokenizer) -> None:
    spec = TOKENIZERS[tokenizer]
    conn.execute(
        "CREATE VIRTUAL TABLE fts_index_term USING fts5"
        f"(title, modifiers, path, content='', tokenize='{spec}')"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE fts_tabular USING fts5(text, content='', tokenize='{spec}')"
    )


def optimize_search_tables(conn: sqlite3.Connection) -> None:
    """Merge the FTS5 segments into one: smaller and faster to query."""
    for table in SEARCH_TABLES:
        conn.execute(f"INSERT INTO {table} ({table}) VALUES ('optimize')")


def match_expression(query: str) -> str | None:
    """The FTS5 query for free text: `Pneumonia, aspiration` -> `"pneumonia" OR "aspiration"`."""
    words = dict.fromkeys(_WORD.findall(query.casefold()))
    return " OR ".join(f'"{word}"' for word in words) or None


def search_index(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    *,
    book: str | None = None,
    main_terms: bool = False,
) -> list[IndexHit]:
    """The `k` Index terms that best match `query`, optionally within one book or main terms."""
    statement = index_statement(query, k, book=book, main_terms=main_terms)
    if statement is None:
        return []
    return [IndexHit(term, score) for term, score in conn.execute(*statement)]


def index_statement(
    query: str, k: int = 10, *, book: str | None = None, main_terms: bool = False
) -> tuple[str, list[object]] | None:
    """The SQL and parameters of `search_index`; None when the query has no word."""
    expression = match_expression(query)
    if expression is None:
        return None
    weights = ", ".join(str(weight) for weight in INDEX_WEIGHTS)
    filters = ""
    params: list[object] = [expression]
    if book is not None:
        filters += " AND t.book = ?"
        params.append(book)
    if main_terms:
        filters += " AND t.level = 0"
    params.append(k)
    # CROSS JOIN keeps the FTS table first. With a plain JOIN and a book filter, SQLite picks
    # the (book, level) index and runs one MATCH per term of the book: 340 ms instead of 2 ms
    # for "of" in the drug table, with the same rows (task 9.9).
    sql = (
        f"SELECT f.rowid, bm25(fts_index_term, {weights}) AS score "
        "FROM fts_index_term f CROSS JOIN index_term t ON t.id = f.rowid "
        f"WHERE fts_index_term MATCH ?{filters} ORDER BY score, f.rowid LIMIT ?"
    )
    return sql, params


def search_tabular(conn: sqlite3.Connection, query: str, k: int = 10) -> list[TabularHit]:
    """The `k` code descriptions or inclusion terms that best match `query`."""
    expression = match_expression(query)
    if expression is None:
        return []
    rows = conn.execute(
        "SELECT t.node, t.kind, t.text, bm25(fts_tabular) AS score "
        "FROM fts_tabular f CROSS JOIN tabular_term t ON t.id = f.rowid "
        "WHERE fts_tabular MATCH ? ORDER BY score, f.rowid LIMIT ?",
        (expression, k),
    )
    return [TabularHit(node, kind, text, score) for node, kind, text, score in rows]
