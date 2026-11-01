import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from text2icd.graph.build import BuildReport, build_graph
from text2icd.graph.search import match_expression, search_index, search_tabular
from text2icd.ontology.files import ReleaseFiles


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Pneumonia, aspiration", '"pneumonia" OR "aspiration"'),
        ("aspiration ASPIRATION", '"aspiration"'),
        ('say "hi"', '"say" OR "hi"'),
        # FTS5 syntax in the input is only words.
        ("a AND (NEAR b*", '"a" OR "and" OR "near" OR "b"'),
        ("Sjögren", '"sjögren"'),
        (" ,. ", None),
    ],
)
def test_match_expression(query: str, expected: str | None) -> None:
    assert match_expression(query) == expected


def paths(
    conn: sqlite3.Connection,
    query: str,
    k: int = 3,
    *,
    book: str | None = None,
    main_terms: bool = False,
) -> list[str]:
    hits = search_index(conn, query, k, book=book, main_terms=main_terms)
    return [
        conn.execute("SELECT path FROM index_term WHERE id = ?", (hit.term,)).fetchone()[0]
        for hit in hits
    ]


def test_search_index(fixture_db: sqlite3.Connection) -> None:
    assert paths(fixture_db, "aspiration pneumonia", 1) == ["Pneumonia > aspiration"]
    assert paths(fixture_db, "pneumonia", 3, main_terms=True) == ["Pneumonia"]
    assert paths(fixture_db, "propanol", book="drug") == ["1-propanol", "2-propanol"]
    assert paths(fixture_db, "propanol", book="index") == []
    assert search_index(fixture_db, "?!") == []
    scores = [hit.score for hit in search_index(fixture_db, "pneumonia", 10)]
    assert scores == sorted(scores)


def test_search_tabular(fixture_db: sqlite3.Connection) -> None:
    (hit,) = search_tabular(fixture_db, "propanol", 5)
    assert (hit.node, hit.kind) == ("T51.3", "inclusion")
    nodes = [hit.node for hit in search_tabular(fixture_db, "diabetic chronic kidney disease", 2)]
    assert nodes[0] == "E11.22"
    assert search_tabular(fixture_db, "") == []


def test_tabular_terms(fixture_db: sqlite3.Connection) -> None:
    kinds = dict(fixture_db.execute("SELECT kind, count(*) FROM tabular_term GROUP BY kind"))
    diag = fixture_db.execute("SELECT count(*) FROM tabular_node WHERE kind = 'diag'").fetchone()
    assert kinds["description"] == diag[0]
    assert set(kinds) == {"description", "inclusion", "includes"}
    e11 = fixture_db.execute(
        "SELECT kind, text FROM tabular_term WHERE node = 'E11' ORDER BY id"
    ).fetchall()
    assert e11[0] == ("description", "Type 2 diabetes mellitus")
    assert ("includes", "diabetes NOS") in e11


def test_search_tables_are_complete(fixture_build: BuildReport) -> None:
    with sqlite3.connect(fixture_build.path) as conn:
        counts = {
            table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("fts_index_term", "fts_tabular", "index_term", "tabular_term")
        }
    assert counts["fts_index_term"] == counts["index_term"]
    assert counts["fts_tabular"] == counts["tabular_term"]
    assert "search (porter tokenizer): 64 Index terms, 23 Tabular texts" in fixture_build.details


@pytest.fixture
def unicode61_db(fixture_release: ReleaseFiles, tmp_path: Path) -> Iterator[sqlite3.Connection]:
    report = build_graph(fixture_release, "fy2099", tmp_path / "g.sqlite", tokenizer="unicode61")
    conn = sqlite3.connect(report.path)
    yield conn
    conn.close()


def test_tokenizers(fixture_db: sqlite3.Connection, unicode61_db: sqlite3.Connection) -> None:
    # Porter stems a plural to the Index title; unicode61 matches exact words only.
    assert paths(fixture_db, "pneumonias", 1) == ["Pneumonia"]
    assert paths(unicode61_db, "pneumonias", 1) == []
    meta = dict(unicode61_db.execute("SELECT key, value FROM meta"))
    assert meta["fts_tokenizer"] == "unicode61"


def test_tokenizer_does_not_change_the_content_hash(
    fixture_build: BuildReport, unicode61_db: sqlite3.Connection
) -> None:
    meta = dict(unicode61_db.execute("SELECT key, value FROM meta"))
    assert meta["content_sha256"] == fixture_build.content_sha256


@pytest.mark.slow
def test_release_search(release_graph: Callable[[str], BuildReport]) -> None:
    report = release_graph("fy2027")
    with sqlite3.connect(report.path) as conn:
        assert paths(conn, "pneumonia aspiration", 1) == ["Pneumonia > aspiration"]
        assert paths(conn, "cirrhosis", 1, main_terms=True) == ["Cirrhosis, cirrhotic"]
        nodes = [hit.node for hit in search_tabular(conn, "type 2 diabetes chronic kidney", 1)]
        assert nodes == ["E11.22"]
        assert conn.execute("SELECT count(*) FROM tabular_term").fetchone()[0] > 60_000
