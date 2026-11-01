import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.graph import build
from text2icd.graph.build import SCHEMA_VERSION, BuildReport, build_graph, content_hash
from text2icd.ontology.files import ReleaseFiles


def rows(conn: sqlite3.Connection, sql: str, *params: object) -> list[tuple[object, ...]]:
    return conn.execute(sql, params).fetchall()


def node(conn: sqlite3.Connection, key: str) -> dict[str, object]:
    cursor = conn.execute("SELECT * FROM tabular_node WHERE key = ?", (key,))
    names = [column[0] for column in cursor.description]
    return dict(zip(names, cursor.fetchone(), strict=True))


def test_fixture_build_counts(fixture_build: BuildReport) -> None:
    assert fixture_build.release == "fy2099"
    assert fixture_build.path.is_file()
    assert fixture_build.rows["tabular_node"] == 2 + 2 + 15 + 20
    assert fixture_build.rows["order_code"] == 10
    assert fixture_build.rows["seventh_char"] == 4 + 3
    assert fixture_build.rows["table_column"] == 7 + 7
    assert "content sha256" in "\n".join(fixture_build.lines())


def test_kinds_and_billable_flags(fixture_db: sqlite3.Connection) -> None:
    kinds = dict(rows(fixture_db, "SELECT kind, count(*) FROM tabular_node GROUP BY kind"))
    assert kinds == {"chapter": 2, "section": 2, "diag": 15, "generated": 20}
    billable = {key for (key,) in rows(fixture_db, "SELECT key FROM tabular_node WHERE billable")}
    assert len(billable) == 21
    assert "E11.22" in billable
    assert "T51.3X1A" in billable
    assert "T51.3X1" not in billable
    assert "E08.321" not in billable


def test_generated_code_from_the_order_file(fixture_db: sqlite3.Connection) -> None:
    assert node(fixture_db, "T51.3X1A") == {
        "key": "T51.3X1A",
        "kind": "generated",
        "parent": "T51.3X1",
        "ordinal": node(fixture_db, "T51.3X1")["ordinal"] + 1,  # type: ignore[operator]
        "depth": 6,
        "description": "Toxic effect of fusel oil, accidental (unintentional), initial encounter",
        "billable": 1,
        "placeholder": 0,
        "seventh_from": "T51",
        "seventh": "A",
    }


def test_generated_code_missing_from_the_order_file(fixture_db: sqlite3.Connection) -> None:
    row = node(fixture_db, "E08.3211")
    assert row["description"] == (
        "Diabetes mellitus due to underlying condition with mild nonproliferative diabetic "
        "retinopathy with macular edema, right eye"
    )
    assert row["seventh_from"] == "E08.32"
    assert row["seventh"] == "1"


def test_tabular_order_and_depth(fixture_db: sqlite3.Connection) -> None:
    keys = [key for (key,) in rows(fixture_db, "SELECT key FROM tabular_node ORDER BY ordinal")]
    assert keys[:3] == ["4", "E08-E13", "E08"]
    assert keys.index("E08.321") < keys.index("E08.3211") < keys.index("E08.329")
    assert keys[keys.index("T51.3X1") + 1 : keys.index("T51.3X1") + 4] == [
        "T51.3X1A",
        "T51.3X1D",
        "T51.3X1S",
    ]
    depths = dict(rows(fixture_db, "SELECT key, depth FROM tabular_node"))
    assert (depths["4"], depths["E08-E13"], depths["E08"], depths["E08.32"]) == (0, 1, 2, 4)
    assert node(fixture_db, "E08.321")["seventh_from"] == "E08.32"
    assert node(fixture_db, "E11.22")["seventh_from"] is None
    assert node(fixture_db, "T51.3X")["placeholder"] == 1


def test_notes_and_seventh_characters(fixture_db: sqlite3.Connection) -> None:
    notes = rows(
        fixture_db, "SELECT kind, text FROM tabular_note WHERE node = 'E11.22' ORDER BY ordinal"
    )
    assert (
        "useAdditionalCode",
        "code to identify stage of chronic kidney disease (N18.1-N18.6)",
    ) in notes
    sevenths = rows(
        fixture_db, "SELECT char, meaning FROM seventh_char WHERE owner = 'T51' ORDER BY ordinal"
    )
    assert sevenths == [
        ("A", "initial encounter"),
        ("D", "subsequent encounter"),
        ("S", "sequela"),
    ]


def test_index_terms(fixture_db: sqlite3.Connection, fixture_release: ReleaseFiles) -> None:
    (food,) = rows(
        fixture_db,
        "SELECT id, book, parent, level, title, printed_title, nonessential FROM index_term "
        "WHERE path = 'Pneumonia > aspiration > due to > food'",
    )
    term_id, book, parent, level, title, printed, nonessential = food
    assert (book, level, title, printed) == ("index", 3, "food", "food (regurgitated)")
    assert json.loads(str(nonessential)) == ["regurgitated"]
    assert rows(fixture_db, "SELECT path FROM index_term WHERE id = ?", parent) == [
        ("Pneumonia > aspiration > due to",)
    ]

    books = dict(rows(fixture_db, "SELECT book, min(id) FROM index_term GROUP BY book"))
    sizes = dict(rows(fixture_db, "SELECT book, count(*) FROM index_term GROUP BY book"))
    assert books["index"] == 0
    assert books["eindex"] == sizes["index"]
    assert books["neoplasm"] == sizes["index"] + sizes["eindex"]
    assert term_id < books["eindex"]

    main_terms = rows(
        fixture_db,
        "SELECT ordinal, title FROM index_term WHERE book = 'index' AND level = 0 ORDER BY id",
    )
    assert [ordinal for ordinal, _ in main_terms] == list(range(len(main_terms)))
    assert ("Pneumonia" in {title for _, title in main_terms}) is True


def test_table_columns(fixture_db: sqlite3.Connection) -> None:
    columns = rows(fixture_db, "SELECT col, heading FROM table_column WHERE book = 'drug'")
    assert columns[0] == (1, "Substance")
    assert columns[-1] == (7, "Underdosing")
    neoplasm = dict(
        rows(fixture_db, "SELECT col, heading FROM table_column WHERE book = 'neoplasm'")
    )
    assert neoplasm[2] == "Malignant Primary"


def test_meta(fixture_db: sqlite3.Connection, fixture_release: ReleaseFiles) -> None:
    meta = dict(rows(fixture_db, "SELECT key, value FROM meta"))
    assert meta["schema_version"] == str(SCHEMA_VERSION)
    assert meta["release"] == "fy2099"
    assert meta["tabular_version"] == "2027"
    assert meta["content_sha256"] == content_hash(fixture_db)
    assert meta["fts_tokenizer"] == "porter"
    assert meta["sha256:icd10cm_tabular_2099.xml"] == build.sha256_of(fixture_release.tabular)
    assert {key for key in meta if key.startswith("sha256:")} == {
        f"sha256:icd10cm_{name}_2099.{'txt' if name == 'order' else 'xml'}"
        for name in ("tabular", "index", "eindex", "neoplasm", "drug", "order")
    }


def test_build_is_deterministic(fixture_release: ReleaseFiles, tmp_path: Path) -> None:
    first = build_graph(fixture_release, "fy2099", tmp_path / "a.sqlite")
    second = build_graph(fixture_release, "fy2099", tmp_path / "b.sqlite")
    assert first.content_sha256 == second.content_sha256
    assert first.rows == second.rows


def test_content_hash_sees_every_row(fixture_db: sqlite3.Connection) -> None:
    before = content_hash(fixture_db)
    fixture_db.execute("UPDATE index_term SET title = 'Pneumonitis' WHERE title = 'Pneumonia'")
    assert content_hash(fixture_db) != before
    fixture_db.execute("UPDATE meta SET value = 'changed' WHERE key = 'builder_version'")
    fixture_db.execute("UPDATE index_term SET title = 'Pneumonia' WHERE title = 'Pneumonitis'")
    assert content_hash(fixture_db) == before


def test_failed_build_keeps_the_previous_file(
    fixture_release: ReleaseFiles, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out" / "graph.sqlite"
    output.parent.mkdir()
    output.write_bytes(b"previous")

    def broken(conn: sqlite3.Connection, sources: build.Sources, tokenizer: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(build, "write_graph", broken)
    with pytest.raises(RuntimeError, match="boom"):
        build_graph(fixture_release, "fy2099", output)
    assert output.read_bytes() == b"previous"
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.slow
@pytest.mark.parametrize("release_id", ["fy2026", "fy2026-apr", "fy2027"])
def test_release_build(release_graph: Callable[[str], BuildReport], release_id: str) -> None:
    report = release_graph(release_id)
    assert report.seconds < 60
    with sqlite3.connect(report.path) as conn:
        mismatched = rows(
            conn,
            "SELECT code FROM order_code o WHERE billable = 1 AND NOT EXISTS "
            "(SELECT 1 FROM tabular_node t WHERE t.key = o.code AND t.billable = 1) "
            "UNION ALL SELECT key FROM tabular_node t WHERE billable = 1 AND NOT EXISTS "
            "(SELECT 1 FROM order_code o WHERE o.code = t.key AND o.billable = 1)",
        )
        assert mismatched == []
        renamed = rows(
            conn,
            "SELECT key FROM tabular_node t JOIN order_code o ON o.code = t.key "
            "WHERE kind = 'generated' AND t.description != o.long_description",
        )
        assert renamed == []
        assert rows(conn, "SELECT count(*) FROM index_term WHERE path LIKE '%>  >%'") == [(0,)]
