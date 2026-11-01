import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.graph.build import BuildReport, build_graph
from text2icd.graph.pointers import Pointer, index_pointers, parse_pointer
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.index import IndexBook, parse_index
from text2icd.ontology.validate import IndexReference, KnownException, load_exceptions

BOOK = """<ICD10CM.index><version>2099</version><title>Index</title><letter><title>O</title>
<mainTerm><title>Osteitis</title><code>M86.9</code>
  <term level="1"><title>pubic ramus</title><manif>pubis] D49.2 [M90.68</manif></term>
  <term level="1"><title>incomplete</title><code>M86.-</code></term>
  <term level="1"><title>unknown</title><code>Z99.99</code></term>
  <term level="1"><title>not a code</title><manif>SCFE</manif></term>
  <term level="1"><title>both</title><code>M86.9</code><manif>garbled</manif></term>
</mainTerm></letter></ICD10CM.index>"""
NODES = {"M86", "M86.9", "D49.2", "M90.68"}


@pytest.fixture
def book(tmp_path: Path) -> IndexBook:
    path = tmp_path / "index.xml"
    path.write_text(BOOK)
    return parse_index(path, "index")


def exception(path: str, field: str, value: str, **correction: str) -> KnownException:
    return KnownException(
        IndexReference("index", path, field, value), frozenset({"fy2099"}), "reason", **correction
    )


def by_path(book: IndexBook, pointers: tuple[Pointer, ...]) -> dict[str, list[Pointer]]:
    paths = book.paths()
    found: dict[str, list[Pointer]] = {}
    for pointer in pointers:
        found.setdefault(paths[pointer.term - 10], []).append(pointer)
    return found


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("E85.4", ("E85.4", False)),
        ("C79.8-", ("C79.8", True)),
        ("E10.-", ("E10", True)),
        (" I43 ", ("I43", False)),
        ("E1122", ("E11.22", False)),
        ("SCFE", (None, False)),
        ("pubis] D49.2 [M90.68", (None, False)),
    ],
)
def test_parse_pointer(value: str, expected: tuple[str | None, bool]) -> None:
    assert parse_pointer(value) == expected


def test_pointers_resolve_to_tabular_nodes(book: IndexBook) -> None:
    result = index_pointers({"index": book}, {"index": 10}, NODES, [])
    found = by_path(book, result.pointers)
    assert found["Osteitis"] == [Pointer(10, "code", 0, "M86.9", "M86.9", False, "M86.9", "ok")]
    assert found["Osteitis > incomplete"] == [
        Pointer(12, "code", 0, "M86.-", "M86", True, "M86", "ok")
    ]
    assert found["Osteitis > unknown"][0].status == "unresolved"
    assert found["Osteitis > unknown"][0].target is None
    assert found["Osteitis > not a code"][0].code is None
    assert result.warnings == ()


def test_exceptions_correct_or_exclude(book: IndexBook) -> None:
    exceptions = [
        exception(
            "Osteitis > pubic ramus",
            "manifestation",
            "pubis] D49.2 [M90.68",
            code="D49.2",
            manifestation="M90.68",
        ),
        exception("Osteitis > not a code", "manifestation", "SCFE"),
        exception("Osteitis > gone", "code", "M86.1"),
    ]
    result = index_pointers({"index": book}, {"index": 10}, NODES, exceptions)
    found = by_path(book, result.pointers)
    written = "pubis] D49.2 [M90.68"
    assert found["Osteitis > pubic ramus"] == [
        Pointer(11, "code", 0, written, "D49.2", False, "D49.2", "corrected"),
        Pointer(11, "manifestation", 0, written, "M90.68", False, "M90.68", "corrected"),
    ]
    assert found["Osteitis > not a code"] == [
        Pointer(14, "manifestation", 0, "SCFE", None, False, None, "excluded")
    ]
    assert len(result.warnings) == 2
    assert result.warnings[0].startswith("excluded index: Osteitis > not a code [manifestation]")
    assert result.warnings[1] == (
        "exception not found in the Index: index: Osteitis > gone [code] 'M86.1'"
    )


def test_correction_must_name_an_existing_code(book: IndexBook) -> None:
    bad = exception("Osteitis > unknown", "code", "Z99.99", code="Z99.98")
    with pytest.raises(ValueError, match=r"corrects Osteitis > unknown to 'Z99.98'"):
        index_pointers({"index": book}, {"index": 0}, NODES, [bad])


def test_correction_cannot_duplicate_a_pointer(book: IndexBook) -> None:
    clash = exception("Osteitis > both", "manifestation", "garbled", code="M86.9")
    with pytest.raises(ValueError, match="duplicates an existing pointer"):
        index_pointers({"index": book}, {"index": 0}, NODES, [clash])


def test_fixture_graph_pointers(fixture_db: sqlite3.Connection) -> None:
    def pointers(path: str) -> list[tuple[object, ...]]:
        return fixture_db.execute(
            "SELECT field, col, written, code, partial, target, status FROM index_pointer "
            "JOIN index_term ON index_term.id = term WHERE path = ? ORDER BY field, col",
            (path,),
        ).fetchall()

    propanol = pointers("1-propanol")
    assert propanol[0] == ("cell", 2, "T51.3X1", "T51.3X1", 0, "T51.3X1", "ok")
    assert [col for _, col, *_ in propanol] == [2, 3, 4, 5]
    assert pointers("Amyloid heart") == [
        ("code", 0, "E85.4", "E85.4", 0, None, "unresolved"),
        ("manifestation", 0, "I43", "I43", 0, None, "unresolved"),
    ]
    secondary = fixture_db.execute(
        "SELECT DISTINCT code, partial FROM index_pointer WHERE written = 'C79.8-'"
    ).fetchall()
    assert secondary == [("C79.8", 1)]


def test_build_reports_exception_warnings(fixture_release: ReleaseFiles, tmp_path: Path) -> None:
    exceptions = tmp_path / "exceptions.toml"
    exceptions.write_text(
        'schema = 1\n[[exception]]\nreleases = ["fy2099"]\nbook = "index"\n'
        'path = "Amyloid heart"\nfield = "manifestation"\nvalue = "I43"\nreason = "test"\n'
    )
    report = build_graph(fixture_release, "fy2099", tmp_path / "g.sqlite", exceptions=exceptions)
    excluded = [w for w in report.warnings if w.startswith("excluded ")]
    assert excluded == ["excluded index: Amyloid heart [manifestation] 'I43': test"]
    assert "  warning: excluded index: Amyloid heart" in "\n".join(report.lines())


def test_fixture_pointers_give_no_warnings(fixture_build: BuildReport) -> None:
    # The fixture's notes name codes it does not contain; those warnings are tested in test_notes.
    assert [w for w in fixture_build.warnings if not w.startswith("no code for ")] == []


@pytest.mark.slow
@pytest.mark.parametrize("release_id", ["fy2026", "fy2026-apr", "fy2027"])
def test_release_pointers(release_graph: Callable[[str], BuildReport], release_id: str) -> None:
    report = release_graph(release_id)
    known = [e for e in load_exceptions() if release_id in e.releases]
    with sqlite3.connect(report.path) as conn:
        statuses = dict(
            conn.execute("SELECT status, count(*) FROM index_pointer GROUP BY status").fetchall()
        )
    assert "unresolved" not in statuses
    assert statuses["corrected"] == 2 * sum(1 for e in known if e.code)
    assert statuses["excluded"] == sum(1 for e in known if not e.code)
    excluded = [w for w in report.warnings if w.startswith("excluded ")]
    assert len(excluded) == statuses["excluded"]
