import shutil
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.graph.build import BuildReport
from text2icd.graph.integrity import CheckResult, check_graph, failed


def outcomes(conn: sqlite3.Connection) -> dict[str, CheckResult]:
    return {result.name: result for result in check_graph(conn)}


@pytest.fixture
def copy(fixture_build: BuildReport, tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """A writable copy of the fixture graph, to break."""
    path = tmp_path / "copy.sqlite"
    shutil.copy(fixture_build.path, path)
    conn = sqlite3.connect(path)
    yield conn
    conn.close()


def test_fixture_graph(fixture_db: sqlite3.Connection) -> None:
    # The fixture holds a slice of the CMS files: its Index and notes name codes it lacks.
    results = outcomes(fixture_db)
    assert {name: result.outcome for name, result in results.items()} == {
        "content hash": "pass",
        "tabular tree": "pass",
        "billable codes": "fail",
        "index tree": "pass",
        "index pointers": "fail",
        "cross-references": "fail",
        "note edges": "warn",
        "search tables": "pass",
        "reachable from the Index": "warn",
    }
    assert "index: Amyloid heart [code] E85.4" in results["index pointers"].problems
    assert "index: Abandonment [see] Maltreatment" in results["cross-references"].problems
    assert "E11.22 useAdditionalCode: N18.1-N18.6" in results["note edges"].problems
    assert failed(list(results.values()))


def test_changed_rows_break_the_content_hash(copy: sqlite3.Connection) -> None:
    copy.execute("UPDATE tabular_node SET description = 'changed' WHERE key = 'E11.22'")
    assert outcomes(copy)["content hash"].outcome == "fail"


def test_cycles_and_bad_depths(copy: sqlite3.Connection) -> None:
    copy.execute("UPDATE tabular_node SET parent = 'E11.22' WHERE key = 'E11'")
    copy.execute("UPDATE index_term SET level = 3 WHERE path = 'Pneumonia > aspiration'")
    results = outcomes(copy)
    tabular = results["tabular tree"]
    assert tabular.outcome == "fail"
    assert "E11 depth 2" in tabular.problems
    assert "3 nodes not reached from a chapter" in tabular.problems
    index = results["index tree"]
    assert index.outcome == "fail"
    assert any(problem.endswith("Pneumonia > aspiration") for problem in index.problems)


def test_a_missing_table_fails_its_check(copy: sqlite3.Connection) -> None:
    copy.execute("DROP TABLE fts_tabular")
    result = outcomes(copy)["search tables"]
    assert result.outcome == "fail"
    assert result.summary == "database error: no such table: fts_tabular"


def test_listed_unresolved_xrefs_pass(copy: sqlite3.Connection) -> None:
    copy.execute("UPDATE index_xref SET reason = 'known' WHERE status = 'unresolved'")
    # Half the fixture's six references are unresolved: over the 1% limit even when listed.
    result = outcomes(copy)["cross-references"]
    assert result.outcome == "fail"
    assert result.problems == ("3 unresolved of 6, over 1%",)


def test_result_lines() -> None:
    result = CheckResult("x", "warn", "7 found", tuple(f"p{i}" for i in range(7)))
    assert result.lines() == [
        "WARN  x: 7 found",
        "        p0",
        "        p1",
        "        p2",
        "        p3",
        "        p4",
        "        ... and 2 more",
    ]


def test_graph_check_command(fixture_build: BuildReport) -> None:
    result = CliRunner().invoke(cli.app, ["graph", "check", "--graph", str(fixture_build.path)])
    assert result.exit_code == 1
    assert "FAIL  index pointers: ok " in result.output
    assert result.output.endswith("9 checks: 4 passed, 2 warned, 3 failed\n")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("release_id", "unresolved_edges"),
    [
        ("fy2026", ("F02 codeFirst: G31.90",)),
        ("fy2026-apr", ("F02 codeFirst: G31.90",)),
        ("fy2027", ()),
    ],
)
def test_release_integrity(
    release_graph: Callable[[str], BuildReport],
    release_id: str,
    unresolved_edges: tuple[str, ...],
) -> None:
    with sqlite3.connect(release_graph(release_id).path) as conn:
        results = outcomes(conn)
    assert not failed(list(results.values())), [r.lines() for r in results.values()]
    assert results["note edges"].problems == unresolved_edges
    reach = results["reachable from the Index"]
    assert len(reach.problems) < 50
    assert reach.summary.startswith("99.9")
