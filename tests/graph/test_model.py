import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from text2icd.config import Settings
from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, GraphError, load


@pytest.fixture
def graph(fixture_build: BuildReport) -> Iterator[Graph]:
    with load(path=fixture_build.path) as loaded:
        yield loaded


def test_codes(graph: Graph) -> None:
    code = graph.code("E11.22")
    assert graph.code("E1122") == code
    assert (code.kind, code.billable, code.depth, code.parent) == ("diag", True, 4, "E11.2")
    assert "E1122" in graph
    assert "E11.22" in graph
    assert "Z99.99" not in graph
    assert 3 not in graph
    with pytest.raises(KeyError):
        graph.code("Z99.99")
    assert graph.parent("E11.22") == graph.code("E11.2")
    assert graph.parent("4") is None
    assert [c.key for c in graph.ancestors("E11.22")] == ["E11.2", "E11", "E08-E13", "4"]
    assert [c.key for c in graph.children("E11")] == ["E11.2"]
    assert graph.children("E11.22") == ()
    assert next(graph.codes()).key == "4"


def test_generated_codes_and_seventh_characters(graph: Graph) -> None:
    code = graph.code("T51.3X1A")
    assert (code.kind, code.parent, code.seventh_from, code.billable) == (
        "generated",
        "T51.3X1",
        "T51",
        True,
    )
    assert graph.seven_char_def("T51.3X1A") == graph.seven_char_def("T51")
    assert [char for char, _ in graph.seven_char_def("T51")] == ["A", "D", "S"]
    assert graph.seven_char_def("E11.22") == ()


def test_notes_and_edges(graph: Graph) -> None:
    (note,) = graph.notes("E11.22")
    assert (note.kind, note.text) == (
        "useAdditionalCode",
        "code to identify stage of chronic kidney disease (N18.1-N18.6)",
    )
    inherited = graph.notes("E11.22", "excludes1", inherited=True)
    assert {n.node for n in inherited} == {"E11"}
    (edge,) = graph.edges("E11.22")
    assert (edge.start, edge.end, edge.target, edge.note_id) == ("N18.1", "N18.6", None, note.id)
    assert [e.written for e in graph.edges("E08", "excludes1")][-1] == "E11.-"
    assert [e.source for e in graph.edges_to("E11")] == ["E08"]


def test_index_terms(graph: Graph) -> None:
    term = graph.term_at("pneumonia >  aspiration")
    assert term is not None
    assert (term.book, term.level, term.path) == ("index", 1, "Pneumonia > aspiration")
    assert graph.term_at("Pneumonia > nothing") is None
    assert graph.term_at("Pneumonia", "drug") is None
    parent = graph.term_parent(term.id)
    assert parent is not None
    assert parent.title == "Pneumonia"
    assert graph.term_parent(parent.id) is None
    assert [t.title for t in graph.subterms(term.id)] == ["due to", "newborn", "postprocedural"]
    assert graph.main_terms("drug")[0].title == "1-propanol"
    amyloid = graph.term_at("Amyloid heart")
    assert amyloid is not None
    assert amyloid.nonessential == ("disease",)


def test_pointers_and_xrefs(graph: Graph) -> None:
    propanol = graph.term_at("1-propanol", "drug")
    assert propanol is not None
    cells = graph.pointers(propanol.id)
    assert (cells[0].field, cells[0].column, cells[0].target) == ("cell", 2, "T51.3X1")
    assert graph.column_heading("drug", 2) == "Poisoning Accidental (unintentional)"
    assert graph.column_heading("drug", 9) is None
    assert {graph.term(p.term).path for p in graph.pointers_to("T51.3X1")} == {"1-propanol"}
    abdomen = graph.term_at("Abdomen, abdominal")
    assert abdomen is not None
    (xref,) = graph.xrefs(abdomen.id)
    assert (xref.kind, xref.status, xref.instruction) == ("see_also", "instruction", "condition")
    assert graph.xrefs(propanol.id) == ()


def test_search(graph: Graph) -> None:
    (hit, *_) = graph.search_index("aspiration pneumonia")
    assert graph.term(hit.term).path == "Pneumonia > aspiration"
    assert graph.search_tabular("propanol")[0].node == "T51.3"


def test_graph_basics(graph: Graph, fixture_build: BuildReport) -> None:
    assert graph.release == "fy2099"
    assert graph.meta["content_sha256"] == fixture_build.content_sha256
    assert repr(graph).startswith("Graph('fy2099', 39 codes, 64 terms)")


def test_context_manager_closes(fixture_build: BuildReport) -> None:
    with load(path=fixture_build.path) as graph:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        graph.search_index("pneumonia")


def test_load_from_the_data_directory(fixture_build: BuildReport, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    target = settings.graph_path("fy2027")
    target.parent.mkdir(parents=True)
    shutil.copy(fixture_build.path, target)
    with load("fy2027", settings=settings) as graph:
        assert graph.path == target


def test_load_reports_a_missing_graph(tmp_path: Path) -> None:
    with pytest.raises(GraphError, match=r"run `text2icd setup -r fy2026`"):
        load("fy2026", settings=Settings(data_dir=tmp_path))


def test_load_rejects_another_schema_version(fixture_build: BuildReport, tmp_path: Path) -> None:
    copy = tmp_path / "old.sqlite"
    shutil.copy(fixture_build.path, copy)
    with sqlite3.connect(copy) as conn:
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.close()
    with pytest.raises(GraphError, match="schema version 0"):
        load(path=copy)


LOAD_BENCHMARK = """
import resource, sys, time
from pathlib import Path
from text2icd.graph.model import load
start = time.perf_counter()
graph = load(path=Path(sys.argv[1]))
seconds = time.perf_counter() - start
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(seconds, peak / 1e6 if sys.platform == "darwin" else peak / 1e3)
"""


@pytest.mark.slow
@pytest.mark.benchmark
def test_release_load_time_and_memory(release_graph: Callable[[str], BuildReport]) -> None:
    report = release_graph("fy2027")
    # A fresh interpreter, so the peak resident memory is the loader's alone.
    result = subprocess.run(
        [sys.executable, "-c", LOAD_BENCHMARK, str(report.path)],
        capture_output=True,
        text=True,
        check=True,
    )
    seconds, megabytes = map(float, result.stdout.split())
    assert seconds < 1.0
    assert megabytes < 300
