import re
import sqlite3
from collections.abc import Callable

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.notes import CodeIndex, find_references
from text2icd.ontology.codes import parse_reference


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("stage of chronic kidney disease (N18.1-N18.6)", [("N18.1-N18.6", "N181", "N186")]),
        ("type 1 diabetes mellitus (E10.-)", [("E10.-", "E10", "E10")]),
        ("secondary neoplasm (C79.8-)", [("C79.8-", "C798", "C798")]),
        ("category T51", [("T51", "T51", "T51")]),
        ("birth trauma (P10\N{EN DASH}P15)", [("P10\N{EN DASH}P15", "P10", "P15")]),
        (
            "Conditions in O00-O77.-, O85-O94",
            [("O00-O77.-", "O00", "O77"), ("O85-O94", "O85", "O94")],
        ),
        ("classifiable to C7A.098, C76", [("C7A.098", "C7A098", "C7A098"), ("C76", "C76", "C76")]),
        # Stray dots found in the CMS files.
        ("thoracic disc disorders (M51.1.-)", [("M51.1.-", "M511", "M511")]),
        ("assigned to code O33.3. 7th character", [("O33.3.", "O333", "O333")]),
        ("fracture of neck (S12.0--S12.6.-)", [("S12.0--S12.6.-", "S120", "S126")]),
        ("tic disorders (F98.2.-F98.3)", [("F98.2.-F98.3", "F982", "F983")]),
        # A range whose end comes before its start is not a reference.
        ("backwards (K93-K00)", []),
    ],
)
def test_find_references(text: str, expected: list[tuple[str, str, str]]) -> None:
    found = find_references(text)
    assert [(r.written, r.reference.start, r.reference.end) for r in found] == expected


@pytest.mark.parametrize(
    "text", ["G6PD deficiency", "influenza A H1N1", "E. coli O157", "HIV disease", "E11a", "type 2"]
)
def test_find_references_skips_words(text: str) -> None:
    assert find_references(text) == []


@pytest.mark.parametrize(
    ("written", "is_range", "is_bare"),
    [
        ("B12", False, True),
        ("E10.-", False, False),
        ("G31.90", False, False),
        ("J09-J18", True, False),
    ],
)
def test_reference_shape(written: str, is_range: bool, is_bare: bool) -> None:
    (found,) = find_references(f"({written})")
    assert (found.is_range, found.is_bare) == (is_range, is_bare)


CODES = CodeIndex(["K00.0", "K92.9", "N18.1", "N18.30", "N18.6", "E10.9", "D51.0"])


@pytest.mark.parametrize(
    ("written", "inside"),
    [
        ("K00-K93", True),  # K93 is a WHO category, absent from ICD-10-CM
        ("K94-K95", False),
        ("N18.2-N18.5", True),
        ("N18.4-N18.5", False),
        ("E10.-", True),
        ("N18.6", True),
        ("E11", False),
        ("B12", False),
        ("A00-B99", False),
    ],
)
def test_code_index_any_in(written: str, inside: bool) -> None:
    assert CODES.any_in(parse_reference(written)) is inside


def test_code_index_sorts_undotted_codes() -> None:
    assert list(CODES) == ["D510", "E109", "K000", "K929", "N181", "N1830", "N186"]


def test_fixture_edges(fixture_db: sqlite3.Connection) -> None:
    def edges(where: str) -> list[tuple[object, ...]]:
        return fixture_db.execute(
            "SELECT source, kind, written, range_start, range_end, target, status "
            f"FROM tabular_edge WHERE {where} ORDER BY note, ordinal"
        ).fetchall()

    assert edges("status = 'ok'") == [
        ("E08", "excludes1", "E11.-", "E11", "E11", "E11", "ok"),
        ("E11", "excludes1", "E08.-", "E08", "E08", "E08", "ok"),
    ]
    assert edges("source = 'E11.22'") == [
        ("E11.22", "useAdditionalCode", "N18.1-N18.6", "N18.1", "N18.6", None, "unresolved")
    ]
    assert edges("source = '19'") == [
        ("19", "excludes2", "P10-P15", "P10", "P15", None, "unresolved"),
        ("19", "excludes2", "O70-O71", "O70", "O71", None, "unresolved"),
    ]
    # "to be assigned to codes in subcategory E08.32" names the note's own code: not an edge.
    assert edges("target = source") == []


def test_fixture_edge_log(fixture_build: BuildReport) -> None:
    # The fixture holds a few codes only, so most references point outside it.
    unresolved = [w for w in fixture_build.warnings if w.startswith("no code for ")]
    assert len(unresolved) == fixture_build.rows["tabular_edge"] - 2
    assert "no code for 'N18.1-N18.6' in a useAdditionalCode note of E11.22" in unresolved
    assert fixture_build.details[0].startswith("note references: 37; notes with one: excludes1")


WORDS = {"B12", "CR3", "GH1", "GM1", "GM2", "GM3"}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("release_id", "unresolved"),
    [
        ("fy2026", [("F02", "G31.90")]),
        ("fy2026-apr", [("F02", "G31.90")]),
        ("fy2027", []),
    ],
)
def test_release_edges(
    release_graph: Callable[[str], BuildReport], release_id: str, unresolved: list[tuple[str, str]]
) -> None:
    report = release_graph(release_id)
    with sqlite3.connect(report.path) as conn:
        assert (
            conn.execute(
                "SELECT source, written FROM tabular_edge WHERE status = 'unresolved'"
            ).fetchall()
            == unresolved
        )
        untargeted = conn.execute(
            "SELECT count(*) FROM tabular_edge WHERE range_start = range_end AND target IS NULL"
        ).fetchone()
        assert untargeted == (len(unresolved),)
        z85 = conn.execute(
            "SELECT e.ordinal, e.written FROM tabular_edge e JOIN tabular_note n ON n.id = e.note "
            "WHERE e.source = 'Z85.89' AND n.text LIKE 'Conditions classifiable to C7A.098%'"
        ).fetchall()
        assert z85 == [(0, "C7A.098"), (1, "C76"), (2, "C77-C79")]
        assert conn.execute("SELECT count(*) FROM tabular_edge").fetchone()[0] > 11_000
    (words,) = [line for line in report.details if line.startswith("not taken as codes: ")]
    assert set(re.findall(r"(\w+) \(\d+\)", words)) == WORDS
    assert sum(w.startswith("no code for ") for w in report.warnings) == len(unresolved)
