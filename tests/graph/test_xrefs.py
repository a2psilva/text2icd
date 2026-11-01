import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.graph.build import BOOKS, BuildReport, build_graph, term_offsets
from text2icd.graph.xrefs import XREF_EXCEPTIONS, CrossReference, XrefResolver, cross_references
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.index import IndexBook, IndexSource, parse_index
from text2icd.ontology.validate import IndexReference, KnownException, load_exceptions

HEAD = "<ICD10CM.index><version>2099</version><title>{}</title>{}<letter><title>A</title>"
TAIL = "</letter></ICD10CM.index>"
INDEX = """
<mainTerm><title>Carrier<nemod>(suspected)</nemod> of</title>
  <term level="1"><title>hepatitis</title><code>Z22.50</code></term></mainTerm>
<mainTerm><title>Chagas'<nemod>(-Mazza) disease (chronic)</nemod></title>
  <code>B57.2</code></mainTerm>
<mainTerm><title>Complication<nemod>(s) (from) (of)</nemod></title>
  <term level="1"><title>joint prosthesis</title>
    <term level="2"><title>mechanical</title><code>T84.099</code></term></term>
  <term level="1"><title>infusion</title><code>T80.90</code></term></mainTerm>
<mainTerm><title>Diabetes, diabetic<nemod>(mellitus)</nemod></title><code>E11.9</code>
  <term level="1"><title>type 1</title><code>E10.9</code></term>
  <term level="1"><title>with</title>
    <term level="2"><title>amyotrophy</title><code>E11.44</code></term></term></mainTerm>
<mainTerm><title>Fracture, burst</title><code>T14.8</code></mainTerm>
<mainTerm><title>Fracture, traumatic</title><code>T14.8</code>
  <term level="1"><title>femur, femoral</title><code>S72.9-</code></term>
  <term level="1"><title>tarsal bone</title><code>S92.20-</code></term></mainTerm>
<mainTerm><title>Hydronephrosis<nemod>(atrophic) (early)</nemod> NEC</title>
  <code>N13.30</code></mainTerm>
<mainTerm><title>Legg<nemod>(-Calvé)</nemod>-Perthes disease, syndrome or osteochondrosis</title>
  <code>M91.1-</code></mainTerm>
<mainTerm><title>Light-for-dates<nemod>(infant)</nemod></title><code>P05.00</code></mainTerm>
<mainTerm><title>Neoplasm, neoplastic</title><see>Table of Neoplasms</see>
  <term level="1"><title>lipomatous, benign</title><code>D17.9</code></term></mainTerm>
<mainTerm><title>Thyrotoxicosis</title><code>E05.90</code>
  <term level="1"><title>with</title>
    <term level="2"><title>goiter</title><code>E05.00</code></term></term></mainTerm>
"""
EINDEX = """<mainTerm><title>Fracture</title><code>W19</code></mainTerm>"""
NEOPLASM_HEADS = "".join(
    f'<head col="{col}">{head}</head>'
    for col, head in enumerate(
        [
            "Neoplasm",
            "Malignant Primary",
            "Malignant Secondary",
            "Ca in situ",
            "Benign",
            "Uncertain Behavior",
            "Unspecified Behavior",
        ],
        start=1,
    )
)
NEOPLASM = """<mainTerm><title>Neoplasm, neoplastic</title><cell col="2">C80.1</cell>
  <term level="1"><title>skin NOS</title><cell col="2">C44.90</cell>
    <term level="2"><title>ear</title><cell col="2">C44.20-</cell></term></term>
  <term level="1"><title>intestine, intestinal</title><cell col="2">C26.0</cell>
    <term level="2"><title>small</title><cell col="2">C17.9</cell></term></term></mainTerm>"""
DRUG_HEADS = "".join(
    f'<head col="{col}">{head}</head>'
    for col, head in enumerate(
        [
            "Substance",
            "Poisoning Accidental (unintentional)",
            "Poisoning Intentional self-harm",
            "Poisoning Assault",
            "Poisoning Undetermined",
            "Adverse effect",
            "Underdosing",
        ],
        start=1,
    )
)
DRUG = """<mainTerm><title>Levocabastine</title><cell col="2">T45.0X1</cell></mainTerm>
<mainTerm><title>Vitamins NEC</title><cell col="2">T45.2X1</cell></mainTerm>"""


@pytest.fixture(scope="module")
def books(tmp_path_factory: pytest.TempPathFactory) -> dict[IndexSource, IndexBook]:
    root = tmp_path_factory.mktemp("xrefs")
    content: dict[IndexSource, str] = {
        "index": HEAD.format("Index", "") + INDEX + TAIL,
        "eindex": HEAD.format("External Cause Index", "") + EINDEX + TAIL,
        "neoplasm": HEAD.format("Neoplasm Table", f"<indexHeading>{NEOPLASM_HEADS}</indexHeading>")
        + NEOPLASM
        + TAIL,
        "drug": HEAD.format("Drugs", f"<indexHeading>{DRUG_HEADS}</indexHeading>") + DRUG + TAIL,
    }
    parsed = {}
    for source in BOOKS:
        path = root / f"{source}.xml"
        path.write_text(content[source], encoding="utf-8")
        parsed[source] = parse_index(path, source)
    return parsed


@pytest.fixture(scope="module")
def resolve(books: dict[IndexSource, IndexBook]) -> Callable[..., tuple[object, ...]]:
    """Resolve `written` as a `see` of the first index term; give the target as a book path."""
    offsets = term_offsets(books)
    resolver = XrefResolver(books, offsets)
    paths = {
        offsets[source] + i: f"{source}: {path}"
        for source, book in books.items()
        for i, path in enumerate(book.paths())
    }

    def run(written: str, source: IndexSource = "index") -> tuple[object, ...]:
        xref = resolver.resolve(source, 0, "see", written)
        target = paths.get(xref.target) if xref.target is not None else None
        return (xref.status, target, xref.column, xref.instruction, xref.remainder)

    return run


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        # A path of segments, with lenient forms of the titles.
        (
            "Complications, joint prosthesis, mechanical",
            "index: Complication > joint prosthesis > mechanical",
        ),
        ("Complication(s), infusion", "index: Complication > infusion"),
        ("Carrier, hepatitis", "index: Carrier of > hepatitis"),
        ("Chagas disease", "index: Chagas' disease"),
        ("Hydronephrosis", "index: Hydronephrosis NEC"),
        ("Legg-Calvé-Perthes disease", "index: Legg -Perthes disease, syndrome or osteochondrosis"),
        ("Light for dates", "index: Light-for-dates"),
        ("Diabetes, with, amyotrophy", "index: Diabetes, diabetic > with > amyotrophy"),
        # A title holding a comma, and a segment holding two levels.
        ("Fracture, traumatic, femur", "index: Fracture, traumatic > femur, femoral"),
        ("Thyrotoxicosis with goiter", "index: Thyrotoxicosis > with > goiter"),
        # The head of a title before the comma; the main entry wins between equal heads, and the
        # term's own book before the External Cause Index.
        ("Fracture, femoral", "index: Fracture, traumatic > femur, femoral"),
    ],
)
def test_resolved_paths(
    resolve: Callable[..., tuple[object, ...]], written: str, expected: str
) -> None:
    assert resolve(written) == ("resolved", expected, None, None, None)


def test_table_references(resolve: Callable[..., tuple[object, ...]]) -> None:
    # A column of the Neoplasm Table beats a subterm reached through an alternative title.
    assert resolve("Neoplasm, benign, by site") == (
        "resolved",
        "neoplasm: Neoplasm, neoplastic",
        5,
        "by site",
        None,
    )
    assert resolve("Neoplasm, skin, ear, malignant") == (
        "resolved",
        "neoplasm: Neoplasm, neoplastic > skin NOS > ear",
        2,
        None,
        None,
    )
    assert resolve("Table of Neoplasms, intestine, small, in situ") == (
        "resolved",
        "neoplasm: Neoplasm, neoplastic > intestine, intestinal > small",
        4,
        None,
        None,
    )
    assert resolve("Table of Drugs and Chemicals, vitamins, adverse effect") == (
        "resolved",
        "drug: Vitamins NEC",
        6,
        None,
        None,
    )


def test_partial_references_keep_the_words_left(
    resolve: Callable[..., tuple[object, ...]],
) -> None:
    assert resolve("Fracture, tarsal, talus") == (
        "partial",
        "index: Fracture, traumatic",
        None,
        None,
        "tarsal, talus",
    )
    # After "by type" the coder picks the type first: the descent stops at the instruction.
    assert resolve("Diabetes, by type, with amyotrophy") == (
        "partial",
        "index: Diabetes, diabetic",
        None,
        "by type",
        "with amyotrophy",
    )


@pytest.mark.parametrize(
    ("written", "column", "instruction"),
    [
        ("condition", None, "condition"),
        ("specific condition", None, "specific condition"),
        ("category M01", None, "category M01"),
        ("N00-N07 with fourth character .8", None, "N00-N07 with fourth character .8"),
        ("Levo-", None, "Levo-"),
        ("Table of Drugs and Chemicals", None, None),
        ("Table of Drugs and Chemicals, by drug, poisoning", 2, "by drug"),
        (
            "Table of Drugs and Chemicals, categories T36-T50, with final character 6",
            None,
            "categories T36-T50, with final character 6",
        ),
    ],
)
def test_instructions(
    resolve: Callable[..., tuple[object, ...]],
    written: str,
    column: int | None,
    instruction: str | None,
) -> None:
    assert resolve(written) == ("instruction", None, column, instruction, None)


def test_unresolved(resolve: Callable[..., tuple[object, ...]]) -> None:
    assert resolve("Meralgia paraesthetica") == (
        "unresolved",
        None,
        None,
        None,
        "meralgia paraesthetica",
    )
    assert resolve("also")[0] == "unresolved"


def test_exceptions_give_reasons(books: dict[IndexSource, IndexBook], tmp_path: Path) -> None:
    path = tmp_path / "index.xml"
    path.write_text(
        HEAD.format("Index", "")
        + "<mainTerm><title>Roth disease</title><see>Meralgia paraesthetica</see></mainTerm>"
        + "<mainTerm><title>Gait</title><see>Walk</see></mainTerm>"
        + TAIL
    )
    small = {**books, "index": parse_index(path, "index")}
    known = [
        KnownException(
            IndexReference("index", "Roth disease", "see", "Meralgia paraesthetica"),
            frozenset({"fy2099"}),
            "Spelling.",
        ),
        KnownException(IndexReference("index", "Gone", "see", "X"), frozenset({"fy2099"}), "Old."),
    ]
    result = cross_references(small, term_offsets(small), known)
    by_written = {xref.written: xref for xref in result.xrefs}
    assert by_written["Meralgia paraesthetica"] == CrossReference(
        0,
        "see",
        "Meralgia paraesthetica",
        "unresolved",
        remainder="meralgia paraesthetica",
        reason="Spelling.",
    )
    assert by_written["Walk"].reason is None
    assert result.warnings == (
        "unresolved see of index: Gait: 'Walk'",
        "cross-reference exception not observed: index: Gone [see] 'X'",
    )


def test_packaged_exceptions_load() -> None:
    exceptions = load_exceptions(resource=XREF_EXCEPTIONS)
    assert len(exceptions) == 16
    assert {e.reference.field for e in exceptions} == {"see", "see_also"}
    assert all(e.reason and e.code is None for e in exceptions)


def test_fixture_xrefs(fixture_db: sqlite3.Connection) -> None:
    rows = fixture_db.execute(
        "SELECT t.path, x.status, g.path, x.col, x.instruction, x.remainder FROM index_xref x "
        "JOIN index_term t ON t.id = x.term LEFT JOIN index_term g ON g.id = x.target "
        "ORDER BY x.term, x.kind"
    ).fetchall()
    assert ("Abdomen, abdominal", "instruction", None, None, "condition", None) in rows
    assert (
        "Neoplasm, neoplastic > abdomen, abdominal > wall",
        "resolved",
        "Neoplasm, neoplastic > abdomen, abdominal > wall > skin",
        None,
        None,
        None,
    ) in rows
    assert len(rows) == 6


def test_fixture_build_reports_xrefs(fixture_build: BuildReport) -> None:
    (line,) = [d for d in fixture_build.details if d.startswith("cross-references: ")]
    assert line.startswith("cross-references: 6; resolved ")
    unresolved = [w for w in fixture_build.warnings if w.startswith("unresolved ")]
    assert "unresolved see of index: Abandonment: 'Maltreatment'" in unresolved


def test_build_uses_xref_exceptions(fixture_release: ReleaseFiles, tmp_path: Path) -> None:
    exceptions = tmp_path / "xref.toml"
    exceptions.write_text(
        'schema = 1\n[[exception]]\nreleases = ["fy2099"]\nbook = "index"\n'
        'path = "Abandonment"\nfield = "see"\nvalue = "Maltreatment"\nreason = "test"\n'
    )
    report = build_graph(
        fixture_release, "fy2099", tmp_path / "g.sqlite", xref_exceptions=exceptions
    )
    assert not any("Abandonment" in w for w in report.warnings)
    with sqlite3.connect(report.path) as conn:
        reason = conn.execute("SELECT reason FROM index_xref WHERE written = 'Maltreatment'")
        assert reason.fetchone() == ("test",)


@pytest.mark.slow
@pytest.mark.parametrize("release_id", ["fy2026", "fy2026-apr", "fy2027"])
def test_release_xrefs(release_graph: Callable[[str], BuildReport], release_id: str) -> None:
    report = release_graph(release_id)
    with sqlite3.connect(report.path) as conn:
        statuses = dict(
            conn.execute("SELECT status, count(*) FROM index_xref GROUP BY status").fetchall()
        )
        total = sum(statuses.values())
        assert total > 15_000
        assert statuses["unresolved"] / total < 0.01
        assert statuses["resolved"] / total > 0.85
        # Every unresolved reference is a reviewed error in the Index, with its reason.
        unexplained = conn.execute(
            "SELECT count(*) FROM index_xref WHERE status = 'unresolved' AND reason IS NULL"
        ).fetchone()
        assert unexplained == (0,)
        malignant = conn.execute(
            "SELECT x.col, x.instruction, g.book, g.path FROM index_xref x "
            "JOIN index_term g ON g.id = x.target "
            "WHERE x.written = 'Neoplasm, malignant, by site' GROUP BY 1, 2, 3, 4"
        ).fetchall()
        assert malignant == [(2, "by site", "neoplasm", "Neoplasm, neoplastic")]
    assert not [w for w in report.warnings if "cross-reference" in w or w.startswith("unresolved")]
