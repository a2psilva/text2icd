"""The Tabular notes inside a code set: companion codes, Excludes1 and sequencing (6.6 to 6.8)."""

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load
from text2icd.ontology.codes import CodeReference
from text2icd.resolve.coder import EXCLUDES1, UNRESOLVED, Coder, describe_code_set
from text2icd.resolve.instructions import (
    Entry,
    Instruction,
    Kind,
    Notes,
    Order,
    resolve_excludes1,
    sequence,
)
from text2icd.trace import TraceStep


def entry(code: str, role: str = "diagnosis", source: str | None = None, group: int = 0) -> Entry:
    return Entry(code, code, role, source, group, 1.0, True)  # type: ignore[arg-type]


class FakeNotes(Notes):
    """Notes given by hand: instructions, Excludes1 references and manifestation codes."""

    def __init__(
        self,
        instructions: dict[str, list[tuple[Kind, str]]] | None = None,
        excludes: dict[str, str] | None = None,
        elsewhere: tuple[str, ...] = (),
    ) -> None:
        self._given = {
            code: tuple(Instruction(code, kind, ref, (_ref(ref),)) for kind, ref in found)
            for code, found in (instructions or {}).items()
        }
        self._excluding = excludes or {}
        self._manifestations = elsewhere

    def instructions(self, code: str) -> tuple[Instruction, ...]:
        return self._given.get(code, ())

    def excludes1(self, code: str) -> tuple[tuple[CodeReference, str, str], ...]:
        ref = self._excluding.get(code)
        return ((_ref(ref), f"note ({ref})", code),) if ref else ()

    def classified_elsewhere(self, code: str) -> bool:
        return code in self._manifestations


def _ref(written: str) -> CodeReference:
    start, _, end = written.replace(".", "").partition("-")
    return CodeReference(start, end or start)


def codes(entries: list[Entry]) -> list[str]:
    return [e.code for e in entries]


@pytest.fixture
def noted(graph_file: Path) -> Iterator[Graph]:
    """The fixture graph with notes of the FY2027 kind on codes its Index can reach. The
    fixture's notes name codes its Tabular List lacks (N18, Z79): taken as resolved here."""
    with sqlite3.connect(graph_file) as conn:
        conn.execute("UPDATE tabular_edge SET status = 'ok'")
        notes = [
            # I43: Cardiomyopathy in diseases classified elsewhere
            (9001, "I43", "codeFirst", "underlying disease, such as: amyloidosis (E85.-)"),
            (9002, "J69.0", "excludes1", "organ-limited amyloidosis (E85.4)"),
            (9003, "J85.1", "useAdditionalCode", "code to identify the infection (A49.9)"),
        ]
        for note_id, node, kind, text in notes:
            conn.execute(
                "INSERT INTO tabular_note VALUES (?, ?, ?, ?, ?)", (note_id, node, kind, 1, text)
            )
        edges = [
            (9001, "I43", "codeFirst", "E85.-", "E85", "E85", "E85"),
            (9002, "J69.0", "excludes1", "E85.4", "E85.4", "E85.4", "E85.4"),
            (9003, "J85.1", "useAdditionalCode", "A49.9", "A49.9", "A49.9", None),
        ]
        for note_id, source, kind, written, start, end, target in edges:
            conn.execute(
                "INSERT INTO tabular_edge VALUES (?, 1, ?, ?, ?, ?, ?, ?, 'ok')",
                (note_id, source, kind, written, start, end, target),
            )
    conn.close()
    with load(path=graph_file) as loaded:
        yield loaded


# -- reading the notes -----------------------------------------------------------------------------


def test_instructions_are_inherited_and_read_with_their_heading(noted: Graph) -> None:
    found = Notes(noted).instructions("E11.22")
    assert [(i.node, i.kind) for i in found] == [
        ("E11.22", "useAdditionalCode"),
        ("E11", "useAdditionalCode"),
        ("E11", "useAdditionalCode"),
        ("E11", "useAdditionalCode"),  # "oral hypoglycemic drugs" names Z79.84 again: read once
    ]
    stage, *control = found
    assert stage.text == "code to identify stage of chronic kidney disease (N18.1-N18.6)"
    assert stage.mandatory  # one note, no condition
    assert control[1].text == "code to identify control using: insulin (Z79.4)"
    assert all(i.alternative and not i.conditional for i in control)  # whichever applies
    assert stage.covers("N18.31")
    assert not stage.covers("N18.9")
    assert stage.reaches("N18")
    assert stage.reaches("N18.3")
    assert not stage.reaches("N18.9")
    assert Instruction("D63.1", "codeFirst", "", (_ref("N18"),)).reaches("N18.9")


def test_conditional_wording(noted: Graph) -> None:
    (first, *_) = (i for i in Notes(noted).instructions("E08.3211") if i.kind == "codeFirst")
    assert first.text == "the underlying condition, such as: congenital rubella (P35.0)"
    assert first.alternative
    assert not first.conditional


def test_excludes1_and_manifestation_codes(noted: Graph) -> None:
    notes = Notes(noted)
    assert notes.excluding("E11.22", "E08.3211") == (
        "diabetes mellitus due to underlying condition (E08.-)",
        "E11",
    )
    assert notes.excluding("E11.22", "T51.3X1A") is None
    assert notes.classified_elsewhere("I43")  # Cardiomyopathy in diseases classified elsewhere
    assert not notes.classified_elsewhere("E85.4")


# -- Excludes1 ---------------------------------------------------------------------------------


def test_excludes1_keeps_the_more_specific_code() -> None:
    notes = FakeNotes(excludes={"R55": "I95.1"})
    entries = [entry("R55", group=0), entry("I95.1", group=1)]
    kept, excluded, trace = resolve_excludes1(entries, notes)
    assert codes(kept) == ["I95.1"]
    assert [(e.code, e.kept) for e in excluded] == [("R55", "I95.1")]
    assert trace[0].guideline == "I.A.12.a"
    # Either code's note counts.
    assert codes(resolve_excludes1(entries, FakeNotes(excludes={"I95.1": "R55"}))[0]) == ["I95.1"]


def test_excludes1_drops_what_the_code_brought_in() -> None:
    notes = FakeNotes(excludes={"J69.0": "E85.4"})
    entries = [
        entry("E85.4", group=0),
        entry("I43", "manifestation", "E85.4", group=0),
        entry("J69.0", group=1),
        entry("J85.1", "use_additional", "J69.0", group=1),
    ]
    kept, excluded, _ = resolve_excludes1(entries, notes)
    # Equal length and score: the first one stays; the excluded code's companion goes with it.
    assert codes(kept) == ["E85.4", "I43"]
    assert [e.code for e in excluded] == ["J69.0"]


def test_excludes1_within_one_diagnosis_is_left_alone() -> None:
    notes = FakeNotes(excludes={"E85.4": "I43"})
    entries = [entry("E85.4"), entry("I43", "manifestation", "E85.4")]
    assert codes(resolve_excludes1(entries, notes)[0]) == ["E85.4", "I43"]


def test_excludes1_prefers_a_specified_code() -> None:
    notes = FakeNotes(excludes={"A01.1": "B02.2"})
    vague = Entry("A01.1", "", "diagnosis", None, 0, 3.0, specified=False)
    kept, _, _ = resolve_excludes1([vague, entry("B02.2", group=1)], notes)
    assert codes(kept) == ["B02.2"]


def test_excludes1_spares_the_pairs_a_guideline_codes_together() -> None:
    notes = FakeNotes(excludes={"K86.1": "K85"})
    entries = [entry("K85.90"), entry("K86.1", group=1)]  # acute and chronic (I.B.8)
    assert codes(resolve_excludes1(entries, notes, {frozenset({0, 1})})[0]) == ["K85.90", "K86.1"]
    assert codes(resolve_excludes1(entries, notes)[0]) == ["K85.90"]  # the more specific


# -- sequencing ------------------------------------------------------------------------------------


def test_guideline_orders_move_codes() -> None:
    step = TraceStep(
        layer="rules", action="sequence", detail="R73.9 first", guideline="I.C.19.e.5.c"
    )
    ordered, trace = sequence(
        [entry("T38.3X6A"), entry("R73.9", group=1)],
        FakeNotes(),
        [Order("R73.9", "T38.3X6A", step)],
    )
    assert codes(ordered) == ["R73.9", "T38.3X6A"]
    assert trace == [step]


def test_code_first_puts_the_named_code_before() -> None:
    notes = FakeNotes({"N18.31": [("codeFirst", "E11.22")]})
    ordered, trace = sequence([entry("N18.31"), entry("E11.22", group=1)], notes)
    assert codes(ordered) == ["E11.22", "N18.31"]
    assert trace[0].detail.startswith("E11.22 before N18.31: code first")
    assert trace[0].guideline == "I.A.13"


def test_use_additional_code_puts_the_named_code_after() -> None:
    notes = FakeNotes({"N39.0": [("useAdditionalCode", "B95-B97")]})
    ordered, trace = sequence([entry("B96.20"), entry("N39.0", group=1)], notes)
    assert codes(ordered) == ["N39.0", "B96.20"]
    assert trace[0].guideline == "I.B.7"


def test_code_also_gives_no_order() -> None:
    notes = FakeNotes({"J44.0": [("codeAlso", "J18.9")]})
    ordered, trace = sequence([entry("J18.9"), entry("J44.0", group=1)], notes)
    assert (codes(ordered), trace) == (["J18.9", "J44.0"], [])


def test_manifestation_never_primary() -> None:
    notes = FakeNotes(elsewhere=("I43",))
    ordered, trace = sequence([entry("I43"), entry("E85.4", group=1)], notes)
    assert codes(ordered) == ["E85.4", "I43"]
    assert "never primary" in trace[-1].detail
    # Alone, it stays: there is nothing to put first.
    assert codes(sequence([entry("I43")], notes)[0]) == ["I43"]


def test_cycles_keep_the_phrase_order() -> None:
    notes = FakeNotes({"A01.1": [("codeFirst", "B02.2")], "B02.2": [("codeFirst", "A01.1")]})
    ordered, _ = sequence([entry("A01.1"), entry("B02.2", group=1)], notes)
    assert codes(ordered) == ["A01.1", "B02.2"]


# -- the coder ----------------------------------------------------------------------------------


def test_coder_reads_the_notes(noted: Graph) -> None:
    coder = Coder(noted)
    amyloid = coder.code("amyloid heart disease")
    assert amyloid.codes == ("E85.4", "I43")
    assert [s.role for s in amyloid.sequence] == ["diagnosis", "manifestation"]
    (answered,) = amyloid.companions
    assert (answered.source, answered.status, answered.code) == ("I43", "present", "E85.4")
    # A mandatory instruction the phrase says nothing about is listed for review.
    abscess = coder.code("pneumonia with lung abscess")
    assert abscess.codes == ("J85.1",)
    (missing,) = abscess.companions
    assert (missing.status, missing.code, abscess.flags) == (UNRESOLVED, None, (UNRESOLVED,))
    assert describe_code_set(abscess)[-2].startswith('? J85.1: use additional code "code to')


def test_coder_resolves_excludes1(noted: Graph) -> None:
    both = Coder(noted).code("amyloid heart disease, aspiration pneumonia due to food")
    assert [d.code for d in both.diagnoses] == ["E85.4", "J69.0"]
    # E85.4 is the default code of its main term: the specified J69.0 stays, and the
    # manifestation E85.4 brought in goes with it.
    assert both.codes == ("J69.0",)
    assert [(e.code, e.kept) for e in both.excluded] == [("E85.4", "J69.0")]
    assert EXCLUDES1 in both.flags
    assert both.trace[0].action == "excludes1"


# -- FY2027 --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release(release_graph: Callable[[str], BuildReport]) -> Iterator[Graph]:
    with load(path=release_graph("fy2027").path) as graph:
        yield graph


@pytest.fixture(scope="module")
def coder(release: Graph) -> Coder:
    return Coder.from_graph(release)


@pytest.mark.slow
def test_readme_example(coder: Coder) -> None:
    code_set = coder.code("type 2 DM with diabetic CKD stage 3a")
    assert code_set.codes == ("E11.22", "N18.31")
    assert [s.role for s in code_set.sequence] == ["diagnosis", "use_additional"]
    added = next(c for c in code_set.companions if c.status == "added")
    assert (added.source, added.kind, added.code) == ("E11.22", "useAdditionalCode", "N18.31")
    assert added.trace[0].guideline == "I.B.7"
    assert "Disease, diseased > kidney > chronic > stage 3a" in added.trace[-1].detail
    assert code_set.flags == ()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("phrase", "codes", "flags"),
    [
        ("type 2 diabetes mellitus with CKD stage 3a", ("E11.22", "N18.31"), ()),
        # "type 2" never gives a stage 2: each condition is walked alone for the stage.
        ("type 2 diabetes with chronic kidney disease", ("E11.22",), (UNRESOLVED,)),
        # The I12.9 note names N18.9 too.
        ("hypertensive chronic kidney disease", ("I12.9", "N18.9"), ()),
        ("dementia in Alzheimer's disease", ("G30.9", "F02.80"), ()),
        # B95-B97 to identify the infectious agent: not documented.
        ("urinary tract infection", ("N39.0",), (UNRESOLVED,)),
        # R55 Excludes1 orthostatic hypotension (I95.1).
        ("syncope and collapse, orthostatic hypotension", ("I95.1",), (EXCLUDES1,)),
        # Code first the underlying CKD (I.A.13): N18.9 from the same Index walk.
        ("anemia in chronic kidney disease", ("N18.9", "D63.1"), ()),
        # N18.6: "use additional code to identify dialysis status (Z99.2)", not documented.
        ("anemia in end stage renal disease", ("N18.6", "D63.1"), (UNRESOLVED,)),
        # Lists of alternatives and "if applicable" notes are coded only when documented.
        ("heart failure", ("I50.9",), ()),
        ("type 2 diabetes mellitus", ("E11.9",), ()),
    ],
)
def test_release_notes(
    coder: Coder, phrase: str, codes: tuple[str, ...], flags: tuple[str, ...]
) -> None:
    code_set = coder.code(phrase)
    assert (code_set.codes, code_set.flags) == (codes, flags)


@pytest.mark.slow
def test_release_manifestation_alone_needs_its_etiology(coder: Coder) -> None:
    code_set = coder.code("anemia in neoplastic disease")
    assert code_set.codes == ("D63.0",)  # Anemia in neoplastic disease: code first the neoplasm
    assert UNRESOLVED in code_set.flags
    assert {c.kind for c in code_set.companions if c.status == UNRESOLVED} == {"codeFirst"}
