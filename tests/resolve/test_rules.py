"""The guideline rule pack (task 6.9): each rule on its own, then on the FY2027 release."""

from collections.abc import Callable, Iterator

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load
from text2icd.ontology.codes import CodeReference
from text2icd.resolve import rules
from text2icd.resolve.coder import EXCLUDES1, UNRESOLVED, Coder
from text2icd.resolve.instructions import Entry, Excluded, Instruction, Notes, Order
from text2icd.resolve.rules import Fact, hypertension_code, intent, orders, sequela
from text2icd.text.model import Modifier


def facts(*codes: str) -> list[Fact]:
    return [Fact(code, number) for number, code in enumerate(codes)]


def entry(code: str, group: int = 0, role: str = "diagnosis") -> Entry:
    return Entry(code, code, role, None, group, 1.0, True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        (("I10",), "I10"),
        (("I10", "I50.9"), "I11.0"),  # I.C.9.a.1
        (("I10", "I51.7"), "I11.9"),
        (("I10", "N18.4"), "I12.9"),  # I.C.9.a.2
        (("I10", "N18.6"), "I12.0"),
        (("I10", "I50.9", "N18.30"), "I13.0"),  # I.C.9.a.3
        (("I10", "I50.9", "N18.5"), "I13.2"),
        (("I10", "I51.9", "N18.4"), "I13.10"),
        (("I10", "I51.9", "N18.6"), "I13.11"),
        (("I13.10",), "I13.10"),  # written as one condition
        (("I12.9", "I50.22"), "I13.0"),
        (("I13.2", "N18.30"), "I13.0"),  # the stage of the N18 code wins over a joined guess
        (("I50.9", "N18.4"), None),  # no hypertension
        (("I27.20", "I50.9"), None),  # pulmonary hypertension is no I10
    ],
)
def test_hypertension_combination(codes: tuple[str, ...], expected: str | None) -> None:
    assert hypertension_code(facts(*codes)) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("T39.011A", "poisoning"),
        ("T40.1X1A", "poisoning"),
        ("T42.4X4A", "poisoning"),  # undetermined intent
        ("T36.0X5A", "adverse_effect"),
        ("T38.3X6A", "underdosing"),
        ("T51.0X1A", "poisoning"),  # a toxic effect
        ("T14.8XXA", None),
        ("S72.91XA", None),
        ("R21", None),
    ],
)
def test_intent(code: str, expected: str | None) -> None:
    assert intent(code) == expected


def test_sequela() -> None:
    assert sequela("S72.91XS")
    assert sequela("T23.001S")
    assert not sequela("S72.91XA")
    assert not sequela("I69.359")


def pairs(found: list[Order]) -> list[tuple[str, str, str | None]]:
    return [(o.first, o.then, o.step.guideline) for o in found]


def test_orders_follow_the_drug_intent() -> None:
    poisoning = [entry("T40.1X1A"), entry("R40.20", 1)]
    assert pairs(orders(poisoning, [0, 0])) == [("T40.1X1A", "R40.20", "I.C.19.e.5(b)")]
    adverse = [entry("T46.4X5A"), entry("I95.9", 1)]
    assert pairs(orders(adverse, [0, 0])) == [("I95.9", "T46.4X5A", "I.C.19.e.5(a)")]
    underdosing = [entry("T38.3X6A"), entry("R73.9", 1)]
    assert pairs(orders(underdosing, [0, 0])) == [("R73.9", "T38.3X6A", "I.C.19.e.5(c)")]
    assert orders(underdosing, [0, 1]) == []  # another diagnosis of the phrase


def test_orders_put_the_residual_before_the_sequela() -> None:
    found = orders([entry("S01.81XS"), entry("L90.5", 1)], [0, 0])
    assert pairs(found) == [("L90.5", "S01.81XS", "I.B.10")]


def test_drug_status_answers_the_note_it_names() -> None:
    insulin = Modifier(kind="diabetes_drug", value="insulin", text="on insulin", start=0, end=10)
    note = Instruction(
        "E11", "useAdditionalCode", "insulin (Z79.4)", (CodeReference("Z794", "Z794"),)
    )
    oral = Instruction(
        "E11", "useAdditionalCode", "oral (Z79.84)", (CodeReference("Z7984", "Z7984"),)
    )
    first = Instruction("E11", "codeFirst", "insulin (Z79.4)", (CodeReference("Z794", "Z794"),))
    assert rules.drug_status(note, [insulin]) == insulin
    assert rules.drug_status(oral, [insulin]) is None
    assert rules.drug_status(first, [insulin]) is None


def test_presumed_exclusion_is_no_conflict() -> None:
    assert rules.presumed(Excluded("I51.7", "I11.9", "cardiomegaly due to hypertension (I11.-)"))
    assert not rules.presumed(Excluded("E85.4", "J69.0", "(E85.4)"))


# The rules on the FY2027 release.


@pytest.fixture(scope="module")
def release(release_graph: Callable[[str], BuildReport]) -> Iterator[Graph]:
    with load(path=release_graph("fy2027").path) as graph:
        yield graph


@pytest.fixture(scope="module")
def coder(release: Graph) -> Coder:
    return Coder.from_graph(release)


@pytest.mark.slow
def test_whole_instructions_are_not_alternatives(release: Graph) -> None:
    found = Notes(release).instructions("I13.0")
    assert len(found) == 2
    assert all(i.mandatory for i in found)
    assert all(i.alternative for i in Notes(release).instructions("I50.9"))  # a list


@pytest.mark.slow
@pytest.mark.parametrize(
    ("phrase", "codes"),
    [
        # I.C.9.a: hypertension with heart disease, CKD, or both
        ("hypertension with heart failure", ("I11.0", "I50.9")),
        ("heart failure with hypertension", ("I11.0", "I50.9")),
        ("CHF, HTN", ("I11.0", "I50.9")),
        ("cardiomegaly with hypertension", ("I11.9",)),  # I51.7 Excludes1: due to hypertension
        ("hypertension with CKD stage 4", ("I12.9", "N18.4")),
        ("HTN, CKD 4", ("I12.9", "N18.4")),
        ("hypertension and CKD", ("I12.9", "N18.9")),
        ("hypertension with heart failure and CKD stage 3", ("I13.0", "I50.9", "N18.30")),
        ("hypertension, CKD stage 5, heart failure", ("I13.2", "I50.9", "N18.5")),
        (
            "hypertensive heart and chronic kidney disease with heart failure",
            ("I13.0", "I50.9", "N18.9"),
        ),
        # I.C.4.a: diabetes type 2 by default, drug therapy codes
        ("diabetes", ("E11.9",)),
        ("type 2 diabetes on insulin", ("E11.9", "Z79.4")),
        ("type 2 diabetes with oral hypoglycemics", ("E11.9", "Z79.84")),
        ("diabetes on insulin and metformin", ("E11.9", "Z79.4", "Z79.84")),
        ("diabetes with neuropathy on insulin", ("E11.40", "Z79.4")),
        ("type 1 diabetes on insulin", ("E10.9",)),  # E10 has no drug-therapy note
        # I.C.1.d.4: severe sepsis and septic shock
        ("severe sepsis", ("A41.9", "R65.20")),
        ("severe sepsis without septic shock", ("A41.9", "R65.20")),
        ("sepsis without septic shock", ("A41.9",)),  # no main term the phrase says is absent
        ("postmenopausal osteoporosis without current pathological fracture", ("M81.0",)),
        ("septic shock", ("A41.9", "R65.21")),
        ("severe sepsis, septic shock", ("A41.9", "R65.21")),
        ("sepsis due to E. coli with septic shock", ("A41.51", "R65.21")),
        ("pneumonia with severe sepsis", ("A41.9", "R65.20", "J18.9")),
        ("sepsis", ("A41.9",)),
        # I.B.8: acute and chronic
        ("acute on chronic renal failure", ("N17.9", "N18.9")),
        ("acute and chronic sinusitis", ("J01.90", "J32.9")),
        ("acute on chronic blood loss anemia", ("D62", "D50.0")),
        ("acute and chronic respiratory failure", ("J96.20",)),  # a combination code
        ("acute on chronic diastolic heart failure", ("I50.33",)),
        ("acute and chronic cholecystitis", ("K81.2",)),
        # I.B.9 and I.C.19.e.5: every condition coded, in the guidelines' order
        ("COPD with pneumonia", ("J44.9", "J18.9")),
        ("hyperglycemia due to underdosing of insulin", ("R73.9", "T38.3X6A")),
        ("coma due to heroin overdose", ("T40.1X1A", "R40.20")),
        ("acetaminophen overdose with liver failure", ("T39.1X1A", "K72.90")),
        ("rash due to adverse effect of amoxicillin", ("R21", "T36.0X5A")),
        # I.C.19: 7th characters
        ("fracture of right femur", ("S72.91XA",)),
        ("closed fracture of shaft of left tibia, subsequent encounter", ("S82.202D",)),
        ("sequela of fracture of right femur", ("S72.91XS",)),
        ("poisoning by aspirin", ("T39.011A",)),
        ("adverse effect of penicillin", ("T36.0X5A",)),
        ("underdosing of insulin", ("T38.3X6A",)),
    ],
)
def test_release_rule(coder: Coder, phrase: str, codes: tuple[str, ...]) -> None:
    assert coder.code(phrase).codes == codes


@pytest.mark.slow
@pytest.mark.parametrize(
    ("phrase", "codes"),
    [
        # Parts that are no condition of their own are not coded alone.
        ("anogenital warts due to papillomavirus", ("A63.0",)),  # A63.0 names the HPV
        ("leukoplakia of gingiva, lips, tongue", ("K13.21",)),  # sites
        ("adverse effect of captodiame, iniital encounter", ("T43.595A",)),  # a 7th character
        ("juvenile myelomonocytic leukemia with failed remisdion", ("C93.30",)),  # a typo
        (
            "intentional self-poisoning with PCP, meaning pentachlorophenol, herbicide",
            ("T60.3X2A",),
        ),
        ("hip fracture due to fall", ("S72.009A", "W19.XXXA")),  # an external cause
        # Hypertension the I10-I13 rule leaves alone.
        ("hypertension, pulmonary, group 1", ("I27.21",)),
        ("hypertension, emergency", ("I16.1",)),
        ("hypertension complicating childbirth, pre-existing, with heart disease", ("O10.12",)),
    ],
)
def test_release_rule_leaves_alone(coder: Coder, phrase: str, codes: tuple[str, ...]) -> None:
    assert coder.code(phrase).codes == codes


@pytest.mark.slow
def test_release_etiology_is_not_presumed(coder: Coder) -> None:
    code_set = coder.code("Avellis' syndrome")  # G46.8: code first the cerebrovascular disease
    assert code_set.codes == ("G46.8",)
    assert UNRESOLVED in code_set.flags


@pytest.mark.slow
def test_release_rule_traces_cite_their_section(coder: Coder) -> None:
    code_set = coder.code("hypertension with CKD stage 4")
    found = {(s.action, s.guideline) for s in code_set.trace if s.layer == "rules"}
    assert ("hypertension", "I.C.9.a.2") in found
    piece = next(d for d in code_set.diagnoses if d.code == "I10")
    assert any(s.guideline == "I.B.9" for s in piece.trace)  # "hypertension" coded on its own
    sepsis = coder.code("severe sepsis")
    assert {s.guideline for s in sepsis.trace} >= {"I.C.1.d.4"}
    assert [c.role for c in sepsis.sequence] == ["code_first", "diagnosis"]
    drug = coder.code("type 2 diabetes on insulin")
    added = next(c for c in drug.companions if c.code == "Z79.4")
    assert any(s.guideline == "I.C.4.a.3" for s in added.trace)
    default = coder.code("diabetes")
    assert any(s.guideline == "I.C.4.a.2" for s in default.trace)
    both = coder.code("acute on chronic renal failure")
    assert any(s.guideline == "I.B.8" for s in both.trace)


@pytest.mark.slow
def test_release_hypertensive_heart_drop_is_not_flagged(coder: Coder) -> None:
    code_set = coder.code("cardiomegaly with hypertension")
    assert [e.code for e in code_set.excluded] == ["I51.7"]
    assert EXCLUDES1 not in code_set.flags


@pytest.mark.slow
def test_release_unstated_stage_takes_the_unspecified_code(coder: Coder) -> None:
    code_set = coder.code("hypertensive heart and chronic kidney disease with heart failure")
    added = next(c for c in code_set.companions if c.code == "N18.9")
    assert added.status == "added"
    assert "unspecified_chosen" in added.flags
    assert UNRESOLVED not in code_set.flags
    # E11.22's note lists no unspecified stage: left for review.
    assert UNRESOLVED in coder.code("type 2 diabetes with chronic kidney disease").flags
