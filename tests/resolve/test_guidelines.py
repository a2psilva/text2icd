"""Every guideline section the coder cites, and every worked example of `docs/coder-workflow.md`,
coded on the FY2027 release (task 6.10).

`test_every_cited_section_has_a_case` reads the source for the sections its trace steps cite, so
a new rule cannot be added without a case here."""

import io
import re
import tokenize
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load
from text2icd.resolve.coder import Coder, CodeSet
from text2icd.resolve.walk import DEFAULT_CODE
from text2icd.text.model import Setting

SOURCE = Path(__file__).resolve().parents[2] / "src" / "text2icd"
_SECTION = re.compile(r'"((?:I|II|III|IV)\.[A-Z][^"]*)"')

# section cited -> (phrase, codes in order, setting)
CASES: dict[str, tuple[str, tuple[str, ...], Setting]] = {
    "I.A.5": ("fracture of right femur", ("S72.91XA",), "inpatient"),
    "I.A.9.b": (
        "hypertensive heart and chronic kidney disease with heart failure",
        ("I13.0", "I50.9", "N18.9"),
        "inpatient",
    ),
    "I.A.9.b, I.B.18": ("hip fracture", ("S72.009A",), "inpatient"),
    "I.A.12.a": ("syncope and collapse, orthostatic hypotension", ("I95.1",), "inpatient"),
    "I.A.13": ("anemia in chronic kidney disease", ("N18.9", "D63.1"), "inpatient"),
    "I.A.15": ("type 2 diabetes, CKD stage 3a", ("E11.22", "N18.31"), "inpatient"),
    "I.A.16": ("cardiomegaly", ("I51.7",), "inpatient"),
    "I.A.18": ("appendicitis", ("K37",), "inpatient"),
    "I.B.1": ("accidental poisoning by 1-propanol, initial encounter", ("T51.3X1A",), "inpatient"),
    "I.B.7": ("type 2 DM with diabetic CKD stage 3a", ("E11.22", "N18.31"), "inpatient"),
    "I.B.8": ("acute on chronic renal failure", ("N17.9", "N18.9"), "inpatient"),
    "I.B.9": ("hypertension with CKD stage 4", ("I12.9", "N18.4"), "inpatient"),
    "I.B.10": ("sequela of laceration of face with keloid", ("L91.0", "S01.81XS"), "inpatient"),
    "I.C.1.d.4": ("severe sepsis", ("A41.9", "R65.20"), "inpatient"),
    "I.C.2": ("malignant neoplasm of upper lobe of lung", ("C34.10",), "inpatient"),
    "I.C.4.a.2": ("diabetes", ("E11.9",), "inpatient"),
    "I.C.4.a.3": ("type 2 diabetes on insulin", ("E11.9", "Z79.4"), "inpatient"),
    "I.C.9.a.1": ("heart failure with hypertension", ("I11.0", "I50.9"), "inpatient"),
    "I.C.9.a.2": ("HTN, CKD 4", ("I12.9", "N18.4"), "inpatient"),
    "I.C.9.a.3": (
        "hypertension with heart failure and CKD stage 3",
        ("I13.0", "I50.9", "N18.30"),
        "inpatient",
    ),
    "I.C.19.a": ("closed fracture of shaft of left tibia", ("S82.202A",), "inpatient"),
    "I.C.19.c.1": ("hip fracture due to fall", ("S72.009A", "W19.XXXA"), "inpatient"),
    "I.C.19.e": ("adverse effect of penicillin", ("T36.0X5A",), "inpatient"),
    "I.C.19.e.5(a)": ("digoxin toxicity", ("T46.0X5A",), "inpatient"),
    "I.C.19.e.5(b)": ("poisoning by aspirin", ("T39.011A",), "inpatient"),
    "I.C.19.e.5(c)": (
        "hyperglycemia due to underdosing of insulin",
        ("R73.9", "T38.3X6A"),
        "inpatient",
    ),
    "I.C.20": ("fracture of right femur due to fall", ("S72.91XA", "W19.XXXA"), "inpatient"),
    "I.C.21.c.3": ("status post kidney transplant", ("Z94.0",), "inpatient"),
    "I.C.21.c.4": ("family history of lung cancer", ("Z80.1",), "inpatient"),
    "II.D": ("pneumonia versus bronchitis", ("J18.9", "J40"), "inpatient"),
    "II.H, III.C": ("probable pneumonia", ("J18.9",), "inpatient"),
    "IV.H": ("probable pneumonia", (), "outpatient"),  # not coded: its symptoms would be
}


def cited_in_source() -> set[str]:
    """The guideline sections written as string literals in the package (not in comments)."""
    found: set[str] = set()
    for path in SOURCE.rglob("*.py"):
        tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        for token in tokens:
            if token.type == tokenize.STRING and (match := _SECTION.fullmatch(token.string)):
                found.add(match.group(1))
    return found


def cited(code_set: CodeSet) -> set[str]:
    """Every guideline section a code set's trace cites."""
    steps = [*code_set.parsed.trace, *code_set.trace]
    for coded in (*code_set.diagnoses, *code_set.external_causes):
        steps += coded.trace
    for companion in code_set.companions:
        steps += companion.trace
    return {step.guideline for step in steps if step.guideline}


def test_every_cited_section_has_a_case() -> None:
    sections = cited_in_source()
    assert len(sections) > 25  # the scan finds them
    assert sorted(sections - set(CASES)) == []
    assert sorted(set(CASES) - sections) == []  # no case for a section nothing cites


@pytest.fixture(scope="module")
def release(release_graph: Callable[[str], BuildReport]) -> Iterator[Graph]:
    with load(path=release_graph("fy2027").path) as graph:
        yield graph


@pytest.fixture(scope="module")
def coders(release: Graph) -> dict[str, Coder]:
    return {
        "inpatient": Coder.from_graph(release),
        "outpatient": Coder.from_graph(release, setting="outpatient"),
    }


@pytest.mark.slow
@pytest.mark.parametrize("section", sorted(CASES))
def test_release_section(coders: dict[str, Coder], section: str) -> None:
    phrase, codes, setting = CASES[section]
    code_set = coders[setting].code(phrase)
    assert code_set.codes == codes
    assert section in cited(code_set)


# The worked examples of docs/coder-workflow.md, and the README example.


@pytest.mark.slow
def test_worked_example_default_code(coders: dict[str, Coder]) -> None:
    code_set = coders["inpatient"].code("Appendicitis")
    assert code_set.codes == ("K37",)
    assert DEFAULT_CODE in code_set.diagnoses[0].flags


@pytest.mark.slow
def test_worked_example_subterm_descent(coders: dict[str, Coder]) -> None:
    code_set = coders["inpatient"].code("aspiration pneumonia due to food")
    assert code_set.codes == ("J69.0",)
    assert code_set.diagnoses[0].candidate.path == "Pneumonia > aspiration > due to > food"


@pytest.mark.slow
def test_worked_example_combination_and_use_additional_code(coders: dict[str, Coder]) -> None:
    code_set = coders["inpatient"].code("type 2 DM with diabetic CKD stage 3a")
    assert [(c.code, c.role, c.source) for c in code_set.sequence] == [
        ("E11.22", "diagnosis", None),
        ("N18.31", "use_additional", "E11.22"),
    ]
    assert code_set.diagnoses[0].candidate.path.startswith("Diabetes, diabetic > type 2 > with")
    added = code_set.companions[0]
    assert added.trace[0].guideline == "I.B.7"
    assert any("stage 3a" in step.detail for step in added.trace)


@pytest.mark.slow
def test_worked_example_etiology_and_manifestation(coders: dict[str, Coder]) -> None:
    code_set = coders["inpatient"].code("amyloid heart disease")
    assert [(c.code, c.role) for c in code_set.sequence] == [
        ("E85.4", "diagnosis"),
        ("I43", "manifestation"),
    ]


@pytest.mark.slow
def test_worked_example_generated_seventh_character(
    release: Graph, coders: dict[str, Coder]
) -> None:
    phrase = (
        "mild nonproliferative diabetic retinopathy with macular edema due to underlying "
        "condition, right eye"
    )
    assert coders["inpatient"].code(phrase).codes == ("E08.3211",)
    assert release.code("E08.321").seventh_from == "E08.32"  # the inherited definition


@pytest.mark.slow
def test_worked_example_placeholder_and_seventh_character(coders: dict[str, Coder]) -> None:
    code_set = coders["inpatient"].code("accidental poisoning by 1-propanol, initial encounter")
    assert code_set.codes == ("T51.3X1A",)
    assert code_set.diagnoses[0].candidate.book == "drug"
