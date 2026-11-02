from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph
from text2icd.text.model import ParsedPhrase
from text2icd.text.parse import Parser, describe_parsed, parse
from text2icd.text.terms import Terms

# A small stand-in for the graph's terms: main terms that name conditions, body sites, a CMS term
# written with "and", and an Index "with" combination.
TERMS = Terms.build(
    conditions=[
        "Anemia",
        "Anxiety",
        "Ascites",
        "Aspiration",
        "Carcinoma",
        "Cirrhosis",
        "Dehydration",
        "Depression",
        "Diabetes, diabetic",
        "Disease, diseased",
        "Effusion",
        "Embolism",
        "Exacerbation",
        "Failure",
        "Fall",
        "Fibrillation",
        "Flutter",
        "Fracture",
        "Hernia",
        "Hyperlipidemia",
        "Hypertension",
        "Infarct, infarction",
        "Infection",
        "Injury",
        "Lung",
        "Mass",
        "Nausea",
        "Neuropathy",
        "Obstruction",
        "Pain",
        "Pneumonia",
        "Response",
        "Retinopathy",
        "Sepsis",
        "Thrombosis",
        "Ulcer",
        "Vomiting",
        "Behcet's disease",
        "Idiopathic",
    ],
    joined=["Hand, foot and mouth disease"],
    combinations=[("Nausea", "vomiting")],
    sites=["lung", "lobe", "lower", "upper", "liver", "hip", "extremity", "vulva"],
    paths=[("Anemia", "hemolytic"), ("Anemia", "idiopathic"), ("Pneumonia", "aspiration")],
    table_paths=[("Albumin", "human serum")],
)
PARSER = Parser(TERMS)

# (phrase, condition texts, number of diagnoses, relations as (kind, head, linked))
CASES = [
    ("pneumonia", ["pneumonia"], 1, []),
    ("HTN, HLD", ["hypertension", "hyperlipidemia"], 2, []),
    ("HTN; HLD; CAD", ["hypertension", "hyperlipidemia", "coronary artery disease"], 3, []),
    ("HTN and HLD", ["hypertension", "hyperlipidemia"], 2, [("and", 0, 1)]),
    ("HTN/HLD", ["hypertension", "hyperlipidemia"], 2, [("and", 0, 1)]),
    ("anxiety and depression", ["anxiety", "depression"], 2, [("and", 0, 1)]),
    ("afib/flutter", ["atrial fibrillation", "flutter"], 2, [("and", 0, 1)]),
    # kept together: a CMS term with "and", an Index "with" combination, modifiers
    ("nausea and vomiting", ["nausea and vomiting"], 1, []),
    ("N/V", ["nausea and vomiting"], 1, []),
    ("intractable nausea and vomiting", ["intractable nausea and vomiting"], 1, []),
    ("hand, foot and mouth disease", ["hand, foot and mouth disease"], 1, []),
    ("acute and chronic respiratory failure", ["acute and chronic respiratory failure"], 1, []),
    ("right and left hip fracture", ["right and left hip fracture"], 1, []),
    (
        "acute on chronic systolic CHF",
        ["acute on chronic systolic congestive heart failure"],
        1,
        [],
    ),
    ("DM2 without complications", ["type 2 diabetes mellitus without complications"], 1, []),
    ("findings consistent with pneumonia", ["findings pneumonia"], 1, []),
    # linking connectors: one diagnosis
    ("Afib w RVR", ["atrial fibrillation", "rapid ventricular response"], 1, [("with", 0, 1)]),
    (
        "type 2 DM with diabetic CKD stage 3a",
        ["type 2 diabetes mellitus", "diabetic chronic kidney disease stage 3a"],
        1,
        [("with", 0, 1)],
    ),
    ("AKI 2/2 dehydration", ["acute kidney injury", "dehydration"], 1, [("due_to", 0, 1)]),
    ("AKI secondary to dehydration", ["acute kidney injury", "dehydration"], 1, [("due_to", 0, 1)]),
    ("cirrhosis c/b ascites", ["cirrhosis", "ascites"], 1, [("complicated_by", 0, 1)]),
    (
        "AKI in the setting of sepsis",
        ["acute kidney injury", "sepsis"],
        1,
        [("in", 0, 1)],
    ),
    ("AKI on CKD", ["acute kidney injury", "chronic kidney disease"], 1, [("on", 0, 1)]),
    ("hernia without obstruction", ["hernia", "obstruction"], 1, [("without", 0, 1)]),
    (
        "DM2 with neuropathy and retinopathy",
        ["type 2 diabetes mellitus", "neuropathy", "retinopathy"],
        1,
        [("with", 0, 1), ("with", 0, 2), ("and", 1, 2)],
    ),
    (
        "HTN and DM2 with neuropathy",
        ["hypertension", "type 2 diabetes mellitus", "neuropathy"],
        2,
        [("and", 0, 1), ("with", 1, 2)],
    ),
    ("pneumonia, with sepsis", ["pneumonia", "sepsis"], 1, [("with", 0, 1)]),
    # qualifiers and refinements: parts that name no condition
    ("UTI due to E. coli", ["urinary tract infection due to escherichia coli"], 1, []),
    ("ESRD on HD", ["end stage renal disease on hemodialysis"], 1, []),
    ("pneumonia, right lower lobe", ["pneumonia, right lower lobe"], 1, []),
    ("PNA, RLL", ["pneumonia, right lower lobe"], 1, []),
    ("left hip fracture, initial encounter", ["left hip fracture, initial encounter"], 1, []),
    ("pneumonia, left lung", ["pneumonia, left lung"], 1, []),  # lung is a site
    (
        "respiratory failure, requiring intubation",
        ["respiratory failure, requiring intubation"],
        1,
        [],
    ),
    ("DM2, uncontrolled", ["type 2 diabetes mellitus, uncontrolled"], 1, []),
    ("left, hip fracture", ["left, hip fracture"], 1, []),
    # Index paths written with commas, and the "in" subterms
    ("anemia, hemolytic, idiopathic", ["anemia, hemolytic, idiopathic"], 1, []),
    ("pneumonia, aspiration", ["pneumonia, aspiration"], 1, []),
    ("aspiration, pneumonia", ["aspiration", "pneumonia"], 2, []),  # not an Index path
    ("ulcer, vulva, in Behcet's disease", ["ulcer, vulva", "behcet's disease"], 1, [("in", 0, 1)]),
    ("fracture, subsqeuent encounter", ["fracture, subsqeuent encounter"], 1, []),
    (
        "skull fracture, initial encounter for closed fracture",
        ["skull fracture, initial encounter for closed fracture"],
        1,
        [],
    ),
    # status cues start a condition of their own
    (
        "CAD s/p CABG",
        ["coronary artery disease", "coronary artery bypass graft"],
        2,
        [],
    ),
    ("s/p fall with hip fracture", ["fall", "hip fracture"], 1, [("with", 0, 1)]),
    # nothing names a condition: the phrase stays whole
    ("right lower lobe", ["right lower lobe"], 1, []),
    ("", [], 0, []),
    # comparisons
    ("pneumonia vs aspiration", ["pneumonia", "aspiration"], 2, [("versus", 0, 1)]),
    ("PE or DVT", ["pulmonary embolism", "deep vein thrombosis"], 2, [("or", 0, 1)]),
    ("liver mass, possibly HCC", ["liver mass", "hepatocellular carcinoma"], 2, []),
]


@pytest.mark.parametrize(("phrase", "texts", "diagnoses", "relations"), CASES)
def test_split(
    phrase: str, texts: list[str], diagnoses: int, relations: list[tuple[str, int, int]]
) -> None:
    parsed = PARSER.parse(phrase)
    assert [c.text for c in parsed.conditions] == texts
    assert len(parsed.diagnoses) == diagnoses
    assert [(r.kind, r.head, r.linked) for r in parsed.relations] == relations
    for condition in parsed.conditions:
        assert parsed.normalized[condition.start : condition.end]
        assert condition in [parsed.conditions[i] for i in parsed.diagnoses[condition.diagnosis]]


# (phrase, (status, certainty, reportable inpatient) per condition)
JUDGED = [
    ("h/o MI", [("history", "affirmed", True)]),
    ("FHx colon cancer", [("family_history", "affirmed", True)]),
    ("pneumonia, resolved", [("resolved", "affirmed", True)]),
    ("possible pneumonia", [("current", "uncertain", True)]),
    ("pneumonia, likely", [("current", "uncertain", True)]),
    ("no PE", [("current", "negated", False)]),
    ("PE ruled out", [("current", "negated", False)]),
    ("possible PE, ruled out", [("current", "negated", False)]),
    ("no PE or DVT", [("current", "negated", False), ("current", "negated", False)]),
    ("PE and DVT ruled out", [("current", "negated", False), ("current", "negated", False)]),
    (
        "sepsis due to probable UTI",
        [("current", "affirmed", True), ("current", "uncertain", True)],
    ),
    (
        "probable UTI with sepsis",
        [("current", "uncertain", True), ("current", "uncertain", True)],
    ),
    (
        "s/p fall with hip fracture",
        [("status_post", "affirmed", True), ("current", "affirmed", True)],
    ),
    (
        "h/o breast cancer s/p lumpectomy",
        [("history", "affirmed", True), ("status_post", "affirmed", True)],
    ),
    (
        "HTN, h/o MI",
        [("current", "affirmed", True), ("history", "affirmed", True)],
    ),
    (
        "hernia without obstruction",
        [("current", "affirmed", True), ("current", "negated", False)],
    ),
    (
        "pneumonia vs aspiration",
        [("current", "uncertain", True), ("current", "uncertain", True)],
    ),
    (
        "chest pain, r/o MI",
        [("current", "affirmed", True), ("current", "uncertain", True)],
    ),
]


@pytest.mark.parametrize(("phrase", "expected"), JUDGED)
def test_judged(phrase: str, expected: list[tuple[str, str, bool]]) -> None:
    parsed = PARSER.parse(phrase)
    assert [(c.status, c.certainty, c.reportable) for c in parsed.conditions] == expected


def test_outpatient_setting() -> None:
    parsed = PARSER.parse("chest pain, possible MI", setting="outpatient")
    pain, infarction = parsed.conditions
    assert pain.reportable
    assert not infarction.reportable
    assert parsed.setting == "outpatient"
    assert any(step.guideline == "IV.H" for step in parsed.trace)
    inpatient = PARSER.parse("chest pain, possible MI")
    assert all(c.reportable for c in inpatient.conditions)
    assert any(step.guideline == "II.H, III.C" for step in inpatient.trace)
    versus = PARSER.parse("pneumonia vs aspiration")
    assert any(step.guideline == "II.D" for step in versus.trace)
    assert not any(
        step.guideline == "II.D"
        for step in PARSER.parse("pneumonia vs aspiration", "outpatient").trace
    )


def test_cue_words_leave_the_condition_text() -> None:
    parsed = PARSER.parse("possible PE, ruled out")
    (condition,) = parsed.conditions
    assert condition.text == "pulmonary embolism"
    assert [cue.text for cue in condition.cues] == ["possible", "ruled out"]


def test_not_inside_a_condition_qualifies_what_follows() -> None:
    (epilepsy,) = PARSER.parse("epilepsy not intractable").conditions
    assert (epilepsy.text, epilepsy.certainty, epilepsy.reportable) == (
        "epilepsy not intractable",
        "affirmed",
        True,
    )
    (hernia,) = PARSER.parse("inguinal hernia not specified as recurrent").conditions
    assert hernia.reportable
    assert "not specified" in hernia.text
    # A `not` before the condition still negates it.
    assert [c.certainty for c in PARSER.parse("not pneumonia").conditions] == ["negated"]


def test_modifiers_belong_to_their_condition() -> None:
    parsed = PARSER.parse("left hip fracture, initial encounter; DM2")
    fracture, diabetes = parsed.conditions
    assert [(m.kind, m.value) for m in fracture.modifiers] == [
        ("laterality", "left"),
        ("episode", "initial"),
    ]
    assert [(m.kind, m.value) for m in diabetes.modifiers] == [("diabetes_type", "2")]


def test_flags() -> None:
    assert PARSER.parse("MS").flags == ("ambiguous_abbreviation",)
    assert "unknown_abbreviation" in PARSER.parse("XYZ syndrome").flags
    assert PARSER.parse("h/o MI, possible PE").flags == ("history", "uncertain")
    assert PARSER.parse("").flags == ("no_condition",)
    assert PARSER.parse("pneumonia").flags == ()


def test_trace() -> None:
    parsed = PARSER.parse("PNA, RLL; UTI due to E. coli")
    actions = [step.action for step in parsed.trace]
    assert "abbreviation" in actions
    assert "refinement" in actions
    assert "qualifier" in actions
    kept = PARSER.parse("nausea and vomiting").trace
    assert any(step.action == "kept_together" for step in kept)
    split = PARSER.parse("HTN, HLD").trace
    assert any(step.action == "split" and step.detail.startswith('","') for step in split)
    judged = PARSER.parse("h/o MI").trace
    assert judged[-1].detail.startswith('"myocardial infarction": personal history')


def test_without_graph() -> None:
    """Without the graph's terms, any words besides modifiers and cues name a condition."""
    parsed = parse("HTN, HLD")
    assert [c.text for c in parsed.conditions] == ["hypertension", "hyperlipidemia"]
    assert len(parse("pneumonia, right lower lobe").diagnoses) == 2
    assert len(parse("pneumonia, left").diagnoses) == 1
    assert [c.text for c in parse("nausea and vomiting").conditions] == ["nausea", "vomiting"]


def test_serialisation() -> None:
    parsed = PARSER.parse("type 2 DM with diabetic CKD stage 3a, possible UTI")
    again = ParsedPhrase.model_validate_json(parsed.model_dump_json())
    assert again == parsed
    lines = describe_parsed(parsed)
    assert lines[0].startswith("normalized: type 2 diabetes mellitus")
    assert "diagnosis 2:" in lines
    assert any(line.startswith("relation: [1] with [2]") for line in lines)


def test_terms_build() -> None:
    assert TERMS.is_condition("pneumonia")
    assert TERMS.is_condition("right lower lobe pneumonia")
    assert not TERMS.is_condition("right lower lobe")
    assert not TERMS.is_condition("lung")  # a condition main term, but only a site
    assert TERMS.is_condition("pneumonias")  # lenient forms
    assert TERMS.combines("nausea", "vomiting")
    assert TERMS.combines("vomiting", "nausea")
    assert not TERMS.combines("hypertension", "hyperlipidemia")
    words = ["hand", "foot", "and", "mouth", "disease"]
    assert TERMS.joins(words, 1)  # the comma after "hand"
    assert TERMS.joins(words, 2)  # "and"
    assert not TERMS.joins(["htn", "and", "hld"], 1)
    assert TERMS.continues("anemia", "hemolytic")
    assert TERMS.continues("anemia, hemolytic", "idiopathic")  # the main term starts the phrase
    assert not TERMS.continues("hemolytic", "anemia")
    assert TERMS.continues("accidental poisoning by albumin", "human serum")  # a table ends it
    assert not TERMS.continues("albumin poisoning", "human serum")


def test_terms_from_fixture_graph(graph: Graph) -> None:
    terms = Terms.from_graph(graph)
    assert terms.is_condition("pneumonia")
    assert terms.is_condition("pyelitis")
    assert "pneumonia" in terms.vocabulary
    assert terms.combines("pneumonia", "lung abscess")  # Pneumonia > with > lung abscess
    parser = Parser.from_graph(graph)
    parsed = parser.parse("pneumonia and lung abscess")
    assert len(parsed.diagnoses) == 1


def test_parse_command(graph_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["parse", "pneumonia and lung abscess", "--graph", str(graph_file)]
    )
    assert result.exit_code == 0, result.output
    assert "[1] pneumonia and lung abscess  (current, affirmed)" in result.output
    assert "kept_together" in result.output
    linked = runner.invoke(cli.app, ["parse", "AKI 2/2 dehydration", "--no-graph"])
    assert "normalized: acute kidney injury due to dehydration" in linked.output
    assert "relation: [1] due_to [2]" in linked.output
    as_json = runner.invoke(cli.app, ["parse", "HTN", "--json", "--no-graph"])
    assert ParsedPhrase.model_validate_json(as_json.output).conditions[0].text == "hypertension"
    outpatient = runner.invoke(
        cli.app, ["parse", "possible MI", "--setting", "outpatient", "--no-graph"]
    )
    assert "not coded" in outpatient.output
    bad = runner.invoke(cli.app, ["parse", "MI", "--setting", "clinic", "--no-graph"])
    assert bad.exit_code == 2


@pytest.mark.slow
def test_fy2027_terms(release_graph: Callable[[str], BuildReport]) -> None:
    with Graph(release_graph("fy2027").path) as graph:
        parser = Parser.from_graph(graph)
    expected = {
        "nausea and vomiting": 1,  # Nausea > with vomiting
        "hand, foot and mouth disease": 1,  # a Tabular inclusion term
        "pneumonia, right lower lobe": 1,
        "pneumonia, left lower extremity cellulitis": 2,
        "HTN, HLD": 2,
        "type 2 DM with diabetic CKD stage 3a": 1,
        "UTI due to E. coli": 1,
        "HTN and HLD": 2,
        "CAD s/p CABG": 2,
    }
    found = {phrase: len(parser.parse(phrase).diagnoses) for phrase in expected}
    assert found == expected
