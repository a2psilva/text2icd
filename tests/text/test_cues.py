import pytest

from text2icd.text.abbreviations import default_lexicon
from text2icd.text.cues import cues, judge
from text2icd.text.normalize import normalize


def read(phrase: str) -> str:
    return normalize(phrase, default_lexicon()).text


# (phrase, cues as (kind, position, words), status, certainty)
CASES = [
    # status (4.6)
    ("h/o DVT", [("history", "before", "history of")], "history", "affirmed"),
    ("history of MI", [("history", "before", "history of")], "history", "affirmed"),
    ("Hx of stroke", [("history", "before", "history of")], "history", "affirmed"),
    (
        "personal history of breast cancer",
        [("history", "before", "personal history of")],
        "history",
        "affirmed",
    ),
    ("remote history of TB", [("history", "before", "remote history of")], "history", "affirmed"),
    (
        "past medical history of asthma",
        [("history", "before", "past medical history of")],
        "history",
        "affirmed",
    ),
    ("prior CVA", [("history", "before", "prior")], "history", "affirmed"),
    ("previous MI", [("history", "before", "previous")], "history", "affirmed"),
    ("former smoker", [("history", "before", "former")], "history", "affirmed"),
    (
        "FHx colon cancer",
        [("family_history", "before", "family history")],
        "family_history",
        "affirmed",
    ),
    (
        "family history of CAD",
        [("family_history", "before", "family history of")],
        "family_history",
        "affirmed",
    ),
    (
        "fam hx of breast cancer",
        [("family_history", "before", "family history of")],
        "family_history",
        "affirmed",
    ),
    ("s/p CABG", [("status_post", "before", "status post")], "status_post", "affirmed"),
    (
        "status post appendectomy",
        [("status_post", "before", "status post")],
        "status_post",
        "affirmed",
    ),
    ("pneumonia, resolved", [("resolved", "after", "resolved")], "resolved", "affirmed"),
    ("AKI - now resolved", [("resolved", "after", "now resolved")], "resolved", "affirmed"),
    ("hyponatremia (resolved)", [("resolved", "after", "resolved")], "resolved", "affirmed"),
    ("resolved delirium", [("resolved", "before", "resolved")], "resolved", "affirmed"),
    ("fracture, healed", [("resolved", "after", "healed")], "resolved", "affirmed"),
    ("known CAD", [], "current", "affirmed"),
    ("resolving pneumonia", [], "current", "affirmed"),  # still present
    ("diabetes in remission", [], "current", "affirmed"),  # remission is a modifier, not resolved
    # negation (4.7)
    ("no PE", [("negated", "before", "no")], "current", "negated"),
    ("no evidence of DVT", [("negated", "before", "no evidence of")], "current", "negated"),
    ("no signs of infection", [("negated", "before", "no signs of")], "current", "negated"),
    ("denies chest pain", [("negated", "before", "denies")], "current", "negated"),
    ("negative for malignancy", [("negated", "before", "negative for")], "current", "negated"),
    ("ruled out MI", [("negated", "before", "ruled out")], "current", "negated"),
    ("MI ruled out", [("negated", "after", "ruled out")], "current", "negated"),
    ("PE was ruled out", [("negated", "after", "was ruled out")], "current", "negated"),
    ("C diff neg", [("negated", "after", "negative")], "current", "negated"),
    ("PE unlikely", [("negated", "after", "unlikely")], "current", "negated"),
    ("unlikely PE", [("negated", "before", "unlikely")], "current", "negated"),
    ("PE excluded", [("negated", "after", "excluded")], "current", "negated"),
    ("no history of DVT", [("negated", "before", "no history of")], "current", "negated"),
    # uncertainty (4.7, Guideline II.H terms)
    ("possible PNA", [("uncertain", "before", "possible")], "current", "uncertain"),
    ("probable UTI", [("uncertain", "before", "probable")], "current", "uncertain"),
    ("suspected sepsis", [("uncertain", "before", "suspected")], "current", "uncertain"),
    ("likely viral gastroenteritis", [("uncertain", "before", "likely")], "current", "uncertain"),
    ("questionable seizure", [("uncertain", "before", "questionable")], "current", "uncertain"),
    ("r/o MI", [("uncertain", "before", "rule out")], "current", "uncertain"),
    (
        "still to be ruled out: PE",
        [("uncertain", "before", "still to be ruled out")],
        "current",
        "uncertain",
    ),
    (
        "findings c/w pneumonia",
        [("uncertain", "before", "consistent with")],
        "current",
        "uncertain",
    ),
    (
        "compatible with sarcoidosis",
        [("uncertain", "before", "compatible with")],
        "current",
        "uncertain",
    ),
    ("concern for endocarditis", [("uncertain", "before", "concern for")], "current", "uncertain"),
    ("presumed viral meningitis", [("uncertain", "before", "presumed")], "current", "uncertain"),
    ("cannot rule out PE", [("uncertain", "before", "cannot rule out")], "current", "uncertain"),
    ("PE not ruled out", [("uncertain", "after", "not ruled out")], "current", "uncertain"),
    ("pneumonia, likely", [("uncertain", "after", "likely")], "current", "uncertain"),
    ("sepsis, presumed", [("uncertain", "after", "presumed")], "current", "uncertain"),
    ("?PE", [("uncertain", "before", "?")], "current", "uncertain"),
    ("PE?", [("uncertain", "after", "?")], "current", "uncertain"),
    ("PE (?)", [("uncertain", "after", "(?)")], "current", "uncertain"),
    ("query sepsis", [("uncertain", "before", "query")], "current", "uncertain"),
    # combinations: negation wins, status and certainty together
    (
        "possible PE, ruled out",
        [("uncertain", "before", "possible"), ("negated", "after", "ruled out")],
        "current",
        "negated",
    ),
    (
        "possible history of MI",
        [("uncertain", "before", "possible"), ("history", "before", "history of")],
        "history",
        "uncertain",
    ),
    (
        "history of pneumonia, resolved",
        [("history", "before", "history of"), ("resolved", "after", "resolved")],
        "history",
        "affirmed",
    ),
    ("pneumonia", [], "current", "affirmed"),
    ("unlikelihood", [], "current", "affirmed"),
]


@pytest.mark.parametrize(("phrase", "expected", "status", "certainty"), CASES)
def test_cues(
    phrase: str, expected: list[tuple[str, str, str]], status: str, certainty: str
) -> None:
    found = cues(read(phrase))
    assert [(c.kind, c.position, c.text) for c in found] == expected
    judgement = judge(found)
    assert (judgement.status, judgement.certainty) == (status, certainty)


@pytest.mark.parametrize(
    ("phrase", "inpatient", "outpatient"),
    [
        ("pneumonia", True, True),
        ("possible pneumonia", True, False),  # II.H, III.C against IV.H
        ("no pneumonia", False, False),
        ("history of pneumonia", True, True),  # a history code
        ("pneumonia, resolved", True, True),
    ],
)
def test_setting(phrase: str, inpatient: bool, outpatient: bool) -> None:
    found = cues(read(phrase))
    assert judge(found, "inpatient").reportable is inpatient
    assert judge(found, "outpatient").reportable is outpatient


def test_judgement_trace() -> None:
    inpatient = judge(cues("possible pneumonia"), "inpatient").trace
    assert [(s.action, s.guideline) for s in inpatient] == [("uncertain", "II.H, III.C")]
    outpatient = judge(cues("possible pneumonia"), "outpatient").trace
    assert outpatient[0].guideline == "IV.H"
    assert "symptoms" in outpatient[0].detail
    history = judge(cues("history of stroke")).trace
    assert [(s.action, s.guideline) for s in history] == [("history", "I.C.21.c.4")]
    status = judge(cues("status post cabg")).trace
    assert status[0].guideline == "I.C.21.c.3"
    resolved = judge(cues("pneumonia, resolved")).trace
    assert "not in remission" in resolved[0].detail
    assert judge(cues("no pe")).trace[0].action == "negated"
    assert judge(cues("pneumonia")).trace == ()


def test_spans() -> None:
    text = "possible pulmonary embolism, ruled out"
    possible, ruled_out = cues(text)
    assert text[possible.start : possible.end] == "possible"
    assert text[ruled_out.start : ruled_out.end] == "ruled out"
    assert possible.family == ruled_out.family == "certainty"
