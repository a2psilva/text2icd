import pytest

from text2icd.text.normalize import clean, normalize
from text2icd.trace import TraceStep

# (phrase as written, normalised text)
CASES = [
    # clean: Unicode, dashes, quotes, spacing, list markers, punctuation at the ends
    ("Pneumonia", "pneumonia"),
    ("  Acute   kidney\tinjury  ", "acute kidney injury"),
    ("Crohn\u2019s disease", "crohn's disease"),
    ("acute\u2013on\u2013chronic heart failure", "acute-on-chronic heart failure"),
    ("\u201cpneumonia\u201d", '"pneumonia"'),
    ("\uff28\uff34\uff2e", "htn"),  # full-width letters
    ("hyper\u00adtension", "hypertension"),  # soft hyphen
    ("anemia\u200b", "anemia"),
    ("HTN,HLD", "htn, hld"),
    ("HTN , HLD ;CAD", "htn, hld; cad"),
    ("WBC 1,000", "wbc 1,000"),
    ("Pneumonia ( RLL )", "pneumonia (rll)"),
    ("1. Sepsis", "sepsis"),
    ("2) Hypotension", "hypotension"),
    ("- Hyperkalemia", "hyperkalemia"),
    ("* Delirium.", "delirium"),
    ("a) anemia", "anemia"),
    ("UTI.", "uti"),
    ("Hypokalemia;", "hypokalemia"),
    ("?PE", "?pe"),
    ("PE?", "pe?"),
    ("1.5 cm lung nodule", "1.5 cm lung nodule"),
    # shorthand
    ("CHF w/ preserved EF", "chf with preserved ef"),
    ("PNA w/sepsis", "pna with sepsis"),
    ("DM2 w/o complications", "dm type 2 without complications"),
    ("hernia w/out obstruction", "hernia without obstruction"),
    ("PNA w/oxygen requirement", "pna with oxygen requirement"),
    ("cirrhosis c/b ascites", "cirrhosis complicated by ascites"),
    ("s/p CABG", "status post cabg"),
    ("S/P appendectomy", "status post appendectomy"),
    ("status-post TKA", "status post tka"),
    ("h/o DVT", "history of dvt"),
    ("Hx of stroke", "history of stroke"),
    ("hx DVT", "history dvt"),
    ("FHx colon cancer", "family history colon cancer"),
    ("fam hx of CAD", "family history of cad"),
    ("r/o MI", "rule out mi"),
    ("AKI 2/2 dehydration", "aki due to dehydration"),
    ("AKI 2/2 to dehydration", "aki due to dehydration"),
    ("anemia d/t blood loss", "anemia due to blood loss"),
    ("hypercalcemia 2\u00b0 to malignancy", "hypercalcemia secondary to malignancy"),
    ("HTN 2' to RAS", "htn secondary to ras"),
    ("anemia sec. to CKD", "anemia secondary to ckd"),
    ("b/l LE edema", "bilateral le edema"),
    ("bilat. pleural effusions", "bilateral pleural effusions"),
    ("hyponatremia a/w SIADH", "hyponatremia associated with siadh"),
    ("rash assoc w/ fever", "rash associated with fever"),
    ("findings c/w pneumonia", "findings consistent with pneumonia"),
    ("anxiety r/t pain", "anxiety related to pain"),
    ("seizure d/o", "seizure disorder"),
    ("PNA vs. aspiration", "pna vs aspiration"),
    ("HTN & HLD", "htn and hld"),
    ("C diff neg", "c diff negative"),
    # no shorthand inside words or longer forms
    ("PMHx of asthma", "pmhx of asthma"),
    ("show/o", "show/o"),
    ("D&C", "d&c"),
    # numerals: stages, CKD stages, diabetes types
    ("CKD stage III", "ckd stage 3"),
    ("CKD stage IIIb", "ckd stage 3b"),
    ("pressure ulcer stage IV", "pressure ulcer stage 4"),
    ("stage II pressure injury of sacrum", "stage 2 pressure injury of sacrum"),
    ("CKD 3", "ckd stage 3"),
    ("CKD IV", "ckd stage 4"),
    ("CKD-3a", "ckd stage 3a"),
    ("chronic kidney disease, stage 5", "chronic kidney disease stage 5"),
    ("chronic kidney disease stage 3a", "chronic kidney disease stage 3a"),
    ("type II diabetes", "type 2 diabetes"),
    ("Type I DM", "type 1 dm"),
    ("diabetes mellitus II", "diabetes mellitus type 2"),
    ("diabetes mellitus, type 2", "diabetes mellitus type 2"),
    ("DM 2", "dm type 2"),
    ("diabetes 2 years", "diabetes 2 years"),
    ("CKD 2 years", "ckd 2 years"),
    ("NYHA class III", "nyha class iii"),
    ("factor V Leiden", "factor v leiden"),
    ("CKD in the setting of DM", "ckd in the setting of dm"),
]


@pytest.mark.parametrize(("phrase", "expected"), CASES)
def test_normalize(phrase: str, expected: str) -> None:
    result = normalize(phrase)
    assert result.text == expected
    assert result.original == phrase


def test_trace_records_each_rewrite() -> None:
    result = normalize("  DM II w/ CKD III ")
    assert result.text == "dm type 2 with ckd stage 3"
    assert [step.action for step in result.trace] == ["clean", "shorthand", "numeral", "numeral"]
    assert result.trace[1] == TraceStep(layer="text", action="shorthand", detail='"w/" -> "with"')
    assert result.trace[2].detail == '"ckd iii" -> "ckd stage 3"'
    assert result.trace[3].detail == '"dm ii" -> "dm type 2"'


def test_nothing_to_record() -> None:
    assert normalize("pneumonia").trace == ()
    assert normalize("Pneumonia").trace == ()  # case folding is not a rewrite


def test_clean_is_idempotent() -> None:
    for phrase, _ in CASES:
        once = clean(phrase)
        assert clean(once) == once
