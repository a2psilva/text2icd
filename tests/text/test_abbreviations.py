from collections.abc import Callable

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph
from text2icd.text.abbreviations import Lexicon, LexiconError, default_lexicon, fold
from text2icd.text.normalize import normalize
from text2icd.text.terms import vocabulary, words

LEXICON = default_lexicon()

# (phrase, normalised text with the shipped lexicon)
CASES = [
    ("HTN", "hypertension"),
    ("htn", "hypertension"),
    ("Htn", "hypertension"),
    ("HTN, HLD, CAD", "hypertension, hyperlipidemia, coronary artery disease"),
    ("HTN/HLD", "hypertension/hyperlipidemia"),
    ("s/p CABG", "status post coronary artery bypass graft"),
    ("Afib w RVR", "atrial fibrillation with rapid ventricular response"),
    ("A-fib", "atrial fibrillation"),
    ("a fib", "atrial fibrillation"),
    ("AFib", "atrial fibrillation"),
    ("CHF exacerbation", "congestive heart failure exacerbation"),
    ("HFrEF", "heart failure with reduced ejection fraction"),
    ("HFpEF", "heart failure with preserved ejection fraction"),
    ("NSTEMI", "non-st elevation myocardial infarction"),
    ("non-STEMI", "non-st elevation myocardial infarction"),
    ("AKI on CKD", "acute kidney injury on chronic kidney disease"),
    ("CKD3", "chronic kidney disease stage 3"),
    ("CKD-4", "chronic kidney disease stage 4"),
    ("CKD IIIb", "chronic kidney disease stage 3b"),
    ("ESRD on HD", "end stage renal disease on hemodialysis"),
    ("DM2", "type 2 diabetes mellitus"),
    ("T2DM w/ neuropathy", "type 2 diabetes mellitus with neuropathy"),
    ("DM II", "diabetes mellitus type 2"),
    ("IDDM", "insulin dependent diabetes mellitus"),
    ("DKA", "diabetic ketoacidosis"),
    ("COPD exac", "chronic obstructive pulmonary disease exac"),
    ("PNA", "pneumonia"),
    ("RLL PNA", "right lower lobe pneumonia"),
    ("UTI", "urinary tract infection"),
    ("UTI (E. coli)", "urinary tract infection (escherichia coli)"),
    ("C. diff colitis", "clostridioides difficile colitis"),
    ("C diff colitis", "clostridioides difficile colitis"),
    ("cdiff", "clostridioides difficile"),
    ("H. pylori gastritis", "helicobacter pylori gastritis"),
    ("Hep C cirrhosis", "hepatitis c cirrhosis"),
    ("HepC", "hepatitis c"),
    ("HCV", "hepatitis c"),
    ("EtOH withdrawal", "alcohol withdrawal"),
    ("ETOH abuse", "alcohol abuse"),
    ("N/V", "nausea and vomiting"),
    ("N/V/D", "nausea, vomiting and diarrhea"),
    ("hypoK", "hypokalemia"),
    ("HypoNa", "hyponatremia"),
    ("GIB", "gastrointestinal bleeding"),
    ("UGIB", "upper gastrointestinal bleeding"),
    ("SBO", "small bowel obstruction"),
    ("R hip fx", "right hip fracture"),
    ("L knee OA", "left knee osteoarthritis"),
    ("LLE cellulitis", "left lower extremity cellulitis"),
    ("multiple PEs", "multiple pulmonary embolism"),
    ("PVCs", "premature ventricular contraction"),
    ("COVID pneumonia", "covid-19 pneumonia"),
    ("COVID-19 pneumonia", "covid-19 pneumonia"),
    ("ACUTE MI", "acute myocardial infarction"),
    ("vit D def", "vitamin d deficiency"),
    ("SOB", "shortness of breath"),
    ("AMS", "altered mental status"),
    ("MRSA bacteremia", "mrsa bacteremia"),  # an Index main term, left as written
    # cased entries: only in capitals, and not in a phrase written all in capitals
    ("ALL", "acute lymphoblastic leukemia"),
    ("ALL in remission", "acute lymphoblastic leukemia in remission"),
    ("all", "all"),
    ("CAP", "community acquired pneumonia"),
    ("knee cap fracture", "knee cap fracture"),
    ("severe AS", "severe aortic stenosis"),
    ("SEPSIS AS CAUSE", "sepsis as cause"),
    ("SOB AND CHEST PAIN", "sob and chest pain"),
    ("breast CA", "breast cancer"),
    ("Ca deficiency", "ca deficiency"),
    ("PAD", "peripheral arterial disease"),
    ("heating pad burn", "heating pad burn"),
    # never inside words, before "-" and a digit, or for single letters not followed by a space
    ("pes planus", "pes planus"),
    ("L5 radiculopathy", "l5 radiculopathy"),
    ("L-spine stenosis", "l-spine stenosis"),
    ("HIV-1", "hiv-1"),
    ("hypertensive", "hypertensive"),
    # ambiguous entries: hints, else the first expansion
    ("altered MS", "altered mental status"),
    ("MS changes", "mental status changes"),
    ("MS flare", "multiple sclerosis flare"),
    ("severe MS, mitral valve", "severe mitral stenosis, mitral valve"),
    ("MS", "multiple sclerosis"),
    ("ARF", "acute renal failure"),
    ("hypoxic ARF", "hypoxic acute respiratory failure"),
    ("ARF requiring intubation", "acute respiratory failure requiring intubation"),
    ("PD", "parkinson's disease"),
    ("ESRD on PD", "end stage renal disease on peritoneal dialysis"),
    ("AMA", "advanced maternal age"),
    ("left AMA", "left against medical advice"),
    ("CP", "chest pain"),
    ("spastic CP", "spastic cerebral palsy"),
    ("SI joint dysfunction", "sacroiliac joint dysfunction"),
    ("SI", "suicidal ideation"),
    ("PCP", "pneumocystis pneumonia"),
    ("PCP abuse", "phencyclidine abuse"),
    ("GBS positive", "group b streptococcus positive"),
    ("AD", "alzheimer's disease"),
    ("AD flare, eczema", "atopic dermatitis flare, eczema"),
]


@pytest.mark.parametrize(("phrase", "expected"), CASES)
def test_expansions(phrase: str, expected: str) -> None:
    assert normalize(phrase, LEXICON).text == expected


@pytest.mark.parametrize(
    ("phrase", "flagged"),
    [
        ("MS", True),
        ("altered MS", False),  # resolved by a hint
        ("HTN", False),
        ("ARF", True),
        ("hypoxic ARF", False),
    ],
)
def test_ambiguous_flag(phrase: str, flagged: bool) -> None:
    result = normalize(phrase, LEXICON)
    assert ("ambiguous_abbreviation" in {s.action for s in result.trace}) is flagged
    assert all(e.ambiguous == (e.chosen_by != "only") for e in result.expansions)


def test_expansion_records() -> None:
    result = normalize("altered MS, HTN", LEXICON)
    ms, htn = result.expansions
    assert (ms.written, ms.key, ms.expansion, ms.chosen_by, ms.hint) == (
        "MS",
        "MS",
        "mental status",
        "hint",
        "altered",
    )
    assert ms.alternatives == ("multiple sclerosis", "mitral stenosis")
    assert (htn.expansion, htn.chosen_by, htn.alternatives) == ("hypertension", "only", ())
    details = [step.detail for step in result.trace if step.action == "abbreviation"]
    assert details == [
        '"MS" -> "mental status" (hint "altered"; also: multiple sclerosis, mitral stenosis)',
        '"HTN" -> "hypertension"',
    ]
    default = normalize("MS", LEXICON).trace[0]
    assert default.action == "ambiguous_abbreviation"
    assert default.detail.endswith("(no hint; also: mitral stenosis, mental status)")


@pytest.mark.parametrize(
    ("phrase", "unknown"),
    [
        ("XYZ syndrome", ("XYZ",)),
        ("HTN, QRS widening", ("QRS",)),
        ("CKD3 2/2 HTN", ()),
        ("CKD stage IIIb", ()),
        ("multiple PEs", ()),
        ("PNEUMONIA", ()),  # all capitals: nothing looks like an abbreviation
        ("Pneumonia NOS", ()),
        ("Pneumonia", ()),
        ("XYZ and XYZ", ("XYZ",)),
    ],
)
def test_unknown(phrase: str, unknown: tuple[str, ...]) -> None:
    assert normalize(phrase, LEXICON).unknown == unknown


def test_unknown_with_vocabulary() -> None:
    assert normalize("STEMI, IgA nephropathy", LEXICON).unknown == ("STEMI", "IgA")
    known = normalize("STEMI, IgA nephropathy", LEXICON, vocabulary={"stemi", "iga"})
    assert known.unknown == ()
    assert any(step.action == "unknown_abbreviation" for step in normalize("XYZ", LEXICON).trace)


def test_shouting() -> None:
    assert LEXICON.shouting("PNEUMONIA WITH SEPSIS")
    assert not LEXICON.shouting("HTN, HLD, CAD")  # abbreviations only
    assert not LEXICON.shouting("Pneumonia AS")
    assert not LEXICON.shouting("NSTEMI")


def test_fold_and_lookup() -> None:
    assert fold("A-fib") == fold("AFib") == fold("a fib") == "afib"
    assert fold("C. diff") == "cdiff"
    assert "afib" in LEXICON
    assert "A FIB" in LEXICON
    assert "xyz" not in LEXICON
    entry = LEXICON.get("ms")
    assert entry is not None
    assert entry.ambiguous
    assert entry.cased


def test_shipped_lexicon() -> None:
    assert len(LEXICON) >= 300
    for entry in LEXICON.entries.values():
        assert entry.key not in {"NOS", "NEC"}  # ICD conventions, not abbreviations to expand
        for expansion in entry.expansions:
            assert expansion == expansion.strip()
            assert "  " not in expansion


LEXICON_HEADER = "schema = 1\n[abbreviations]\n"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('A-fib = "atrial fibrillation"\nAFib = "atrial fibrillation"\n', "same key"),
        ("X = 1\n", "a string or a table"),
        ('X = { expansions = ["a"], colour = "red" }\n', "unknown fields"),
        ("X = { expansions = [] }\n", "at least one expansion"),
        ('X = { expansions = ["a", "a"] }\n', "repeated expansion"),
        ('X = { expansions = ["a", "b"], hints.c = ["x"] }\n', "unknown expansions"),
        ('X = { expansions = ["a"], hints.a = ["x"] }\n', "more than one expansion"),
        ('X = { expansions = ["a", "b"], hints.b = ["Valve"] }\n', "lower case"),
        ('"X+Y" = "a"\n', "a key is"),
    ],
)
def test_lexicon_errors(body: str, message: str) -> None:
    with pytest.raises(LexiconError, match=message):
        Lexicon.from_toml(LEXICON_HEADER + body)


def test_lexicon_schema() -> None:
    with pytest.raises(LexiconError, match="schema"):
        Lexicon.from_toml('schema = 2\n[abbreviations]\nX = "a"\n')


def test_custom_lexicon() -> None:
    lexicon = Lexicon.from_toml(LEXICON_HEADER + 'ZZ = "zebra zone"\n')
    assert normalize("ZZ and zz", lexicon).text == "zebra zone and zebra zone"
    assert normalize("ZZ", None).text == "zz"


# Words of expansions that the CMS diagnosis files do not use: procedures, therapies and measures,
# and a few clinical terms the Index words differently, matched by their other words.
NOT_IN_CMS = {
    "antiretroviral",
    "cholangiopancreatography",
    "esophagogastroduodenoscopy",
    "normalized",
    "obtuse",  # obtuse marginal artery: a coronary branch the CMS files do not name
    "video",
    "mildly",
    "percutaneous",
    "transluminal",
    "transcatheter",
    "gap",
    "ventriculoperitoneal",
    "appendectomy",
    "endarterectomy",
    "deceased",
    "endovascular",
    "transurethral",
    "resection",
    "transjugular",
    "portosystemic",
}


@pytest.mark.slow
def test_expansions_use_cms_words(release_graph: Callable[[str], BuildReport]) -> None:
    with Graph(release_graph("fy2027").path) as graph:
        known = vocabulary(graph)
        main_terms = {fold(term.title) for term in graph.main_terms()}
    missing = {
        word
        for entry in LEXICON.entries.values()
        for expansion in entry.expansions
        for word in words(expansion)
        if word not in known
    }
    assert missing <= NOT_IN_CMS, sorted(missing - NOT_IN_CMS)
    # The Index finds an abbreviation that is a main term (GERD, STEMI): no entry for it, unless
    # it is cased and the main term is an English word (Bad, Gas).
    clashes = {entry.key for entry in LEXICON.entries.values() if fold(entry.key) in main_terms}
    assert all(LEXICON.entries[fold(key)].cased for key in clashes), clashes
