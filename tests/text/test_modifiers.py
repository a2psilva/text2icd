import pytest

from text2icd.text.abbreviations import default_lexicon
from text2icd.text.modifiers import modifiers
from text2icd.text.normalize import normalize


def found(phrase: str) -> list[tuple[str, str]]:
    text = normalize(phrase, default_lexicon()).text
    return [(m.kind, m.value) for m in modifiers(text)]


# (phrase as written, modifiers as (kind, value), in order)
CASES = [
    # laterality
    ("left hip fracture", [("laterality", "left")]),
    ("R knee OA", [("laterality", "right")]),
    ("right-sided weakness", [("laterality", "right")]),
    ("rt shoulder pain", [("laterality", "right")]),
    ("lt. foot ulcer", [("laterality", "left")]),
    ("bilateral pleural effusions", [("laterality", "bilateral")]),
    ("b/l LE edema", [("laterality", "bilateral")]),
    ("pneumonia, both sides", [("laterality", "bilateral")]),
    ("unilateral hearing loss", [("laterality", "unilateral")]),
    ("RLL pneumonia", [("laterality", "right")]),
    ("left ventricular thrombus", []),  # heart anatomy, not a side
    ("right heart failure", []),
    ("LBBB", []),
    ("left main coronary artery disease", []),
    ("left atrial enlargement", []),
    # acuity
    ("acute kidney injury", [("acuity", "acute")]),
    ("chronic pancreatitis", [("acuity", "chronic")]),
    ("acute on chronic systolic CHF", [("acuity", "acute_on_chronic")]),
    ("acute-on-chronic respiratory failure", [("acuity", "acute_on_chronic")]),
    ("acute and chronic respiratory failure", [("acuity", "acute_on_chronic")]),
    ("subacute thyroiditis", [("acuity", "subacute")]),
    ("acutely ill", [("acuity", "acute")]),
    # stage
    ("CKD stage 3a", [("acuity", "chronic"), ("stage", "3a")]),
    ("CKD IV", [("acuity", "chronic"), ("stage", "4")]),
    ("ESRD", [("stage", "end_stage")]),
    ("ESKD on HD", [("stage", "end_stage")]),
    ("pressure ulcer of sacrum, stage 2", [("stage", "2")]),
    ("unstageable pressure injury of heel", [("stage", "unstageable")]),
    ("deep tissue pressure injury", [("stage", "deep_tissue")]),
    ("stage IV lung cancer", [("stage", "4")]),
    ("end stage liver disease", []),
    # severity
    ("mild intermittent asthma", [("severity", "mild")]),
    ("moderate persistent asthma", [("severity", "moderate")]),
    ("severe sepsis", [("severity", "severe")]),
    ("moderate to severe aortic stenosis", [("severity", "moderate_to_severe")]),
    ("morbid obesity", [("severity", "morbid")]),
    ("critical limb ischemia", [("severity", "critical")]),
    ("profound intellectual disability", [("severity", "profound")]),
    # episode of care
    ("femur fracture, initial encounter", [("episode", "initial")]),
    ("burn of hand, subsequent encounter", [("episode", "subsequent")]),
    ("sequela of stroke", [("episode", "sequela")]),
    ("late effects of polio", [("episode", "sequela")]),
    ("fracture with delayed healing", [("episode", "delayed_healing")]),
    ("fracture with routine healing", [("episode", "routine_healing")]),
    ("tibia fx, nonunion", [("episode", "nonunion")]),
    ("radius fracture, malunion", [("episode", "malunion")]),
    ("delayed healing of wound", []),  # healing types only for fractures
    # fracture type
    ("open fracture of tibia", [("fracture", "open")]),
    ("closed fracture of left radius", [("fracture", "closed"), ("laterality", "left")]),
    ("displaced fracture of femoral neck", [("fracture", "displaced")]),
    ("non-displaced fx of R ulna", [("fracture", "nondisplaced"), ("laterality", "right")]),
    ("open wound of scalp", []),
    # organisms and resistance
    ("UTI due to E. coli", [("organism", "Escherichia coli")]),
    ("MRSA bacteremia", [("organism", "MRSA")]),
    ("MSSA pneumonia", [("organism", "MSSA")]),
    ("Staph aureus endocarditis", [("organism", "Staphylococcus aureus")]),
    ("coag negative staph bacteremia", [("organism", "Staphylococcus, other")]),
    ("coagulase negative staph bacteremia", [("organism", "Staphylococcus, other")]),
    ("strep pneumoniae pneumonia", [("organism", "Streptococcus pneumoniae")]),
    ("pneumococcal pneumonia", [("organism", "Streptococcus pneumoniae")]),
    ("GAS pharyngitis", [("organism", "Streptococcus, group A")]),
    ("VRE UTI", [("organism", "Enterococcus, vancomycin resistant")]),
    ("enterococcus faecalis bacteremia", [("organism", "Enterococcus")]),
    ("klebsiella pneumonia", [("organism", "Klebsiella pneumoniae")]),
    ("pseudomonas aeruginosa VAP", [("organism", "Pseudomonas")]),
    ("C. diff colitis", [("organism", "Clostridioides difficile")]),
    ("H. pylori gastritis", [("organism", "Helicobacter pylori")]),
    ("GNR bacteremia", [("organism", "gram-negative")]),
    ("candida esophagitis", [("organism", "Candida")]),
    ("PJP", [("organism", "Pneumocystis")]),
    ("influenza A", [("organism", "influenza A")]),
    ("COVID-19 pneumonia", [("organism", "COVID-19")]),
    ("RSV bronchiolitis", [("organism", "respiratory syncytial virus")]),
    ("CMV colitis", [("organism", "cytomegalovirus")]),
    ("HSV encephalitis", [("organism", "herpes simplex")]),
    ("HIV", [("organism", "HIV")]),
    ("UTI 2/2 ESBL E. coli", [("resistance", "ESBL"), ("organism", "Escherichia coli")]),
    ("CRE bacteremia", [("resistance", "carbapenem")]),
    ("MDR pseudomonas", [("resistance", "multiple drug"), ("organism", "Pseudomonas")]),
    # diabetes
    ("DM2", [("diabetes_type", "2")]),
    ("type 1 diabetes", [("diabetes_type", "1")]),
    ("diabetes mellitus type 2", [("diabetes_type", "2")]),
    ("DM II", [("diabetes_type", "2")]),
    ("type II DM", [("diabetes_type", "2")]),
    ("juvenile diabetes", [("diabetes_type", "1")]),
    ("NIDDM", [("diabetes_type", "2")]),
    ("GDM", [("diabetes_type", "gestational")]),
    ("steroid-induced diabetes", [("diabetes_type", "drug_induced")]),
    ("secondary diabetes", [("diabetes_type", "underlying_condition")]),
    ("diabetes due to underlying condition", [("diabetes_type", "underlying_condition")]),
    ("type 2 MI", []),  # a type of myocardial infarction, not of diabetes
    ("IDDM", [("diabetes_drug", "insulin")]),
    ("DM2 on insulin", [("diabetes_type", "2"), ("diabetes_drug", "insulin")]),
    ("DM2 on oral hypoglycemics", [("diabetes_type", "2"), ("diabetes_drug", "oral_hypoglycemic")]),
    ("uncontrolled DM2", [("control", "uncontrolled"), ("diabetes_type", "2")]),
    ("poorly controlled diabetes", [("control", "poorly_controlled")]),
    ("well-controlled HTN", [("control", "controlled")]),
    # complication
    ("DM2 without complications", [("diabetes_type", "2"), ("complication", "without")]),
    ("DM2 w/o complication", [("diabetes_type", "2"), ("complication", "without")]),
    ("uncomplicated appendicitis", [("complication", "without")]),
    ("complicated UTI", [("complication", "with")]),
    ("type 2 diabetes with complications", [("diabetes_type", "2"), ("complication", "with")]),
    ("cirrhosis complicated by ascites", []),  # links two conditions
    # pregnancy
    ("pregnant", [("pregnancy", "pregnancy")]),
    ("anemia in pregnancy", [("pregnancy", "pregnancy")]),
    ("postpartum hemorrhage", [("pregnancy", "puerperium")]),
    ("peripartum cardiomyopathy", [("pregnancy", "puerperium")]),
    ("puerperal sepsis", [("pregnancy", "puerperium")]),
    ("fever in labor", [("pregnancy", "childbirth")]),
    ("preeclampsia, third trimester", [("trimester", "3")]),
    ("UTI in pregnancy, 2nd trimester", [("pregnancy", "pregnancy"), ("trimester", "2")]),
    (
        "IUP at 32 weeks",
        [("pregnancy", "pregnancy"), ("gestation_weeks", "32"), ("trimester", "3")],
    ),
    (
        "pregnancy at 13 weeks",
        [("pregnancy", "pregnancy"), ("gestation_weeks", "13"), ("trimester", "1")],
    ),
    ("14 weeks gestation", [("gestation_weeks", "14"), ("trimester", "2")]),
    ("27 3/7 weeks gestation", [("gestation_weeks", "27"), ("trimester", "2")]),
    ("28 weeks gestation", [("gestation_weeks", "28"), ("trimester", "3")]),
    ("fever at 2 weeks", []),  # no pregnancy in the phrase
    # remission
    ("ALL in remission", [("acuity", "acute"), ("remission", "remission")]),
    ("alcohol dependence in early remission", [("remission", "early")]),
    ("depression in partial remission", [("remission", "partial")]),
    ("MDD, in full remission", [("remission", "full")]),
    ("diabetes resolved", []),  # resolved is not remission (I.C.4.a.1(b))
]


@pytest.mark.parametrize(("phrase", "expected"), CASES)
def test_modifiers(phrase: str, expected: list[tuple[str, str]]) -> None:
    assert found(phrase) == expected


def test_spans() -> None:
    text = "acute on chronic systolic heart failure, left"
    acuity, side = modifiers(text)
    assert (acuity.start, acuity.end, acuity.text) == (0, 16, "acute on chronic")
    assert text[side.start : side.end] == side.text == "left"


def test_longer_match_wins() -> None:
    text = "non-displaced fracture"
    (only,) = modifiers(text)
    assert (only.kind, only.value, only.text) == ("fracture", "nondisplaced", "non-displaced")
