"""Modifier extractors: the details of a condition that Tabular verification needs beyond the
words the Index walk uses up.

`modifiers(text)` reads a normalised phrase and returns non-overlapping modifiers with canonical
values and spans: laterality, acuity, stage, severity, episode of care, fracture type, organism,
drug resistance, diabetes type and drugs, control, with or without complication, pregnancy,
trimester and weeks of gestation, and remission. Where two matches overlap the longer wins, so
`acute on chronic` is one acuity and `non-displaced` is not `displaced`. 7th-character wording
may hold other modifiers (`closed` in `initial encounter for closed fracture`).

Words stay in the phrase: a modifier says what they mean, it does not remove them. Some rules
apply only in context: fracture types and healing only when the phrase names a fracture, and
diabetes types only when it names diabetes. The trimester is derived from the weeks of gestation
when the phrase gives weeks but no trimester (Chapter 15: first trimester before 14 weeks 0 days,
second before 28 weeks 0 days, third from 28 weeks 0 days).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from text2icd.text.model import Modifier, ModifierKind

Value = str | Callable[[re.Match[str]], str | None]


def _words(pattern: str) -> str:
    return rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])"


@dataclass(frozen=True, slots=True)
class Rule:
    kind: ModifierKind
    pattern: re.Pattern[str]
    value: Value
    when: re.Pattern[str] | None = None  # the phrase must match this for the rule to apply

    def values(self, text: str) -> list[tuple[int, int, str]]:
        if self.when is not None and not self.when.search(text):
            return []
        found = []
        for match in self.pattern.finditer(text):
            value = self.value if isinstance(self.value, str) else self.value(match)
            if value is not None:
                found.append((match.start(), match.end(), value))
        return found


def _rule(kind: ModifierKind, pattern: str, value: Value, when: str | None = None) -> Rule:
    return Rule(kind, re.compile(_words(pattern)), value, re.compile(when) if when else None)


_FRACTURE = r"fractur|(?<![a-z])fx(?![a-z])"
_GUSTILO = r"(?:i{1,3}|[1-3])[abc]?"  # open fracture types: I, II, IIIA, ...
_FOR_FRACTURE = (
    rf"(?: for (?:open |closed )?fracture"
    rf"(?: type {_GUSTILO}(?:,? {_GUSTILO})*(?: or {_GUSTILO})?)?)?"
)
_DIABETES = r"diabet|(?<![a-z])dm(?![a-z])"
_BEFORE_DIABETES = r"\s+(?:diabetes|dm)\b"
_PREGNANCY = r"pregnan|gestation|intrauterine|gravid|antepartum|trimester|(?<![a-z])ga(?![a-z])"
_HEART_PARTS = (
    r"ventric|atri|heart\b|bundle|main\b|anterior descending|circumflex|coronary|internal mammary"
)
_ORDINAL = {"first": "1", "1st": "1", "second": "2", "2nd": "2", "third": "3", "3rd": "3"}


_DAYS = r"(?:\s*(?:and\s*)?[0-6]/7)?"  # 32 3/7 weeks
_WEEKS = r"(?:weeks?|wks?)"


def _weeks(match: re.Match[str]) -> str | None:
    return match[1] if 0 < int(match[1]) <= 45 else None


def _group(match: re.Match[str]) -> str:
    return match[1]


def _side(match: re.Match[str]) -> str:
    return {"rt": "right", "lt": "left"}.get(match[1], match[1])


def _trimester(weeks: int) -> str:
    return "1" if weeks < 14 else "2" if weeks < 28 else "3"


def _remission(match: re.Match[str]) -> str:
    for word, value in (("partial", "partial"), ("full", "full"), ("complete", "full")):
        if word in match[0]:
            return value
    for word in ("early", "sustained"):
        if word in match[0]:
            return word
    return "remission"


_ORGANISMS: tuple[tuple[str, str], ...] = (
    (r"methicillin[\s-]resistant staph(?:ylococcus|ylococcal)? aureus|mrsa", "MRSA"),
    (r"methicillin[\s-]susceptible staph(?:ylococcus|ylococcal)? aureus|mssa", "MSSA"),
    (r"staph(?:ylococcus|ylococcal)? aureus|s\.? aureus", "Staphylococcus aureus"),
    (
        r"coag(?:ulase)?[\s-](?:negative|neg) staph(?:ylococcus|ylococci|ylococcal)?",
        "Staphylococcus, other",
    ),
    (r"staph(?:ylococcus|ylococci|ylococcal)?", "Staphylococcus"),
    (
        r"strep(?:tococcus|tococcal)? pneumoniae|s\.? pneumoniae|pneumococc(?:us|al)",
        "Streptococcus pneumoniae",
    ),
    (r"group a strep(?:tococcus|tococcal)?|strep(?:tococcus)? pyogenes", "Streptococcus, group A"),
    (
        r"group b strep(?:tococcus|tococcal)?|strep(?:tococcus)? agalactiae",
        "Streptococcus, group B",
    ),
    (r"strep(?:tococcus|tococci|tococcal)?", "Streptococcus"),
    (r"vancomycin[\s-]resistant enterococc(?:us|i|al)|vre", "Enterococcus, vancomycin resistant"),
    (r"enterococc(?:us|i|al)(?: faecalis| faecium)?", "Enterococcus"),
    (r"escherichia coli|e\.? coli", "Escherichia coli"),
    (r"klebsiella(?: pneumoniae| oxytoca)?", "Klebsiella pneumoniae"),
    (r"pseudomonas(?: aeruginosa)?", "Pseudomonas"),
    (r"proteus(?: mirabilis)?", "Proteus"),
    (r"enterobacter(?: cloacae| aerogenes)?", "Enterobacter"),
    (r"serratia(?: marcescens)?", "Serratia"),
    (r"acinetobacter(?: baumannii)?", "Acinetobacter"),
    (r"citrobacter", "Citrobacter"),
    (r"haemophilus influenzae|hemophilus influenzae|h\.? influenzae", "Haemophilus influenzae"),
    (r"helicobacter pylori|h\.? pylori", "Helicobacter pylori"),
    (r"clostridioides difficile|clostridium difficile|c\.? difficile", "Clostridioides difficile"),
    (r"legionella(?: pneumophila)?", "Legionella"),
    (r"mycoplasma(?: pneumoniae)?", "Mycoplasma pneumoniae"),
    (r"listeria(?: monocytogenes)?", "Listeria"),
    (r"bacteroides(?: fragilis)?", "Bacteroides fragilis"),
    (r"salmonella", "Salmonella"),
    (r"campylobacter", "Campylobacter"),
    (r"gram[\s-]negative(?: rods?| bacilli| bacteria| organisms?)?", "gram-negative"),
    (r"gram[\s-]positive(?: cocci| rods?| bacteria| organisms?)?", "gram-positive"),
    (r"anaerob(?:es|ic bacteria)", "anaerobic"),
    (r"candida(?: albicans| glabrata| auris| krusei| tropicalis)?", "Candida"),
    (r"aspergill(?:us|osis)?(?: fumigatus)?", "Aspergillus"),
    (r"pneumocystis(?: jirovecii| jiroveci| carinii)?", "Pneumocystis"),
    (r"cryptococc(?:us|al)(?: neoformans)?", "Cryptococcus"),
    (r"influenza a", "influenza A"),
    (r"influenza b", "influenza B"),
    (r"influenza", "influenza"),
    (r"covid[\s-]?19|sars[\s-]cov[\s-]?2", "COVID-19"),
    (r"respiratory syncytial virus|rsv", "respiratory syncytial virus"),
    (r"cytomegalovir(?:us|al)|cmv", "cytomegalovirus"),
    (r"herpes simplex(?: virus)?(?: type [12])?|hsv(?:[\s-]?[12])?", "herpes simplex"),
    (r"varicella[\s-]zoster(?: virus)?|vzv", "varicella zoster"),
    (r"epstein[\s-]barr(?: virus)?|ebv", "Epstein-Barr virus"),
    (r"human immunodeficiency virus|hiv(?:[\s-]?1)?", "HIV"),
    (r"mycobacterium avium(?: complex| intracellulare)?", "Mycobacterium avium complex"),
)

RULES: tuple[Rule, ...] = (
    # laterality; "left" in heart anatomy (left ventricular, left main) is a name, not a side
    _rule("laterality", rf"(left|right)(?:[\s-]sided)?(?!\s+(?:{_HEART_PARTS}))", _side),
    _rule("laterality", r"(rt|lt)\.?(?=\s+[a-z])", _side),
    _rule("laterality", r"bilateral(?:ly)?|both sides", "bilateral"),
    _rule("laterality", r"unilateral", "unilateral"),
    # acuity
    _rule(
        "acuity", r"acute[\s-]+on[\s-]+chronic|acute and chronic|acute/chronic", "acute_on_chronic"
    ),
    _rule("acuity", r"subacute", "subacute"),
    _rule("acuity", r"acute(?:ly)?|hyperacute", "acute"),
    _rule("acuity", r"chronic(?:ally)?", "chronic"),
    # stage
    _rule("stage", r"stage\s+([1-5][ab]?)", _group),
    _rule("stage", r"end[\s-]stage(?=\s+(?:renal|kidney) disease)", "end_stage"),
    _rule("stage", r"unstageable", "unstageable"),
    _rule("stage", r"deep tissue(?=\s+(?:pressure )?(?:injury|damage))", "deep_tissue"),
    # severity
    _rule("severity", r"moderate(?:ly)?[\s-]+(?:to[\s-]+)?severe", "moderate_to_severe"),
    _rule("severity", r"mild(?:ly)?", "mild"),
    _rule("severity", r"moderate(?:ly)?", "moderate"),
    _rule("severity", r"severe(?:ly)?", "severe"),
    _rule("severity", r"critical", "critical"),
    _rule("severity", r"profound", "profound"),
    _rule("severity", r"morbid(?:ly)?", "morbid"),
    # episode of care (7th characters)
    # 7th-character wording of the Tabular List: "initial encounter for closed fracture"
    _rule("episode", rf"initial (?:encounter|visit|episode of care){_FOR_FRACTURE}", "initial"),
    _rule(
        "episode", rf"subsequent (?:encounter|visit|episode of care){_FOR_FRACTURE}", "subsequent"
    ),
    _rule("episode", r"sequelae?|late effects? of|late effects?", "sequela"),
    _rule("episode", r"(?:with )?routine healing", "routine_healing", _FRACTURE),
    _rule("episode", r"(?:with )?delayed healing", "delayed_healing", _FRACTURE),
    _rule("episode", r"(?:with )?non[\s-]?union", "nonunion", _FRACTURE),
    _rule("episode", r"(?:with )?mal[\s-]?union", "malunion", _FRACTURE),
    # fracture type
    _rule("fracture", r"open", "open", _FRACTURE),
    _rule("fracture", r"closed", "closed", _FRACTURE),
    _rule("fracture", r"non[\s-]?displaced|undisplaced", "nondisplaced", _FRACTURE),
    _rule("fracture", r"displaced", "displaced", _FRACTURE),
    # organisms, as the Index names them
    *(_rule("organism", pattern, value) for pattern, value in _ORGANISMS),
    # drug resistance (Z16)
    _rule(
        "resistance", r"esbl|extended[\s-]spectrum beta[\s-]?lactamase(?:[\s-]producing)?", "ESBL"
    ),
    _rule("resistance", r"carbapenem[\s-]resistant|cre", "carbapenem"),
    _rule("resistance", r"vancomycin[\s-]resistant", "vancomycin"),
    _rule("resistance", r"methicillin[\s-]resistant", "methicillin"),
    _rule(
        "resistance",
        r"multiple drug[\s-]resistant|multi[\s-]?drug[\s-]resistant|mdro?",
        "multiple drug",
    ),
    # diabetes
    # Diabetes types span only the words that give the type, next to "diabetes".
    _rule("diabetes_type", rf"type\s*([12])(?={_BEFORE_DIABETES})", _group),
    _rule("diabetes_type", r"(?<=diabetes mellitus )type\s*([12])", _group),
    _rule("diabetes_type", r"(?<=diabetes )type\s*([12])", _group),
    _rule("diabetes_type", r"(?<=dm )type\s*([12])", _group),
    _rule("diabetes_type", rf"juvenile(?:[\s-]onset)?(?={_BEFORE_DIABETES})", "1"),
    _rule("diabetes_type", rf"non[\s-]?insulin[\s-]dependent(?={_BEFORE_DIABETES})", "2"),
    _rule("diabetes_type", rf"gestational(?={_BEFORE_DIABETES})", "gestational"),
    _rule("diabetes_type", rf"secondary(?={_BEFORE_DIABETES})", "underlying_condition"),
    _rule(
        "diabetes_type",
        r"(?:due to|secondary to) (?:an? )?underlying (?:condition|disease)",
        "underlying_condition",
        _DIABETES,
    ),
    _rule(
        "diabetes_type",
        rf"(?:drug|chemical|steroid|medication)[\s-]induced(?={_BEFORE_DIABETES})",
        "drug_induced",
    ),
    _rule("diabetes_type", rf"other specified(?={_BEFORE_DIABETES})", "other"),
    _rule(
        "diabetes_drug",
        r"insulin[\s-](?:dependent|dependence|requiring|treated|controlled)|on insulin|"
        r"requiring insulin|long[\s-]term (?:\(current\) )?use of insulin",
        "insulin",
    ),
    _rule(
        "diabetes_drug",
        r"oral hypoglycemics?|oral (?:diabetic |antidiabetic )?(?:agents?|medications?)",
        "oral_hypoglycemic",
    ),
    _rule(
        "diabetes_drug",
        r"injectable non[\s-]insulin(?: antidiabetic)?(?: drugs?)?",
        "injectable_non_insulin",
    ),
    # control
    _rule("control", r"uncontrolled", "uncontrolled"),
    _rule(
        "control",
        r"(?:poorly|inadequately|sub[\s-]?optimally|not well) controlled|out of control",
        "poorly_controlled",
    ),
    _rule("control", r"well[\s-]controlled", "controlled"),
    # with or without complication; "complicated by" links conditions and is not one
    _rule("complication", r"without (?:any )?complications?|uncomplicated", "without"),
    _rule("complication", r"with complications?|complicated(?! by)", "with"),
    # pregnancy
    _rule(
        "pregnancy",
        r"pregnan(?:t|cy)|gravid|antepartum|prenatal|antenatal|intrauterine pregnancy",
        "pregnancy",
    ),
    _rule(
        "pregnancy",
        r"intrapartum|in labou?r|during labou?r|during delivery|in childbirth",
        "childbirth",
    ),
    _rule("pregnancy", r"post[\s-]?partum|puerper(?:al|ium)|peripartum", "puerperium"),
    _rule("trimester", r"(first|second|third|1st|2nd|3rd) trimester", lambda m: _ORDINAL[m[1]]),
    _rule(
        "gestation_weeks",
        rf"(\d{{1,2}}){_DAYS}\s*{_WEEKS}(?:'|\s+of)?\s+"
        r"(?:gestation(?:al age)?|ga|pregnant|pregnancy|intrauterine pregnancy)",
        _weeks,
    ),
    _rule(
        "gestation_weeks",
        rf"(?:at|ga|gestational age(?: of)?)\s+(\d{{1,2}}){_DAYS}\s*{_WEEKS}",
        _weeks,
        _PREGNANCY,
    ),
    # remission (not "resolved": Guideline I.C.4.a.1(b))
    _rule(
        "remission",
        r"in (?:early |sustained )?(?:full |partial |complete )?remission|remitted",
        _remission,
    ),
)


def modifiers(text: str) -> tuple[Modifier, ...]:
    """The modifiers of a normalised phrase, in order. Longer matches win over shorter ones they
    overlap, except inside 7th-character wording, which may hold modifiers of other kinds
    ("closed" in "initial encounter for closed fracture")."""
    candidates = [
        (start, end, order, rule.kind, value)
        for order, rule in enumerate(RULES)
        for start, end, value in rule.values(text)
    ]
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0], c[2]))
    taken: list[tuple[int, int, int, ModifierKind, str]] = []
    for candidate in candidates:
        start, end, _, kind, _ = candidate
        if all(
            end <= other[0]
            or start >= other[1]
            or (other[0] <= start and end <= other[1] and other[3] == "episode" != kind)
            for other in taken
        ):
            taken.append(candidate)
    found = [
        Modifier(kind=kind, value=value, text=text[start:end], start=start, end=end)
        for start, end, _, kind, value in sorted(taken)
    ]
    return tuple(sorted(found + _derived_trimester(found), key=lambda m: m.start))


def _derived_trimester(found: list[Modifier]) -> list[Modifier]:
    if any(m.kind == "trimester" for m in found):
        return []
    weeks = [m for m in found if m.kind == "gestation_weeks"]
    if not weeks:
        return []
    first = weeks[0]
    return [first.model_copy(update={"kind": "trimester", "value": _trimester(int(first.value))})]
