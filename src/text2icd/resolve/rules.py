"""The guideline rule pack: the Official Guidelines' rules that neither the Index nor the Tabular
notes apply on their own (task 6.9). Every rule writes a `rules` trace step citing its section.

- **Combination codes (I.B.9).** A combination code is used alone only when it identifies every
  condition documented. The coder (`coder`) codes on its own a condition, or a comma-separated
  part of one, that the code of its diagnosis leaves entirely unused ("hypertension with CKD
  stage 4": N18.4 says nothing of the hypertension).
- **Acute and chronic (I.B.8).** A condition described as both, with separate Index subentries
  for each and no combination code, gets both codes, the acute one first (`coder`).
- **Hypertension (I.C.9.a.1-3).** Hypertension with a heart condition (I50.-, I51.4-I51.7,
  I51.89, I51.9) is I11, with chronic kidney disease (N18.-) I12, with both I13: the
  classification presumes the link. The fourth and fifth characters follow from heart failure
  and from stage 5 CKD or ESRD, read from the codes of each condition on its own; the I50 and
  N18 codes come with their `use additional code` notes. A provider who documents the
  conditions as unrelated is a case for review.
- **Sepsis (I.C.1.d.4).** Severe sepsis is R65.20, with septic shock R65.21, sequenced after the
  systemic infection; when none is documented it is A41.9. A localized infection of the same
  diagnosis comes after R65.2-.
- **Diabetes (I.C.4.a.2, I.C.4.a.3).** Diabetes of no documented type is type 2: the Index
  defaults to E11 and the trace says so. Insulin, oral and injectable non-insulin drugs answer
  the `use additional code` notes of E08-E13 with Z79.4, Z79.84 and Z79.85, both when both are
  documented; E10 has no such note.
- **Poisoning, adverse effect, underdosing (I.C.19.e.5).** A poisoning code (T36-T50 with intent
  1 to 4, or a toxic effect, T51-T65) comes before the manifestations of its diagnosis; an
  adverse effect (5) or underdosing (6) code after the condition it caused.
- **Sequelae (I.B.10).** The residual condition comes first, the injury code with 7th
  character S after it.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from text2icd.graph.model import Graph
from text2icd.ontology.codes import undot
from text2icd.resolve.instructions import Entry, Excluded, Instruction, Notes, Order
from text2icd.resolve.tabular import Evidence, TabularVerifier
from text2icd.text.model import Modifier
from text2icd.trace import TraceStep

# Undotted prefixes.
HYPERTENSION = ("I10", "I11", "I12", "I13")
HEART = ("I50", "I514", "I515", "I516", "I517", "I5189", "I519")  # I.C.9.a.1
WITH_HEART_FAILURE = ("I110", "I130", "I132")
CKD_5 = ("N185", "N186")
WITH_CKD_5 = ("I120", "I1311", "I132")
SEVERE_SEPSIS = ("R6520", "R6521")
UNSPECIFIED_SEPSIS = "A41.9"
PREGNANCY = ("O10", "O11", "O12", "O13", "O14", "O15", "O16")  # hypertension in pregnancy
DRUG_STATUS = {  # diabetes_drug modifier -> long-term drug therapy code (I.C.4.a.3)
    "insulin": "Z79.4",
    "oral_hypoglycemic": "Z79.84",
    "injectable_non_insulin": "Z79.85",
}

_SHOCK = re.compile(r"(?<!without )(?<!no )\bseptic shock\b")  # "without septic shock": R65.20
_SEVERE = re.compile(r"(?<!without )(?<!no )\bsevere sepsis\b")
_SEPSIS = re.compile(r"\bsep(?:sis|ticemia|ticaemia)\b|\bmeningococcemia\b")


@dataclass(frozen=True, slots=True)
class Fact:
    """The code of one condition of the phrase on its own, and the diagnosis it is part of."""

    code: str
    group: int


def step(action: str, detail: str, guideline: str) -> TraceStep:
    return TraceStep(layer="rules", action=action, detail=detail, guideline=guideline)


def hypertensive(code: str) -> bool:
    return undot(code).startswith(HYPERTENSION)


def hypertension_code(facts: Sequence[Fact]) -> str | None:
    """The I10-I13 code the conditions of the phrase call for together (I.C.9.a.1-3), or None
    when no hypertension is documented."""
    codes = [undot(f.code) for f in facts]
    if not any(c.startswith(HYPERTENSION) for c in codes):
        return None
    failure_codes = [c for c in codes if c.startswith("I50")]
    failure = bool(failure_codes) or any(c.startswith(WITH_HEART_FAILURE) for c in codes)
    heart = failure or any(c.startswith((*HEART, "I11", "I13")) for c in codes)
    stages = [c for c in codes if c.startswith("N18")]
    kidney = bool(stages) or any(c.startswith(("I12", "I13")) for c in codes)
    stage_5 = (
        any(c.startswith(CKD_5) for c in stages)
        if stages
        else any(c.startswith(WITH_CKD_5) for c in codes)
    )
    if heart and kidney:
        if failure:
            return "I13.2" if stage_5 else "I13.0"
        return "I13.11" if stage_5 else "I13.10"
    if heart:
        return "I11.0" if failure else "I11.9"
    if kidney:
        return "I12.0" if stage_5 else "I12.9"
    return "I10"


def apply_hypertension(
    entries: list[Entry], facts: Sequence[Fact], graph: Graph
) -> tuple[list[Entry], list[TraceStep]]:
    """Put the combination code of I.C.9.a in place of the set's hypertension codes (a
    hypertension the walk did not code is coded on its own first, I.B.9)."""
    target = hypertension_code(facts)
    present = [i for i, e in enumerate(entries) if hypertensive(e.code)]
    if target is None or target not in graph or not present:
        return entries, []  # the set codes no hypertension of I10-I13 ("pulmonary")
    if any(undot(e.code)[:3] in PREGNANCY for e in entries):
        return entries, []  # hypertension in pregnancy has combinations of its own (I.C.15.d)
    if [entries[i].code for i in present] == [target]:
        return entries, []
    found = sorted({f.code for f in facts if not hypertensive(f.code)} - {target})
    detail = f"{target}: hypertension with " + (", ".join(found) or "its combination")
    guideline = {"I11": "I.C.9.a.1", "I12": "I.C.9.a.2", "I13": "I.C.9.a.3"}.get(
        undot(target)[:3], "I.C.9.a"
    )
    description = graph.code(target).description
    out = list(entries)
    first = out[present[0]]
    before = ", ".join(dict.fromkeys(out[i].code for i in present))
    out[present[0]] = replace(
        first, code=target, description=description, specified=True, role="diagnosis"
    )
    out = [e for k, e in enumerate(out) if k not in present[1:]]
    detail = f"{before} -> {detail} (the classification presumes the link)"
    return out, [step("hypertension", detail, guideline)]


def presumed(excluded: Excluded) -> bool:
    """An Excludes1 drop the hypertension rule calls for: a heart condition that the I11 or
    I13 code includes ("cardiomegaly due to hypertension (I11.-)"), no conflict to review."""
    return hypertensive(excluded.kept) and undot(excluded.code).startswith(HEART)


def owned(instruction: Instruction) -> bool:
    """A note a rule answers in place of the companion search: the systemic infection of
    severe sepsis (`apply_sepsis`), which any sepsis code answers, listed or not."""
    return instruction.kind == "codeFirst" and undot(instruction.node).startswith("R652")


def systemic(code: str, verifier: TabularVerifier, notes: Notes) -> bool:
    """A code of a systemic infection: sepsis by its title or inclusion terms, or one the
    `code first` notes of R65.2 name."""
    if undot(code).startswith("R65") or code not in verifier.graph:
        return False
    if any(i.kind == "codeFirst" and i.covers(code) for i in notes.instructions("R65.20")):
        return True
    texts = (verifier.graph.code(code).description, *verifier.inclusions(code))
    return any(_SEPSIS.search(text.casefold()) for text in texts)


def apply_sepsis(
    entries: list[Entry], texts: Mapping[int, str], verifier: TabularVerifier, notes: Notes
) -> tuple[list[Entry], list[Order], list[TraceStep]]:
    """Severe sepsis and septic shock (I.C.1.d.4): R65.20 or R65.21 after the systemic
    infection, A41.9 when none is documented, and the localized infection after them."""
    graph = verifier.graph
    out = list(entries)
    orders: list[Order] = []
    trace: list[TraceStep] = []
    groups = sorted({e.group for e in out} | set(texts))
    # Septic shock anywhere in the phrase makes its severe sepsis R65.21.
    shocked = any(_SHOCK.search(text) for text in texts.values()) or any(
        e.code == "R65.21" for e in out
    )
    for group in groups:
        text = texts.get(group, "")
        members = [e for e in out if e.group == group]
        shock = bool(_SHOCK.search(text)) or any(e.code == "R65.21" for e in members)
        severe = shock or bool(_SEVERE.search(text)) or any(e.code == "R65.20" for e in members)
        if not severe:
            continue
        shock = shocked
        target = "R65.21" if shock else "R65.20"
        description = graph.code(target).description
        found = [
            k for k, e in enumerate(out) if e.group == group and undot(e.code) in SEVERE_SEPSIS
        ]
        if found:
            out[found[0]] = replace(out[found[0]], code=target, description=description)
            out = [e for k, e in enumerate(out) if k not in found[1:]]
        else:
            at = max((k for k, e in enumerate(out) if e.group == group), default=len(out) - 1)
            out.insert(at + 1, Entry(target, description, "diagnosis", None, group, 0.0, True))
            trace.append(step("severe_sepsis", f"{target}: {description}", "I.C.1.d.4"))
        infections = [e for e in out if systemic(e.code, verifier, notes)]
        if not infections:
            at = next(k for k, e in enumerate(out) if e.code == target)
            sepsis = Entry(
                UNSPECIFIED_SEPSIS,
                graph.code(UNSPECIFIED_SEPSIS).description,
                "code_first",
                target,
                group,
                0.0,
                False,
            )
            out.insert(at, sepsis)
            infections = [sepsis]
            trace.append(
                step(
                    "severe_sepsis",
                    f"{UNSPECIFIED_SEPSIS} before {target}: no systemic infection documented",
                    "I.C.1.d.4",
                )
            )
        for infection in infections:
            orders.append(
                Order(
                    infection.code,
                    target,
                    step(
                        "sequence",
                        f"{infection.code} before {target}: the systemic infection first",
                        "I.C.1.d.4",
                    ),
                )
            )
        local = {e.code for e in out if e.group == group} - {target} - {i.code for i in infections}
        for code in sorted(local):
            orders.append(
                Order(
                    target,
                    code,
                    step(
                        "sequence",
                        f"{target} before {code}: a localized infection after severe sepsis",
                        "I.C.1.d.4",
                    ),
                )
            )
    return out, orders, trace


def diabetes_type(entries: Sequence[Entry], evidence: Mapping[int, Evidence]) -> list[TraceStep]:
    """Trace the type 2 default of a diabetes of no documented type (I.C.4.a.2)."""
    out = []
    for entry in entries:
        found = evidence.get(entry.group)
        if entry.role != "diagnosis" or found is None or not undot(entry.code).startswith("E11"):
            continue
        if not found.modifier("diabetes_type") and not re.search(r"\btype (?:2|ii)\b", found.text):
            out.append(
                step("diabetes_type", f"{entry.code}: type not documented, type 2", "I.C.4.a.2")
            )
    return out


def drug_status(instruction: Instruction, modifiers: Sequence[Modifier]) -> Modifier | None:
    """The diabetes drug of the phrase that answers a `use additional code` note of the drug
    therapy it calls for (I.C.4.a.3), if any."""
    if instruction.kind != "useAdditionalCode":
        return None
    for modifier in modifiers:
        code = DRUG_STATUS.get(modifier.value) if modifier.kind == "diabetes_drug" else None
        if code is not None and instruction.covers(code):
            return modifier
    return None


def intent(code: str) -> str | None:
    """`poisoning`, `adverse_effect` or `underdosing` for a code of the Table of Drugs and
    Chemicals (T36-T65), from its intent character."""
    bare = undot(code)
    if not bare.startswith("T") or len(bare) < 6 or not bare[1:3].isdigit():
        return None
    block = int(bare[1:3])
    if 51 <= block <= 65:
        return "poisoning"  # toxic effects are sequenced like poisonings
    if not 36 <= block <= 50:
        return None
    return {"5": "adverse_effect", "6": "underdosing"}.get(
        bare[5], "poisoning" if bare[5] in "1234" else None
    )


def sequela(code: str) -> bool:
    """An injury or poisoning code with 7th character S (I.B.10)."""
    bare = undot(code)
    return bare[0] in "ST" and len(bare) == 7 and bare[-1] == "S"


def orders(entries: Sequence[Entry], origins: Sequence[int]) -> list[Order]:
    """The coding order the guidelines give the codes of one diagnosis: poisoning before its
    manifestations, adverse effects and underdosing after (I.C.19.e.5), the residual before
    the sequela (I.B.10)."""
    out: list[Order] = []

    def origin(entry: Entry) -> int:
        return origins[entry.group] if entry.group < len(origins) else entry.group

    for entry in entries:
        kind = intent(entry.code)
        others = [
            e
            for e in entries
            if origin(e) == origin(entry) and e.code != entry.code and intent(e.code) is None
        ]
        if kind == "poisoning":
            out += [
                Order(
                    entry.code,
                    other.code,
                    step(
                        "sequence",
                        f"{entry.code} before {other.code}: the poisoning first",
                        "I.C.19.e.5.b",
                    ),
                )
                for other in others
            ]
        elif kind in ("adverse_effect", "underdosing"):
            what, guideline = (
                ("the nature of the adverse effect first", "I.C.19.e.5.a")
                if kind == "adverse_effect"
                else ("the condition first, never the underdosing", "I.C.19.e.5.c")
            )
            out += [
                Order(
                    other.code,
                    entry.code,
                    step("sequence", f"{other.code} before {entry.code}: {what}", guideline),
                )
                for other in others
            ]
        elif sequela(entry.code):
            out += [
                Order(
                    other.code,
                    entry.code,
                    step(
                        "sequence",
                        f"{other.code} before {entry.code}: the residual condition first",
                        "I.B.10",
                    ),
                )
                for other in others
                if not sequela(other.code)
            ]
    return out
