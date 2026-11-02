"""The structures text understanding produces, with spans into the normalised phrase."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

ModifierKind = Literal[
    "laterality",  # left, right, bilateral, unilateral
    "acuity",  # acute, subacute, chronic, acute_on_chronic
    "stage",  # 1-5 with a/b, end_stage, unstageable, deep_tissue
    "severity",  # mild, moderate, severe, critical, profound, morbid
    "episode",  # initial, subsequent, sequela, and for fractures routine_healing,
    # delayed_healing, nonunion, malunion
    "fracture",  # open, closed, displaced, nondisplaced
    "organism",  # as the Index names it: "Escherichia coli", "MRSA"
    "resistance",  # ESBL, carbapenem, vancomycin, multiple drug
    "diabetes_type",  # 1, 2, gestational, underlying_condition, drug_induced, other
    "diabetes_drug",  # insulin, oral_hypoglycemic, injectable_non_insulin
    "control",  # uncontrolled, poorly_controlled, controlled
    "complication",  # with, without
    "pregnancy",  # pregnancy, childbirth, puerperium
    "trimester",  # 1, 2, 3
    "gestation_weeks",  # the number of weeks
    "remission",  # remission, partial, full, early, sustained
]


class Modifier(BaseModel):
    """A modifier found in the normalised phrase: its kind, a canonical value, and its span."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ModifierKind
    value: str
    text: str  # the words, as in the normalised phrase
    start: int
    end: int


CueKind = Literal[
    "history",  # personal history of a condition that no longer exists
    "family_history",
    "status_post",  # after a procedure (a status) or an event
    "resolved",  # no longer present; not the same as "in remission"
    "negated",  # absent or ruled out
    "uncertain",  # probable, suspected, possible, rule out, ...
]
CueFamily = Literal["status", "certainty"]
Status = Literal["current", "history", "family_history", "status_post", "resolved"]
Certainty = Literal["affirmed", "negated", "uncertain"]
Setting = Literal["inpatient", "outpatient"]
CUE_FAMILY: dict[CueKind, CueFamily] = {
    "history": "status",
    "family_history": "status",
    "status_post": "status",
    "resolved": "status",
    "negated": "certainty",
    "uncertain": "certainty",
}


class Cue(BaseModel):
    """Words that qualify a condition: `before` it (`possible pneumonia`, `history of MI`) or
    `after` it (`PE ruled out`, `pneumonia, resolved`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: CueKind
    text: str
    start: int
    end: int
    position: Literal["before", "after"]

    @property
    def family(self) -> CueFamily:
        return CUE_FAMILY[self.kind]
