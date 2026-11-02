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
