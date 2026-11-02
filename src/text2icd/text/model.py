"""The structures text understanding produces, with spans into the normalised phrase."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from text2icd.text.abbreviations import Expansion
from text2icd.trace import TraceStep

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


RelationKind = Literal[
    "with",  # with, associated with
    "without",
    "due_to",  # due to, secondary to, caused by, related to
    "complicated_by",
    "in",  # in the setting of, in the context of
    "on",  # AKI on CKD, ESRD on hemodialysis
    "and",
    "or",
    "versus",
]
LINKING: frozenset[RelationKind] = frozenset(
    {"with", "without", "due_to", "complicated_by", "in", "on"}
)


class Relation(BaseModel):
    """Two conditions joined by a connector. `head` and `linked` index `ParsedPhrase.conditions`:
    in "AKI due to dehydration" the head is AKI and the linked condition dehydration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RelationKind
    head: int
    linked: int
    text: str  # the connector, as in the normalised phrase

    @property
    def links(self) -> bool:
        """Whether the conditions belong to one diagnosis (`with`, `due to`), rather than being
        listed side by side (`and`, `or`, `vs`)."""
        return self.kind in LINKING


class Condition(BaseModel):
    """One condition of a phrase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str  # its words without cue words: what the Index walk reads
    start: int  # its span in the normalised phrase, refinements and qualifiers included
    end: int
    modifiers: tuple[Modifier, ...] = ()
    cues: tuple[Cue, ...] = ()  # the cues that cover it
    status: Status = "current"
    certainty: Certainty = "affirmed"
    reportable: bool = True  # whether it leads to a code (for a history, a history code)
    diagnosis: int = 0  # its group in ParsedPhrase.diagnoses


class ParsedPhrase(BaseModel):
    """A phrase as text understanding reads it: its conditions, how they relate, and every step
    that led there."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original: str
    normalized: str
    setting: Setting
    conditions: tuple[Condition, ...]
    relations: tuple[Relation, ...] = ()
    diagnoses: tuple[tuple[int, ...], ...] = ()  # conditions joined by linking relations
    expansions: tuple[Expansion, ...] = ()
    unknown: tuple[str, ...] = ()  # abbreviation-like tokens with no lexicon entry
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()
