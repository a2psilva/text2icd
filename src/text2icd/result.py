"""The public result of coding a phrase (task 9.1): what the Python API, `text2icd code --json`,
`text2icd batch` and the HTTP service return. Its JSON Schema is `docs/schema/coding-result.json`
(`python -m text2icd.result` writes it).

A `CodingResult` is built from the coder's `CodeSet`, which stays the internal form. It holds:

- `chunks`: the conditions the phrase states, each with where it was written in the input;
- `codes`: in coding order, the first being the primary one, each with its role and the chunks
  it codes;
- one calibrated `confidence` for the set (that its primary code is right), the alternatives,
  the flags, a flat `trace`, and what L3 cost.

On request (`explain`), `detail` adds what a UI draws: each coded diagnosis's path through the
Index (graph term ids of the release) and the Tabular List, the instructional notes and what they
led to, the codes Excludes1 took out, the codes L2 weighed, and L3's decisions.
"""

import json
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict

from text2icd import __version__
from text2icd.graph.model import Graph
from text2icd.llm.decisions import Decided
from text2icd.resolve.choice import Pooled
from text2icd.resolve.coder import CodedDiagnosis, CodeSet, Layer
from text2icd.resolve.instructions import Kind, Role
from text2icd.resolve.walk import PathStep
from text2icd.text.model import Certainty, Setting, Status
from text2icd.trace import TraceStep

SCHEMA_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Mark(_Model):
    """A modifier or cue of a chunk, where it was written."""

    kind: str  # laterality, stage, ...; or history, negated, uncertain, ...
    value: str  # the canonical value ("left", "3a"); for a cue, before or after the condition
    text: str  # as written in the input
    start: int
    end: int


class Chunk(_Model):
    """One condition of the phrase, as written (`start`, `end` index the input)."""

    id: int
    text: str
    start: int
    end: int
    condition: str  # the words the Index walk reads (normalised, cue words removed)
    status: Status
    certainty: Certainty
    reportable: bool  # whether it leads to a code
    diagnosis: int  # chunks with the same number form one diagnosis ("DM2 w/ CKD")
    modifiers: tuple[Mark, ...] = ()
    cues: tuple[Mark, ...] = ()


class Expansion(_Model):
    """An abbreviation read as one of its meanings."""

    written: str
    expansion: str
    start: int | None = None
    end: int | None = None
    alternatives: tuple[str, ...] = ()  # the other meanings of an ambiguous abbreviation
    chosen_by: Literal["only", "hint", "default"] = "only"


class AssignedCode(_Model):
    """One code of the result, always billable in the release."""

    code: str
    description: str
    role: Role
    primary: bool  # the first code: what the phrase is mainly coded as
    source: str | None = None  # the code whose Index pair or note brought this one in
    chunks: tuple[int, ...] = ()  # the chunks it codes
    path: int | None = None  # its diagnosis in `detail.paths`, when it is a diagnosis's code


class CodeRef(_Model):
    code: str
    description: str


class DiagnosisPath(_Model):
    """How one diagnosis was coded: its path through the Index, then the Tabular List."""

    code: str
    description: str
    manifestation: str | None = None
    role: Literal["diagnosis", "external_cause"] = "diagnosis"
    chunks: tuple[int, ...] = ()
    book: str  # index, eindex, neoplasm, drug; retrieval when L2 found the code
    written: str  # the code as the Index gives it ("E11.2-": to be completed)
    path: str  # the Index path as text ("Diabetes, diabetic > type 2 > with > ...")
    steps: tuple[PathStep, ...] = ()
    coded_step: int | None = None  # the step that gives the code
    alternates: tuple[CodeRef, ...] = ()  # other codes the Index walk led to
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()  # this diagnosis's steps: walk, path, Tabular


class Companion(_Model):
    """What a `code first`, `use additional code` or `code also` note led to."""

    source: str
    kind: Kind
    note: str
    status: Literal["added", "present", "required_unresolved"]
    code: str | None = None
    description: str | None = None


class Exclusion(_Model):
    code: str  # taken out
    kept: str  # because of this code's Excludes1 note
    note: str


class Detail(_Model):
    paths: tuple[DiagnosisPath, ...] = ()
    companions: tuple[Companion, ...] = ()
    excluded: tuple[Exclusion, ...] = ()
    pool: tuple[Pooled, ...] = ()  # the codes L2 weighed, most probable first
    decisions: tuple[Decided, ...] = ()  # what L3 was asked and answered


class CodingResult(_Model):
    schema_version: int = SCHEMA_VERSION
    input: str
    normalized: str
    release: str
    graph: str  # the graph's content hash: same graph, same answers
    version: str  # of text2icd
    setting: Setting
    chunks: tuple[Chunk, ...] = ()
    expansions: tuple[Expansion, ...] = ()
    codes: tuple[AssignedCode, ...] = ()
    alternatives: tuple[CodeRef, ...] = ()  # for the primary code
    confidence: float | None = None  # calibrated: that the primary code is right
    escalated: tuple[Layer, ...] = ()  # the layers whose confidence was too low
    resolver: Layer = "index"  # the layer that gave the primary code
    cached: bool = False  # answered from the result cache (no work, no cost)
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()
    llm_calls: int = 0
    cost_usd: float = 0.0
    detail: Detail | None = None

    @property
    def primary(self) -> AssignedCode | None:
        return self.codes[0] if self.codes else None

    @classmethod
    def of(
        cls, code_set: CodeSet, graph: Graph, *, explain: bool = False, cached: bool = False
    ) -> "CodingResult":
        """The result of a code set coded with `graph`."""
        parsed = code_set.parsed
        coded = (*code_set.diagnoses, *code_set.external_causes)
        primary = code_set.diagnoses[0] if code_set.diagnoses else None
        trace = [
            *parsed.trace,
            *(step for diagnosis in coded for step in diagnosis.trace),
            *(step for companion in code_set.companions for step in companion.trace),
            *code_set.trace,
        ]
        return cls(
            input=parsed.original,
            normalized=parsed.normalized,
            release=graph.release,
            graph=graph.meta.get("content_sha256", ""),
            version=__version__,
            setting=parsed.setting,
            chunks=_chunks(code_set),
            expansions=tuple(
                Expansion(
                    written=e.written,
                    expansion=e.expansion,
                    start=e.start,
                    end=e.end,
                    alternatives=e.alternatives,
                    chosen_by=e.chosen_by,
                )
                for e in parsed.expansions
            ),
            codes=tuple(
                AssignedCode(
                    code=item.code,
                    description=item.description,
                    role=item.role,
                    primary=number == 0,
                    source=item.source,
                    chunks=item.conditions,
                    path=_path_of(item.code, coded),
                )
                for number, item in enumerate(code_set.sequence)
            ),
            alternatives=_refs(primary.alternates if primary else (), graph),
            confidence=code_set.confidence,
            escalated=code_set.escalated,
            resolver=code_set.resolver,
            cached=cached,
            flags=code_set.flags,
            trace=tuple(trace),
            llm_calls=0 if cached else code_set.llm_calls,
            cost_usd=0.0 if cached else code_set.cost_usd,
            detail=_detail(code_set, coded, graph) if explain else None,
        )


def _chunks(code_set: CodeSet) -> tuple[Chunk, ...]:
    parsed = code_set.parsed
    original = parsed.original

    def mark(kind: str, value: str, start: int, end: int) -> Mark:
        a, b = parsed.source(start, end)
        return Mark(kind=kind, value=value, text=original[a:b], start=a, end=b)

    chunks = []
    for number, condition in enumerate(parsed.conditions):
        start, end = parsed.source(condition.start, condition.end)
        chunks.append(
            Chunk(
                id=number,
                text=original[start:end],
                start=start,
                end=end,
                condition=condition.text,
                status=condition.status,
                certainty=condition.certainty,
                reportable=condition.reportable,
                diagnosis=condition.diagnosis,
                modifiers=tuple(mark(m.kind, m.value, m.start, m.end) for m in condition.modifiers),
                cues=tuple(mark(c.kind, c.position, c.start, c.end) for c in condition.cues),
            )
        )
    return tuple(chunks)


def _path_of(code: str, coded: tuple[CodedDiagnosis, ...]) -> int | None:
    return next(
        (
            i
            for i, diagnosis in enumerate(coded)
            if code in (diagnosis.code, diagnosis.manifestation)
        ),
        None,
    )


def _refs(codes: tuple[str, ...], graph: Graph) -> tuple[CodeRef, ...]:
    return tuple(CodeRef(code=code, description=graph.code(code).description) for code in codes)


def _detail(code_set: CodeSet, coded: tuple[CodedDiagnosis, ...], graph: Graph) -> Detail:
    paths = tuple(
        DiagnosisPath(
            code=diagnosis.code,
            description=diagnosis.description,
            manifestation=diagnosis.manifestation,
            role=diagnosis.role,
            chunks=diagnosis.conditions,
            book=diagnosis.candidate.book,
            written=diagnosis.candidate.written,
            path=diagnosis.candidate.path,
            steps=diagnosis.candidate.steps,
            coded_step=diagnosis.candidate.coded_step,
            alternates=_refs(diagnosis.alternates, graph),
            flags=diagnosis.flags,
            trace=diagnosis.trace,
        )
        for diagnosis in coded
    )
    return Detail(
        paths=paths,
        companions=tuple(
            Companion(
                source=c.source,
                kind=c.kind,
                note=c.note,
                status=c.status,
                code=c.code,
                description=c.description,
            )
            for c in code_set.companions
        ),
        excluded=tuple(Exclusion(code=e.code, kept=e.kept, note=e.note) for e in code_set.excluded),
        pool=code_set.pool,
        decisions=code_set.decisions,
    )


def schema() -> dict[str, object]:
    """The JSON Schema of `CodingResult`, as committed in docs/schema/coding-result.json."""
    return CodingResult.model_json_schema()


if __name__ == "__main__":
    sys.stdout.write(json.dumps(schema(), indent=2, ensure_ascii=False) + "\n")
