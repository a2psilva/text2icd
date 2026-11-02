"""The deterministic coder (L1): the Index walk, then Tabular verification of its candidates.

For each diagnosis the walk ranks Index candidates (`walk`); the verifier (`tabular`) completes
them, in rank order, to billable codes. The first candidate that completes is the code; the next
completed ones are its alternates. A diagnosis none of whose candidates completes gets no code and
the flag `needs_review`: an invalid code is never returned (task 6.5). A manifestation code the
Index pairs with the code is completed the same way and sequenced after it (I.A.13).
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from text2icd.graph.model import Graph
from text2icd.resolve.tabular import REVIEW, Episode, Evidence, TabularVerifier
from text2icd.resolve.walk import Candidate, DiagnosisWalk, IndexWalker
from text2icd.text.model import ParsedPhrase, Setting
from text2icd.trace import TraceStep

ALTERNATES = 4  # completed alternates kept per diagnosis


class CodedDiagnosis(BaseModel):
    """A billable code for one diagnosis, with the Index candidate it came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    description: str
    manifestation: str | None = None
    role: Literal["diagnosis", "external_cause"] = "diagnosis"
    candidate: Candidate
    alternates: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()


class CodeSet(BaseModel):
    """The codes of a phrase: each diagnosis's code (and manifestation), then external causes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parsed: ParsedPhrase
    diagnoses: tuple[CodedDiagnosis, ...] = ()
    external_causes: tuple[CodedDiagnosis, ...] = ()
    flags: tuple[str, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        out: list[str] = []
        for coded in self.diagnoses:
            out.append(coded.code)
            if coded.manifestation:
                out.append(coded.manifestation)
        out += [coded.code for coded in self.external_causes]
        return tuple(dict.fromkeys(out))


class Coder:
    """Codes phrases with one graph: Index walk, then Tabular verification."""

    def __init__(
        self,
        graph: Graph,
        walker: IndexWalker | None = None,
        verifier: TabularVerifier | None = None,
    ) -> None:
        self.graph = graph
        self.walker = walker or IndexWalker(graph)
        self.verifier = verifier or TabularVerifier(graph, self.walker.finder.forms)

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        setting: Setting = "inpatient",
        default_episode: Episode = "initial",
    ) -> "Coder":
        walker = IndexWalker.from_graph(graph, setting)
        return cls(graph, walker, TabularVerifier(graph, walker.finder.forms, default_episode))

    def code(self, phrase: str) -> CodeSet:
        result = self.walker.walk(phrase)
        diagnoses: list[CodedDiagnosis] = []
        external: list[CodedDiagnosis] = []
        flags: set[str] = set()
        for walk in result.diagnoses:
            evidence = self.evidence(result.parsed, walk)
            coded = self._coded(walk.candidates, evidence, walk.trace)
            if coded is None:
                flags.add(REVIEW)
                continue
            diagnoses.append(coded)
            cause = self._coded(walk.external_causes, evidence, ())
            if cause is not None:
                external.append(cause.model_copy(update={"role": "external_cause"}))
        if not diagnoses and result.diagnoses:
            flags.add(REVIEW)
        return CodeSet(
            parsed=result.parsed,
            diagnoses=tuple(diagnoses),
            external_causes=tuple(external),
            flags=tuple(sorted(flags)),
        )

    def evidence(self, parsed: ParsedPhrase, walk: DiagnosisWalk) -> Evidence:
        """What the phrase says about the diagnosis: the words of its conditions, with the
        wording the walk left to the Tabular List ("initial encounter"), and their modifiers."""
        conditions = [parsed.conditions[i] for i in walk.conditions if i < len(parsed.conditions)]
        spans = " ".join(parsed.normalized[c.start : c.end] for c in conditions)
        text = f"{walk.text} {spans}".strip()
        modifiers = tuple(m for condition in conditions for m in condition.modifiers)
        return Evidence(frozenset(self.walker.finder.forms.content(text)), modifiers, text)

    def _coded(
        self,
        candidates: Sequence[Candidate],
        evidence: Evidence,
        walk_trace: Sequence[TraceStep],
    ) -> CodedDiagnosis | None:
        chosen: CodedDiagnosis | None = None
        alternates: list[str] = []
        for candidate in candidates:
            completed = self.verifier.complete(candidate.code, evidence)
            if completed is None:
                continue
            if chosen is None:
                manifestation = None
                flags = [*candidate.flags, *completed.flags]
                if candidate.manifestation:
                    paired = self.verifier.complete(candidate.manifestation, evidence)
                    if paired is not None:
                        manifestation = paired.code
                        flags += paired.flags
                chosen = CodedDiagnosis(
                    code=completed.code,
                    description=completed.description,
                    manifestation=manifestation,
                    role=candidate.role,
                    candidate=candidate,
                    flags=tuple(dict.fromkeys(flags)),
                    trace=(*walk_trace, *candidate.trace, *completed.trace),
                )
            elif completed.code != chosen.code and completed.code not in alternates:
                alternates.append(completed.code)
                if len(alternates) == ALTERNATES:
                    break
        if chosen is None:
            return None
        return chosen.model_copy(update={"alternates": tuple(alternates)})


def describe_code_set(code_set: CodeSet) -> list[str]:
    """Readable lines: the codes, each diagnosis's path and flags, and its trace."""
    lines = [f"normalized: {code_set.parsed.normalized}"]
    if not code_set.diagnoses:
        lines.append("no code")
    for number, coded in enumerate((*code_set.diagnoses, *code_set.external_causes), start=1):
        code = coded.code + (f" [{coded.manifestation}]" if coded.manifestation else "")
        role = "" if coded.role == "diagnosis" else f" ({coded.role.replace('_', ' ')})"
        lines.append(f"{number}. {code}  {coded.description}{role}")
        lines.append(f"   from {coded.candidate.written} at {coded.candidate.path}")
        if coded.flags:
            lines.append(f"   flags: {', '.join(coded.flags)}")
        if coded.alternates:
            lines.append(f"   alternates: {', '.join(coded.alternates)}")
        for step in coded.trace:
            guideline = f"  ({step.guideline})" if step.guideline else ""
            lines.append(f"     {step.layer}/{step.action}: {step.detail}{guideline}")
    if code_set.flags:
        lines.append(f"flags: {', '.join(code_set.flags)}")
    return lines
