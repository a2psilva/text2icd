"""The deterministic coder (L1): the Index walk, then Tabular verification of its candidates.

For each diagnosis the walk ranks Index candidates (`walk`); the verifier (`tabular`) completes
them, in rank order, to billable codes. The first candidate that completes is the code; the next
completed ones are its alternates. A diagnosis none of whose candidates completes gets no code and
the flag `needs_review`: an invalid code is never returned (task 6.5). A manifestation code the
Index pairs with the code is completed the same way and sequenced after it (I.A.13).

The Tabular notes then complete the set (`instructions`, tasks 6.6 to 6.8). For each `code first`,
`use additional code` and `code also` instruction of a code: a code of the set it names satisfies
it; otherwise the phrase is searched for one, among the Index paths of the diagnosis that the
phrase documents in full (no undocumented slot): the walk's candidates for a one-condition
diagnosis, and for a diagnosis of several conditions a second walk of each condition ("type 2 DM
with CKD stage 3a": E11.22, then N18.31 from "chronic kidney disease stage 3a"). A mandatory
instruction found nowhere is listed as `required_unresolved`. Excludes1 conflicts are resolved,
and the codes put in coding order.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from text2icd.graph.model import Graph
from text2icd.resolve.instructions import (
    NAMES,
    ROLES,
    Entry,
    Instruction,
    Kind,
    Notes,
    Role,
    resolve_excludes1,
    sequence,
)
from text2icd.resolve.profiles import MISSING
from text2icd.resolve.tabular import (
    REVIEW,
    UNSPECIFIED,
    Completion,
    Episode,
    Evidence,
    TabularVerifier,
)
from text2icd.resolve.walk import (
    DEFAULT_CODE,
    Candidate,
    DiagnosisWalk,
    IndexWalker,
    diagnosis_text,
)
from text2icd.text.model import ParsedPhrase, Setting
from text2icd.trace import TraceStep

ALTERNATES = 4  # completed alternates kept per diagnosis
MAX_COMPANIONS = 8  # companion codes added per phrase
UNRESOLVED: Final = "required_unresolved"
EXCLUDES1 = "excludes1_conflict"


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


class Companion(BaseModel):
    """What a `code first`, `use additional code` or `code also` instruction of a code of the set
    led to: a code added from the phrase, a code already in the set, or nothing found."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str  # the code the instruction applies to
    kind: Kind
    note: str
    status: Literal["added", "present", "required_unresolved"]
    code: str | None = None
    description: str | None = None
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()


class Exclusion(BaseModel):
    """A code an Excludes1 note took out of the set (I.A.12.a)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    kept: str
    note: str


class SequencedCode(BaseModel):
    """One code of the set, in coding order, with why it is there."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    description: str
    role: Role
    source: str | None = None  # the code whose Index pair or note brought it in


class CodeSet(BaseModel):
    """The codes of a phrase in coding order (`sequence`), with what each diagnosis was coded
    from, the instructions of the Tabular notes, and the codes Excludes1 took out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parsed: ParsedPhrase
    diagnoses: tuple[CodedDiagnosis, ...] = ()
    external_causes: tuple[CodedDiagnosis, ...] = ()
    companions: tuple[Companion, ...] = ()
    excluded: tuple[Exclusion, ...] = ()
    sequence: tuple[SequencedCode, ...] = ()
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(coded.code for coded in self.sequence)


@dataclass
class _Group:
    """A coded diagnosis with what its companions are searched in."""

    walk: DiagnosisWalk
    evidence: Evidence
    coded: CodedDiagnosis
    pool: list[Candidate] | None = None


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
        self.notes = Notes(graph)

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
        groups: list[_Group] = []
        external: list[CodedDiagnosis] = []
        flags: set[str] = set()
        for walk in result.diagnoses:
            evidence = self.evidence(result.parsed, walk)
            coded = self._coded(walk.candidates, evidence, walk.trace)
            if coded is None:
                flags.add(REVIEW)
                continue
            groups.append(_Group(walk, evidence, coded))
            cause = self._coded(walk.external_causes, evidence, ())
            if cause is not None:
                external.append(cause.model_copy(update={"role": "external_cause"}))
        if not groups and result.diagnoses:
            flags.add(REVIEW)
        entries = self._entries(groups)
        companions = self._companions(result.parsed, groups, entries)
        entries, excluded, trace = resolve_excludes1(entries, self.notes)
        if excluded:
            flags.add(EXCLUDES1)
        entries, steps = sequence(entries, self.notes)
        trace += steps
        companions += self._etiologies(entries, companions)
        if any(companion.status == UNRESOLVED for companion in companions):
            flags.add(UNRESOLVED)
        ordered = [
            SequencedCode(code=e.code, description=e.description, role=e.role, source=e.source)
            for e in entries
        ]
        ordered += [
            SequencedCode(code=c.code, description=c.description, role="external_cause")
            for c in external
        ]
        unique: dict[str, SequencedCode] = {}
        for item in ordered:
            unique.setdefault(item.code, item)
        return CodeSet(
            parsed=result.parsed,
            diagnoses=tuple(group.coded for group in groups),
            external_causes=tuple(external),
            companions=tuple(companions),
            excluded=tuple(Exclusion(code=e.code, kept=e.kept, note=e.note) for e in excluded),
            sequence=tuple(unique.values()),
            flags=tuple(sorted(flags)),
            trace=tuple(trace),
        )

    def _entries(self, groups: Sequence[_Group]) -> list[Entry]:
        entries: list[Entry] = []
        seen: set[str] = set()
        for number, group in enumerate(groups):
            coded = group.coded
            specified = not ({UNSPECIFIED, DEFAULT_CODE} & set(coded.flags))
            score, used = coded.candidate.score, frozenset(coded.candidate.used)
            if coded.code not in seen:
                seen.add(coded.code)
                entries.append(
                    Entry(
                        coded.code,
                        coded.description,
                        "diagnosis",
                        None,
                        number,
                        score,
                        specified,
                        used,
                    )
                )
            if coded.manifestation and coded.manifestation not in seen:
                seen.add(coded.manifestation)
                description = self.graph.code(coded.manifestation).description
                entries.append(
                    Entry(
                        coded.manifestation,
                        description,
                        "manifestation",
                        coded.code,
                        number,
                        score,
                        True,
                        used,
                    )
                )
        return entries

    def _companions(
        self, parsed: ParsedPhrase, groups: Sequence[_Group], entries: list[Entry]
    ) -> list[Companion]:
        """Read the companion instructions of every code of the set, the codes they add
        included; codes added go into `entries`."""
        out: list[Companion] = []
        added = 0
        position = 0
        while position < len(entries):
            entry = entries[position]
            position += 1
            for instruction in self.notes.instructions(entry.code):
                present = next(
                    (e for e in entries if e.code != entry.code and instruction.covers(e.code)),
                    None,
                )
                if present is not None:
                    out.append(self._companion(entry, instruction, "present", present.code))
                    continue
                found = None
                if added < MAX_COMPANIONS:
                    found = self._search(parsed, groups[entry.group], instruction, entry)
                if found is not None:
                    candidate, completed = found
                    added += 1
                    entries.append(
                        Entry(
                            completed.code,
                            completed.description,
                            ROLES[instruction.kind],
                            entry.code,
                            entry.group,
                            candidate.score,
                            not (
                                {UNSPECIFIED, DEFAULT_CODE} & {*candidate.flags, *completed.flags}
                            ),
                            frozenset(candidate.used),
                        )
                    )
                    step = TraceStep(
                        layer="tabular",
                        action="companion",
                        detail=f'{entry.code}: {NAMES[instruction.kind]} "{instruction.text}"',
                        guideline="I.B.7",
                    )
                    out.append(
                        self._companion(
                            entry,
                            instruction,
                            "added",
                            completed.code,
                            completed.description,
                            flags=tuple(dict.fromkeys((*candidate.flags, *completed.flags))),
                            trace=(step, *candidate.trace, *completed.trace),
                        )
                    )
                elif instruction.mandatory:
                    out.append(self._companion(entry, instruction, UNRESOLVED))
        return out

    def _companion(
        self,
        entry: Entry,
        instruction: Instruction,
        status: Literal["added", "present", "required_unresolved"],
        code: str | None = None,
        description: str | None = None,
        flags: tuple[str, ...] = (),
        trace: tuple[TraceStep, ...] = (),
    ) -> Companion:
        return Companion(
            source=entry.code,
            kind=instruction.kind,
            note=instruction.text,
            status=status,
            code=code,
            description=description,
            flags=flags,
            trace=trace,
        )

    def _search(
        self, parsed: ParsedPhrase, group: _Group, instruction: Instruction, source: Entry
    ) -> tuple[Candidate, Completion] | None:
        """The first Index path the phrase documents in full whose code completes to one the
        instruction names. A mandatory instruction names a code the condition always has, so
        the words of the source's path may document it (CKD, stage 3a); for an optional one the
        path must use a word of its own ("static internal nasal valve collapse" says nothing of
        the deviated septum that J34.82 lists as an underlying cause)."""
        for candidate in self._pool(parsed, group):
            if candidate.penalty >= MISSING or "undocumented_link" in candidate.flags:
                continue
            if candidate.role != "diagnosis" or not instruction.reaches(candidate.code):
                continue
            if not instruction.mandatory and not set(candidate.used) - source.used:
                continue
            completed = self.verifier.complete(candidate.code, group.evidence)
            if (
                completed is not None
                and completed.code != source.code
                and instruction.covers(completed.code)
            ):
                return candidate, completed
        return None

    def _pool(self, parsed: ParsedPhrase, group: _Group) -> list[Candidate]:
        """The Index paths companions are looked for in: the walk's, and for a diagnosis of
        several conditions, the paths that use all its words and those of each condition alone
        (so that "type 2" never gives a stage 2)."""
        if group.pool is None:
            walk = group.walk
            members = [i for i in walk.conditions if i < len(parsed.conditions)]
            current = all(parsed.conditions[i].status == "current" for i in members)
            if len(members) < 2 or not current:
                group.pool = list(walk.candidates)
            else:
                group.pool = [c for c in walk.candidates if not c.unused]
                for i in members:
                    if not parsed.conditions[i].reportable:
                        continue
                    text = diagnosis_text(parsed, (i,))
                    if text:
                        group.pool += self.walker.walk_text(text, (i,)).candidates
        return group.pool

    def _etiologies(
        self, entries: Sequence[Entry], companions: Sequence[Companion]
    ) -> list[Companion]:
        """A code in diseases classified elsewhere with no other code before it: its etiology is
        missing, and its `code first` instructions are required whatever their wording
        (I.A.13)."""
        out: list[Companion] = []
        listed = {(c.source, c.note) for c in companions}
        for position, entry in enumerate(entries):
            if not self.notes.classified_elsewhere(entry.code):
                continue
            if any(not self.notes.classified_elsewhere(e.code) for e in entries[:position]):
                continue
            first = [i for i in self.notes.instructions(entry.code) if i.kind == "codeFirst"]
            for instruction in first or ():
                if (entry.code, instruction.text) not in listed:
                    out.append(self._companion(entry, instruction, UNRESOLVED))
            if not first:
                out.append(
                    Companion(
                        source=entry.code,
                        kind="codeFirst",
                        note="the underlying disease (etiology/manifestation convention)",
                        status=UNRESOLVED,
                    )
                )
        return out

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
    """Readable lines: the codes in order, each diagnosis's path, flags and trace, then what the
    Tabular notes added, excluded or left unresolved."""
    lines = [f"normalized: {code_set.parsed.normalized}"]
    if not code_set.diagnoses:
        lines.append("no code")
    else:
        lines.append("codes: " + ", ".join(_role(coded) for coded in code_set.sequence))
    for number, coded in enumerate((*code_set.diagnoses, *code_set.external_causes), start=1):
        code = coded.code + (f" [{coded.manifestation}]" if coded.manifestation else "")
        role = "" if coded.role == "diagnosis" else f" ({coded.role.replace('_', ' ')})"
        lines.append(f"{number}. {code}  {coded.description}{role}")
        lines.append(f"   from {coded.candidate.written} at {coded.candidate.path}")
        if coded.flags:
            lines.append(f"   flags: {', '.join(coded.flags)}")
        if coded.alternates:
            lines.append(f"   alternates: {', '.join(coded.alternates)}")
        lines += _steps(coded.trace)
    for companion in code_set.companions:
        name = NAMES[companion.kind]
        if companion.status == "added":
            lines.append(
                f"+ {companion.code}  {companion.description} ({name} at {companion.source})"
            )
            if companion.flags:
                lines.append(f"   flags: {', '.join(companion.flags)}")
            lines += _steps(companion.trace)
        elif companion.status == "present":
            lines.append(f"= {companion.code} answers {name} at {companion.source}")
        else:
            lines.append(f'? {companion.source}: {name} "{companion.note}" (not in the phrase)')
    for excluded in code_set.excluded:
        lines.append(f'- {excluded.code} (Excludes1 with {excluded.kept}: "{excluded.note}")')
    lines += _steps(code_set.trace, "   ")
    if code_set.flags:
        lines.append(f"flags: {', '.join(code_set.flags)}")
    return lines


def _role(coded: SequencedCode) -> str:
    return (
        coded.code
        if coded.role == "diagnosis"
        else f"{coded.code} ({coded.role.replace('_', ' ')})"
    )


def _steps(trace: Sequence[TraceStep], indent: str = "     ") -> list[str]:
    return [
        f"{indent}{step.layer}/{step.action}: {step.detail}"
        + (f"  ({step.guideline})" if step.guideline else "")
        for step in trace
    ]
