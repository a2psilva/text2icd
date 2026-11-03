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
instruction found nowhere takes the unspecified code it names when the code's title states the
condition (I13.0: N18.9), or else is listed as `required_unresolved`. Excludes1 conflicts are
resolved, and the codes put in coding order.

The guideline rule pack (`rules`, task 6.9) runs in between: a condition the code of its
diagnosis leaves unused is coded on its own (I.B.9), a condition both acute and chronic gets
both codes (I.B.8), and the hypertension, sepsis, diabetes, drug and sequela rules change the
set and its order.

A code set whose confidence is below the calibration's threshold is escalated. With a retriever
and a table that has a retrieval stage, L2 then runs (`choice`, task 7.6): the codes retrieval
finds compete with the coder's code and alternates, each scored by the table, and the best one
wins. A retrieved code goes through the coder again as the first candidate of its diagnosis, so
that its 7th character, companion codes and Excludes1 notes are read like those of any other
candidate.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from text2icd.graph.model import Graph
from text2icd.ontology.codes import dot
from text2icd.resolve import choice, confidence, rules
from text2icd.resolve.confidence import (
    Calibration,
    CalibrationError,
    RetrievalStage,
    read_calibration,
)
from text2icd.resolve.instructions import (
    NAMES,
    ROLES,
    Entry,
    Instruction,
    Kind,
    Notes,
    Order,
    Role,
    resolve_excludes1,
    sequence,
)
from text2icd.resolve.profiles import MISSING
from text2icd.resolve.retrieve import Retriever
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
    WalkResult,
    diagnosis_text,
)
from text2icd.text.model import Modifier, ParsedPhrase, Setting
from text2icd.trace import TraceStep

Resolver = Literal["index", "retrieval"]

ALTERNATES = 4  # completed alternates kept per diagnosis
MAX_COMPANIONS = 8  # companion codes added per phrase
UNRESOLVED: Final = "required_unresolved"
EXCLUDES1 = "excludes1_conflict"
# Parts of a condition that are no condition of their own: qualifiers, and the wording of a 7th
# character even misspelled ("iniital encounter").
_QUALIFIER = re.compile(
    r"(?:unspecified|nos|nec|other|other specified|not elsewhere classified|"
    r"not otherwise specified|(?:site|type|organism) unspecified|unspecified (?:site|type)|"
    r"\S+ encounter(?: for .+)?|sequela)"
)
_COMBINED = re.compile(r"\b(?:sub)?acute\b.*\bchronic\b|\bchronic\b.*\b(?:sub)?acute\b")


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
    features: dict[str, float] = {}  # what the coder's confidence is computed from (`confidence`)
    confidence: float | None = None  # that the primary code is right; None without a table
    escalate: bool = False  # confidence below the threshold: for the next layer (L2, then L3)
    escalated: tuple[Resolver, ...] = ()  # the layers whose confidence was below their threshold
    resolver: Resolver = "index"  # where the primary code's candidate comes from
    choice: dict[str, float] = {}  # the features L2 chose the primary code by (`choice`)

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
    origin: int = 0  # the diagnosis of the phrase it codes (all or part of)
    whole: bool = True  # the walk of a whole diagnosis, not of one of its conditions


@dataclass(frozen=True, slots=True)
class _Found:
    """A companion code found for an instruction, with what it was found from."""

    code: str
    description: str
    score: float
    specified: bool
    used: frozenset[str]
    flags: tuple[str, ...]
    trace: tuple[TraceStep, ...]


_Alone = tuple[DiagnosisWalk, Evidence, CodedDiagnosis] | None


class Coder:
    """Codes phrases with one graph: Index walk, then Tabular verification."""

    def __init__(
        self,
        graph: Graph,
        walker: IndexWalker | None = None,
        verifier: TabularVerifier | None = None,
        calibration: Calibration | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        """`calibration` turns features into a confidence; by default the shipped table.
        `retriever`, L2's (`retrieve.load_vectors`): it runs on escalated code sets when the
        table has a retrieval stage fitted with the same embedding model."""
        self.graph = graph
        self.walker = walker or IndexWalker(graph)
        self.verifier = verifier or TabularVerifier(graph, self.walker.finder.forms)
        self.notes = Notes(graph)
        self.calibration = calibration if calibration is not None else read_calibration()
        self.retriever = retriever
        stage = self.stage
        if retriever is not None and stage is not None and stage.model != retriever.name:
            raise CalibrationError(
                f"the calibration's retrieval stage was fitted with {stage.model}, the vectors "
                f"are of {retriever.name}: refit it with text2icd eval calibrate"
            )

    @property
    def stage(self) -> RetrievalStage | None:
        """The table's retrieval stage, if any."""
        return self.calibration.retrieval if self.calibration is not None else None

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        setting: Setting = "inpatient",
        default_episode: Episode = "initial",
        calibration: Calibration | None = None,
        retriever: Retriever | None = None,
    ) -> "Coder":
        walker = IndexWalker.from_graph(graph, setting)
        verifier = TabularVerifier(graph, walker.finder.forms, default_episode)
        return cls(graph, walker, verifier, calibration, retriever)

    def code(self, phrase: str) -> CodeSet:
        result = self.walker.walk(phrase)
        code_set, origin = self._assemble(result)
        stage = self.stage
        if not code_set.escalate or self.retriever is None or stage is None:
            return code_set
        return self._retrieve(result, code_set, origin, stage)

    def _assemble(self, result: WalkResult) -> tuple[CodeSet, int]:
        """The code set of a walk, and the diagnosis of the walk its primary code codes."""
        parsed = result.parsed
        groups: list[_Group] = []
        external: list[CodedDiagnosis] = []
        flags: set[str] = set()
        trace: list[TraceStep] = []
        exempt: set[frozenset[int]] = set()  # acute and chronic pairs (I.B.8)
        alone: dict[str, _Alone] = {}  # conditions coded on their own, by their words
        for origin, walk in enumerate(result.diagnoses):
            evidence = self.evidence(parsed, walk)
            coded = self._coded(walk.candidates, evidence, walk.trace)
            if coded is None:
                flags.add(REVIEW)
                continue
            group = _Group(walk, evidence, coded, origin=origin)
            pair, steps = self._acute_and_chronic(parsed, group)
            if len(pair) == 2:
                exempt.add(frozenset({len(groups), len(groups) + 1}))
            groups += pair
            trace += steps
            cause = self._coded(walk.external_causes, evidence, ())
            if cause is not None:
                external.append(cause.model_copy(update={"role": "external_cause"}))
            groups += self._uncoded(parsed, group, alone)
        if not groups and result.diagnoses:
            flags.add(REVIEW)
        entries = self._entries(groups)
        entries, orders, steps = self._rules(parsed, groups, entries, alone)
        trace += steps
        companions = self._companions(parsed, groups, entries)
        entries, excluded, steps = resolve_excludes1(entries, self.notes, exempt)
        trace += steps
        if any(not rules.presumed(e) for e in excluded):
            flags.add(EXCLUDES1)
        orders += rules.orders(entries, [group.origin for group in groups])
        entries, steps = sequence(entries, self.notes, orders)
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
        primary = next(iter(unique), None)
        basis = [(group.walk, group.coded) for group in groups]
        features = confidence.features(parsed, result.diagnoses, basis, primary, flags)
        sure, escalate = None, False
        if self.calibration is not None:
            sure = self.calibration.probability(features)
            escalate = self.calibration.escalate(sure)
        origin = next(
            (g.origin for g in groups if primary in (g.coded.code, g.coded.manifestation)),
            groups[0].origin if groups else 0,
        )
        code_set = CodeSet(
            parsed=result.parsed,
            diagnoses=tuple(group.coded for group in groups),
            external_causes=tuple(external),
            companions=tuple(companions),
            excluded=tuple(Exclusion(code=e.code, kept=e.kept, note=e.note) for e in excluded),
            sequence=tuple(unique.values()),
            flags=tuple(sorted(flags)),
            trace=tuple(trace),
            features=features,
            confidence=sure,
            escalate=escalate,
            escalated=("index",) if escalate else (),
        )
        return code_set, origin

    def options(self, code_set: CodeSet) -> list[choice.Option]:
        """L2's pool for a code set: the coder's primary code and alternates, then the codes
        retrieval finds for the normalised phrase, completed with all its words and modifiers.
        None without a retriever, or for a phrase that states no condition to code (negated,
        "no pneumonia"): retrieval would code what the phrase rules out."""
        parsed = code_set.parsed
        if self.retriever is None or not parsed.normalized.strip():
            return []
        if not code_set.diagnoses and REVIEW not in code_set.flags:
            return []
        forms = self.walker.finder.forms
        text = parsed.normalized
        evidence = Evidence(frozenset(forms.content(text)), _modifiers(parsed), text)
        hits = self.retriever.search([text], choice.NODES)[0]
        alternates = code_set.diagnoses[0].alternates if code_set.diagnoses else ()
        return choice.options(
            list(dict.fromkeys([*code_set.codes[:1], *alternates])),
            code_set.confidence or 0.0,
            hits,
            lambda node: self.verifier.complete(node, evidence),
            self._titles,
            lambda code: self.graph.code(code).description,
            evidence.words,
        )

    def _retrieve(
        self, result: WalkResult, first: CodeSet, origin: int, stage: RetrievalStage
    ) -> CodeSet:
        """L2: the best of the pool by the table's retrieval stage. The coder's code keeps its
        code set; another code is coded again as the first candidate of the diagnosis, which
        goes first, since the code chosen is the primary one."""
        pool = self.options(first)
        chosen = choice.choose(pool, stage)
        if chosen is None:
            return first
        option, sure = chosen
        escalate = sure < stage.threshold
        update = {
            "confidence": sure,
            "escalate": escalate,
            "escalated": ("index", "retrieval") if escalate else ("index",),
            "choice": option.features,
        }
        step = self._choice_step(first, pool, option, sure, stage)
        if option.coder or not result.diagnoses:
            return first.model_copy(update={**update, "trace": (*first.trace, step)})
        origin = min(origin, len(result.diagnoses) - 1)
        walk = result.diagnoses[origin]
        candidate, rest = self._first_candidate(first.parsed, walk, option)
        others = [w for i, w in enumerate(result.diagnoses) if i != origin]
        walks = (walk.model_copy(update={"candidates": (candidate, *rest)}), *others)
        second, _ = self._assemble(result.model_copy(update={"diagnoses": walks}))
        if option.code not in second.codes:  # the notes took it out: keep the coder's set
            coder = next((o for o in pool if o.coder), None)
            sure = stage.probability(coder.features) if coder is not None else 0.0
            escalate = sure < stage.threshold
            update = {
                "confidence": sure,
                "escalate": escalate,
                "escalated": ("index", "retrieval") if escalate else ("index",),
                "choice": coder.features if coder is not None else {},
            }
            return first.model_copy(update={**update, "trace": (*first.trace, step)})
        resolver: Resolver = "retrieval" if candidate.book == choice.BOOK else "index"
        return second.model_copy(
            update={**update, "resolver": resolver, "trace": (*second.trace, step)}
        )

    def _first_candidate(
        self, parsed: ParsedPhrase, walk: DiagnosisWalk, option: choice.Option
    ) -> tuple[Candidate, list[Candidate]]:
        """The candidate the chosen code is coded from, and the walk's others after it: the
        Index candidate that completes to it, or else one made from retrieval."""
        evidence = self.evidence(parsed, walk)
        if option.index:
            for i, candidate in enumerate(walk.candidates):
                completed = self.verifier.complete(candidate.code, evidence)
                if completed is not None and completed.code == option.code:
                    return candidate, [*walk.candidates[:i], *walk.candidates[i + 1 :]]
        titles = self._titles(option.code)
        node = option.node or option.code
        similarity = option.similarity or 0.0
        detail = f'{node} "{self.graph.code(node).description}", similarity {similarity:.2f}'
        if node != option.code:
            detail += f": {option.code}"
        candidate = Candidate(
            code=option.code,
            score=round(similarity, 6),
            book=choice.BOOK,
            path=option.description,
            coded_at=option.description,
            used=tuple(w for w in walk.words if w in titles),
            unused=tuple(w for w in walk.words if w not in titles),
            flags=(choice.RETRIEVED,),
            trace=(TraceStep(layer="retrieval", action="retrieved", detail=detail),),
        )
        return candidate, list(walk.candidates)

    def _choice_step(
        self,
        first: CodeSet,
        pool: Sequence[choice.Option],
        option: choice.Option,
        sure: float,
        stage: RetrievalStage,
    ) -> TraceStep:
        threshold = self.calibration.threshold if self.calibration is not None else 0.0
        before = first.confidence or 0.0
        coder = next((o for o in pool if o.coder), None)
        if option.coder:
            what = "the coder's code kept"
        elif option.index:
            what = "a coder alternate chosen"
        else:
            what = "a retrieved code chosen"
        detail = (
            f"confidence {before:.2f} below {threshold:.2f}: {what}, {option.code} {sure:.2f} "
            f"of {len(pool)} codes"
        )
        if coder is not None and not option.coder:
            detail += f"; the coder's {coder.code} {stage.probability(coder.features):.2f}"
        return TraceStep(layer="retrieval", action="choice", detail=detail)

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
                if rules.owned(instruction):
                    continue
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
                    added += 1
                    entries.append(
                        Entry(
                            found.code,
                            found.description,
                            ROLES[instruction.kind],
                            entry.code,
                            entry.group,
                            found.score,
                            found.specified,
                            found.used,
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
                            found.code,
                            found.description,
                            flags=found.flags,
                            trace=(step, *found.trace),
                        )
                    )
                elif instruction.mandatory:
                    unspecified = self._unspecified(instruction, entry)
                    if unspecified is None:
                        out.append(self._companion(entry, instruction, UNRESOLVED))
                        continue
                    entries.append(
                        Entry(
                            unspecified.code,
                            unspecified.description,
                            ROLES[instruction.kind],
                            entry.code,
                            entry.group,
                            entry.score,
                            False,
                            frozenset(),
                        )
                    )
                    out.append(
                        self._companion(
                            entry,
                            instruction,
                            "added",
                            unspecified.code,
                            unspecified.description,
                            flags=unspecified.flags,
                            trace=unspecified.trace,
                        )
                    )
        return out

    def _unspecified(self, instruction: Instruction, source: Entry) -> _Found | None:
        """The unspecified code a mandatory note names, when the title of the code that holds it
        states the condition (I13.0 "... with heart failure and ... chronic kidney disease":
        I50.9 and N18.9 when neither type nor stage is documented, I.A.9.b)."""
        if instruction.kind != "useAdditionalCode":
            return None  # an etiology to code first is not presumed (I.A.13)
        forms = self.walker.finder.forms
        stated = set(forms.content(source.description))
        unspecified = set(forms.content("unspecified"))
        best: tuple[int, str, str] | None = None
        for code in self._named(instruction):
            description = self.graph.code(code).description
            if "unspecified" not in description.casefold():
                continue
            words = set(forms.content(description)) - unspecified
            if words and words <= stated and (best is None or len(words) > best[0]):
                best = (len(words), code, description)  # the most the title states
        if best is None:
            return None
        _, code, description = best
        step = TraceStep(
            layer="tabular",
            action="companion",
            detail=(
                f'{source.code}: {NAMES[instruction.kind]} "{instruction.text}": {code}, its '
                "title states the condition, the phrase no more"
            ),
            guideline="I.A.9.b",
        )
        return _Found(code, description, source.score, False, frozenset(), (UNSPECIFIED,), (step,))

    def _named(self, instruction: Instruction) -> list[str]:
        """The billable codes an instruction names, at most a few hundred."""
        out: list[str] = []
        stack = []
        for reference in instruction.references:
            start = dot(reference.start)
            if start not in self.graph:
                continue
            parent = self.graph.parent(start)
            siblings = self.graph.children(parent.key) if parent is not None else ()
            stack += [c.key for c in siblings if c.key in reference] or [start]
        seen: set[str] = set()
        while stack and len(out) < 300:
            key = stack.pop(0)
            if key in seen:
                continue
            seen.add(key)
            node = self.graph.code(key)
            if node.billable:
                out.append(key)
            stack += [c.key for c in self.graph.children(key)]
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
    ) -> _Found | None:
        """A diabetes drug of the phrase for a drug-therapy note (I.C.4.a.3); otherwise the
        first Index path the phrase documents in full whose code completes to one the
        instruction names. A mandatory instruction names a code the condition always has, so
        the words of the source's path may document it (CKD, stage 3a); for an optional one the
        path must use a word of its own ("static internal nasal valve collapse" says nothing of
        the deviated septum that J34.82 lists as an underlying cause)."""
        drug = rules.drug_status(instruction, _modifiers(parsed))
        if drug is not None:
            completed = self.verifier.complete(rules.DRUG_STATUS[drug.value], group.evidence)
            if completed is not None:
                step = rules.step("drug_status", f'"{drug.text}": {completed.code}', "I.C.4.a.3")
                return _Found(
                    completed.code,
                    completed.description,
                    source.score,
                    True,
                    frozenset(),
                    completed.flags,
                    (step, *completed.trace),
                )
        found = self._from_pool(parsed, group, instruction, source)
        if found is None:
            return None
        candidate, completed = found
        flags = (*candidate.flags, *completed.flags)
        return _Found(
            completed.code,
            completed.description,
            candidate.score,
            not ({UNSPECIFIED, DEFAULT_CODE} & set(flags)),
            frozenset(candidate.used),
            tuple(dict.fromkeys(flags)),
            (*candidate.trace, *completed.trace),
        )

    def _from_pool(
        self, parsed: ParsedPhrase, group: _Group, instruction: Instruction, source: Entry
    ) -> tuple[Candidate, Completion] | None:
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

    def _pieces(self, parsed: ParsedPhrase, walk: DiagnosisWalk) -> list[tuple[str, int]]:
        """The current conditions of a diagnosis, and the comma-separated parts of each, with
        the condition they are part of: what a coder codes one by one."""
        out = []
        absent = {r.linked for r in parsed.relations if r.kind == "without"}
        for i in walk.conditions:
            if i >= len(parsed.conditions) or i in absent:
                continue  # "severe sepsis without septic shock"
            condition = parsed.conditions[i]
            if not condition.reportable or condition.status != "current":
                continue
            for part in diagnosis_text(parsed, (i,)).split(","):
                part = part.strip()
                if part and not _QUALIFIER.fullmatch(part):
                    out.append((part, i))
        return out

    def _alone(
        self, parsed: ParsedPhrase, text: str, index: int, cache: dict[str, _Alone]
    ) -> _Alone:
        """A part of a diagnosis coded on its own, when the Index codes it in full: a path that
        uses every word of it but its modifiers, with no undocumented slot but the title a
        cross-reference leads to ("cardiomegaly", see "Hypertrophy, cardiac"). A mechanism of
        injury is an external cause, not a part ("hip fracture due to fall")."""
        if text in cache:
            return cache[text]
        condition = parsed.conditions[index]
        modifiers = tuple(
            m
            for m in condition.modifiers
            if m.text in text or m.kind in ("episode", "fracture")  # 7th characters
        )
        finder = self.walker.finder
        forms = finder.forms
        qualifiers = {w for m in modifiers for w in forms.content(m.text)}
        if not finder.names_main_term(set(forms.content(text)) - qualifiers):
            cache[text] = None  # a modifier or a site ("moderate", "skull"): no condition
            return None
        walk = self.walker.walk_text(text, (index,))
        evidence = Evidence(frozenset(forms.content(text)), modifiers, text)
        coded = self._coded(walk.candidates, evidence, walk.trace)
        found: _Alone = None
        if coded is not None and not walk.external_causes:  # a mechanism of injury (I.C.20)
            candidate = coded.candidate
            core = set(walk.words) - qualifiers  # its words as the walk reads them
            penalty = round(candidate.penalty, 6)
            if (
                core
                and core <= set(candidate.used)
                and (
                    penalty < MISSING
                    or (penalty == MISSING and "cross_reference" in candidate.flags)
                )
                and "undocumented_link" not in candidate.flags
            ):
                found = (walk, evidence, coded)
        cache[text] = found
        return found

    def _uncoded(
        self, parsed: ParsedPhrase, group: _Group, cache: dict[str, _Alone]
    ) -> list[_Group]:
        """The conditions of a diagnosis its code leaves entirely unused, each coded on its own:
        a combination code is used alone only when it identifies every condition (I.B.9)."""
        candidate = group.coded.candidate
        unused = set(candidate.unused)
        pieces = self._pieces(parsed, group.walk) if unused else []
        if len(pieces) < 2:
            return []
        if candidate.book not in ("index", choice.BOOK):
            # A table entry is written "substance, form" or "site, part": its parts are names.
            whole = {diagnosis_text(parsed, (i,)) for i in group.walk.conditions}
            pieces = [(text, index) for text, index in pieces if text in whole]
        forms = self.walker.finder.forms
        words = set(group.walk.words)
        titles = self._titles(group.coded.code)
        out = []
        for text, index in pieces:
            found_forms = set(forms.content(text)) & words
            if not found_forms or not found_forms <= unused or found_forms <= titles:
                continue  # used by the path, or identified by the code ("warts due to HPV")
            found = self._alone(parsed, text, index, cache)
            if found is None or found[2].code == group.coded.code:
                continue
            walk, evidence, coded = found
            step = rules.step(
                "uncoded_condition",
                f'"{text}": {group.coded.code} does not code it; coded on its own',
                "I.B.9",
            )
            coded = coded.model_copy(update={"trace": (step, *coded.trace)})
            out.append(_Group(walk, evidence, coded, origin=group.origin, whole=False))
        return out

    def _titles(self, code: str) -> set[str]:
        """The words of the code's title and inclusion terms, and its category's."""
        forms = self.walker.finder.forms
        nodes = [code, *(a.key for a in self.graph.ancestors(code) if a.kind == "diag")]
        texts = [
            text
            for node in nodes
            for text in (self.graph.code(node).description, *self.verifier.inclusions(node))
        ]
        return {word for text in texts for word in forms.content(text)}

    def _acute_and_chronic(
        self, parsed: ParsedPhrase, group: _Group
    ) -> tuple[list[_Group], list[TraceStep]]:
        """A condition both acute and chronic, with no combination code: both codes, the acute
        one first, when the Index has a subentry for each (I.B.8)."""
        walk, coded = group.walk, group.coded
        modifier = next(
            (
                m
                for i in walk.conditions
                if i < len(parsed.conditions)
                for m in parsed.conditions[i].modifiers
                if m.kind == "acuity" and m.value == "acute_on_chronic"
            ),
            None,
        )
        if modifier is None or modifier.text not in walk.text:
            return [group], []
        texts = (coded.description, *self.verifier.inclusions(coded.code))
        if any(_COMBINED.search(text.casefold()) for text in texts):
            return [group], []  # a combination code
        forms = self.walker.finder.forms
        pair: list[_Group] = []
        for word in ("acute", "chronic"):
            text = walk.text.replace(modifier.text, word, 1)
            single = self.walker.walk_text(text, walk.conditions)
            modifiers = tuple(
                m.model_copy(update={"value": word, "text": word}) if m == modifier else m
                for m in group.evidence.modifiers
            )
            evidence = Evidence(frozenset(forms.content(text)), modifiers, text)
            found = self._coded(single.candidates, evidence, single.trace)
            if found is None or not set(forms.content(word)) <= set(found.candidate.used):
                return [group], []  # the Index has no subentry for it
            pair.append(_Group(single, evidence, found, origin=group.origin, whole=False))
        acute, chronic = pair[0].coded.code, pair[1].coded.code
        if acute == chronic:
            return [group], []
        detail = f'"{walk.text}": {acute} and {chronic}, the acute first'
        return pair, [rules.step("acute_and_chronic", detail, "I.B.8")]

    def _facts(
        self, parsed: ParsedPhrase, groups: Sequence[_Group], cache: dict[str, _Alone]
    ) -> list[rules.Fact]:
        """The code of each condition of the phrase on its own: a diagnosis of one condition
        by its code, one of several by each of its parts walked alone."""
        facts = []
        for number, group in enumerate(groups):
            pieces = self._pieces(parsed, group.walk) if group.whole else []
            if len(pieces) < 2 or not rules.hypertensive(group.coded.code):
                # a joined walk's hypertension code is read from its parts, not trusted
                facts.append(rules.Fact(group.coded.code, number))
            if len(pieces) < 2:
                continue
            for text, index in pieces:
                found = self._alone(parsed, text, index, cache)
                if found is not None:
                    facts.append(rules.Fact(found[2].code, number))
        return facts

    def _rules(
        self,
        parsed: ParsedPhrase,
        groups: Sequence[_Group],
        entries: list[Entry],
        cache: dict[str, _Alone],
    ) -> tuple[list[Entry], list[Order], list[TraceStep]]:
        """The guideline rules that change the set before its notes are read (`rules`)."""
        trace: list[TraceStep] = []
        texts = {number: group.walk.text for number, group in enumerate(groups)}
        if any(rules.hypertensive(e.code) for e in entries) or any(
            "hypertens" in text for text in texts.values()
        ):
            entries, steps = rules.apply_hypertension(
                entries, self._facts(parsed, groups, cache), self.graph
            )
            trace += steps
        entries, orders, steps = rules.apply_sepsis(entries, texts, self.verifier, self.notes)
        trace += steps
        trace += rules.diabetes_type(
            entries, {number: group.evidence for number, group in enumerate(groups)}
        )
        return entries, orders, trace

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


def _modifiers(parsed: ParsedPhrase) -> tuple[Modifier, ...]:
    return tuple(m for condition in parsed.conditions for m in condition.modifiers)


def describe_code_set(code_set: CodeSet) -> list[str]:
    """Readable lines: the codes in order, each diagnosis's path, flags and trace, then what the
    Tabular notes added, excluded or left unresolved."""
    lines = [f"normalized: {code_set.parsed.normalized}"]
    if not code_set.diagnoses:
        lines.append("no code")
    else:
        lines.append("codes: " + ", ".join(_role(coded) for coded in code_set.sequence))
    if code_set.confidence is not None:
        after = " after retrieval" if code_set.choice else ""
        low = " (low: escalate)" if code_set.escalate else ""
        lines.append(f"confidence: {code_set.confidence:.2f}{after}{low}")
    for number, coded in enumerate((*code_set.diagnoses, *code_set.external_causes), start=1):
        code = coded.code + (f" [{coded.manifestation}]" if coded.manifestation else "")
        role = "" if coded.role == "diagnosis" else f" ({coded.role.replace('_', ' ')})"
        lines.append(f"{number}. {code}  {coded.description}{role}")
        if coded.candidate.book == choice.BOOK:
            lines.append(f"   retrieved: {coded.candidate.path}")
        else:
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
