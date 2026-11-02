"""Subterm navigation in the Alphabetic Index: from a main term down to the code a coder would
take, as ranked candidates with their paths (Guidelines I.A.7, I.A.18).

A phrase is parsed first (`text.parse`); each diagnosis, the conditions that linking relations
join, is walked on its own. Wording a 7th character carries ("initial encounter", a trailing
", sequela") is left to the Tabular List and not walked. Two neighbouring diagnoses are joined
when one walk of both uses every word their separate walks use: the parser takes a comma before a
subterm for a list ("drug abuse with psychosis, delusions"), the coder does not.

The walk starts from the best main terms (`main_terms.MainTermFinder`) and descends with a beam
search (`WIDTH` states per level). A state is a term with the phrase words its path has used:

- a subterm is taken when it uses phrase words not used yet (`profiles.match`): 1 for each word
  its essential title uses, `NONESSENTIAL` for each word a nonessential modifier uses, minus
  `MISSING` for each essential word the phrase does not document (a word used above still fills
  a slot, but scores once);
- a subterm can take a word that a nonessential modifier above it used, gaining the difference:
  when a nonessential modifier and a subentry are mutually exclusive, the subentry wins (I.A.7);
- a subterm that is only a connector (`with`, `due to`, `in`) is passed through when terms under
  it hold a phrase word; one that says `without` needs `without` in the phrase, and under `with`
  no subterm may use a word the phrase puts after `without`;
- a subterm the phrase does not document is passed through, at its cost, when terms under it hold
  a phrase word.

The beam keeps the states whose score plus the phrase words still below them is highest. Every
state is a candidate, coded by its own term or the nearest term above it that gives a code; a term
without a code gives that of its "unspecified" subterm. The main term's code, or an "unspecified"
subterm's, is the default code (I.A.18), flagged `default_code_used`. A subterm that only says
"see" gives no candidate: the coder must follow it. Candidates are ranked by score, then by the
rank of their main term, then deeper first.

Following cross-references, the I.A.15 "with" presumption and the Neoplasm and Drug tables come
with tasks 5.5-5.9; codes are completed and verified in the Tabular List in Phase 6.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from pydantic import BaseModel, ConfigDict

from text2icd.graph.model import Graph, Pointer
from text2icd.resolve.main_terms import Entry, MainTermFinder, PhraseWords, index_speller
from text2icd.resolve.profiles import MISSING, NONESSENTIAL, OTHER_PART, Profile, match
from text2icd.text.model import Condition, ParsedPhrase, Setting
from text2icd.text.parse import Parser
from text2icd.trace import TraceStep, change

WIDTH = 5  # states kept per level, and main terms walked from
MAX_EXPANSIONS = 400  # states expanded per diagnosis
TOP = 10  # candidates kept per diagnosis
CODE_FIELDS = ("code", "subcategory", "see_category")  # pointers that give the code itself
SEVENTH_ONLY = frozenset({"initial", "subsequent"})  # episode values only a 7th character carries
DEFAULT_CODE = "default_code_used"
WITH = frozenset({"with", "associated with", "complicated by"})  # connectors a `without` undoes
_WITHOUT = re.compile(r"\bwithout\b([^,;]*?)(?=[,;]|\b(?:with|due to|secondary to|and)\b|$)")
_UNSPECIFIED = re.compile(r"unspecified\b|NOS$", re.IGNORECASE)


class Candidate(BaseModel):
    """A code the Index leads to for one diagnosis, possibly partial (written with a trailing
    "-"), with the path that leads there."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    partial: bool = False
    manifestation: str | None = None  # the code in brackets, sequenced after `code` (I.A.13)
    score: float
    book: str = "index"
    column: int | None = None  # the table column of a cell
    path: str  # the deepest term reached
    coded_at: str  # the term that gives the code: the deepest term or one above it
    used: tuple[str, ...] = ()  # phrase words the path uses
    unused: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()

    @property
    def written(self) -> str:
        return f"{self.code}-" if self.partial else self.code


class DiagnosisWalk(BaseModel):
    """The walk of one diagnosis: the words walked and the candidates, best first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conditions: tuple[int, ...]  # indexes into ParsedPhrase.conditions
    text: str  # what was walked
    words: tuple[str, ...]  # its words in the forms the walk compares
    candidates: tuple[Candidate, ...] = ()
    trace: tuple[TraceStep, ...] = ()

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


class WalkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parsed: ParsedPhrase
    diagnoses: tuple[DiagnosisWalk, ...] = ()


@dataclass(frozen=True, slots=True)
class _Coded:
    code: Pointer
    manifestation: Pointer | None
    term: int  # the term that gives it
    unspecified: bool = False  # given by an "unspecified" subterm of a term without a code


@dataclass(frozen=True, slots=True)
class _State:
    term: int
    essential: frozenset[str]  # phrase words essential titles on the path use
    nonessential: frozenset[str]  # phrase words only nonessential modifiers use
    penalty: float
    path: tuple[tuple[int, tuple[str, ...]], ...]  # (term, words it took) from the main term
    entry: int  # the rank of the main term walked from
    linked: bool = False  # under a `with` connector

    @property
    def score(self) -> float:
        return len(self.essential) + NONESSENTIAL * len(self.nonessential) - self.penalty


class IndexWalker:
    """Walks the main Index for phrases, in one graph."""

    def __init__(
        self,
        graph: Graph,
        parser: Parser | None = None,
        finder: MainTermFinder | None = None,
        width: int = WIDTH,
    ) -> None:
        self.graph = graph
        self.parser = parser or Parser.from_graph(graph)
        if finder is None:
            known = self.parser.terms.vocabulary if self.parser.terms is not None else ()
            finder = MainTermFinder(graph, speller=index_speller(graph, known))
        self.finder = finder
        self.profiles = finder.profiles
        self.width = width
        self._reaching: dict[int, dict[str, tuple[int, ...]]] = {}
        self._coded: dict[int, _Coded | None] = {}

    @classmethod
    def from_graph(
        cls, graph: Graph, setting: Setting = "inpatient", width: int = WIDTH
    ) -> "IndexWalker":
        return cls(graph, Parser.from_graph(graph, setting), width=width)

    def walk(self, phrase: str) -> WalkResult:
        parsed = self.parser.parse(phrase)
        walks = []
        for members in parsed.diagnoses:
            if not any(parsed.conditions[i].reportable for i in members):
                continue
            text = diagnosis_text(parsed, members)
            if text:
                walks.append(self.walk_text(text, members))
        return WalkResult(parsed=parsed, diagnoses=tuple(self.join(walks)))

    def join(self, walks: Sequence[DiagnosisWalk]) -> list[DiagnosisWalk]:
        """Join neighbouring diagnoses that one Index path codes: when the walk of both together
        uses every word the separate walks use ("abuse with psychosis, delusions")."""
        out: list[DiagnosisWalk] = []
        for walk in walks:
            if out:
                joined = self.walk_text(f"{out[-1].text}, {walk.text}", out[-1].conditions)
                if _covers(joined, out[-1], walk):
                    step = TraceStep(
                        layer="index",
                        action="joined",
                        detail=f'"{out[-1].text}" and "{walk.text}": one Index path codes both',
                    )
                    out[-1] = joined.model_copy(
                        update={
                            "conditions": out[-1].conditions + walk.conditions,
                            "trace": (step, *joined.trace),
                        }
                    )
                    continue
            out.append(walk)
        return out

    def walk_text(self, text: str, conditions: tuple[int, ...] = (0,)) -> DiagnosisWalk:
        """Walk the words of one diagnosis."""
        phrase = self.finder.phrase_words(text)
        trace = [
            TraceStep(layer="index", action="spelling", detail=change(c.written, c.corrected))
            for c in phrase.corrections
        ]
        entries = self.finder.candidates(phrase)[: self.width]
        if entries:
            considered = ", ".join(f"{e.term.title} ({e.score:g})" for e in entries)
            trace.append(TraceStep(layer="index", action="main_terms", detail=considered))
        states = self._search(phrase, entries, self._absent(text))
        candidates = self._candidates(phrase, states)
        return DiagnosisWalk(
            conditions=conditions,
            text=text,
            words=phrase.forms,
            candidates=tuple(candidates),
            trace=tuple(trace),
        )

    # -- search ---------------------------------------------------------------------------------

    def _absent(self, text: str) -> frozenset[str]:
        """The words the phrase says are absent: those after `without`, up to the next comma or
        connector ("cough without pneumonia")."""
        found: set[str] = set()
        for segment in _WITHOUT.finditer(text.casefold()):
            found.update(self.finder.forms.content(segment.group(1)))
        return frozenset(found)

    def _search(
        self, phrase: PhraseWords, entries: Sequence[Entry], absent: frozenset[str]
    ) -> list[_State]:
        words = phrase.remaining
        frontier = [
            _State(
                term=entry.term.id,
                essential=entry.match.used,
                nonessential=entry.match.nonessential,
                penalty=MISSING * entry.match.missing + OTHER_PART * entry.match.empty_parts,
                path=((entry.term.id, tuple(sorted(entry.match.consumed))),),
                entry=rank,
            )
            for rank, entry in enumerate(entries)
        ]
        visited = list(frontier)
        expanded = 0
        while frontier and expanded < MAX_EXPANSIONS:
            children: dict[int, tuple[float, _State]] = {}
            for state in frontier:
                expanded += 1
                for child in self._children(state, words, absent):
                    priority = child.score + len(
                        (words - child.essential - child.nonessential)
                        & self.profiles[child.term].below
                    )
                    best = children.get(child.term)
                    if best is None or (priority, child.score) > (best[0], best[1].score):
                        children[child.term] = (priority, child)
            ranked = sorted(
                children.values(), key=lambda pair: (-pair[0], -pair[1].score, pair[1].term)
            )
            frontier = [state for _, state in ranked[: self.width]]
            visited += frontier
        return visited

    def _children(
        self, state: _State, words: frozenset[str], absent: frozenset[str]
    ) -> Iterable[_State]:
        available = words - state.essential  # words a subterm may still use essentially
        fresh = available - state.nonessential  # words nothing has used yet
        for child in self._reaching_children(state.term, available):
            profile = self.profiles[child]
            if "without" in profile.words and "without" not in available:
                continue
            # Words used above still fill the subterm's slots, but only new words score.
            found = match(profile, words)
            used = found.used - state.essential
            linked = state.linked or profile.connector in WITH
            if linked and used & absent:
                continue  # "with pneumonia" when the phrase says "without pneumonia"
            taken = found.nonessential & fresh
            essential = state.essential | used
            new = _State(
                term=child,
                essential=essential,
                nonessential=(state.nonessential | taken) - essential,
                penalty=state.penalty + MISSING * found.missing + OTHER_PART * found.empty_parts,
                path=(*state.path, (child, tuple(sorted(used | taken)))),
                entry=state.entry,
                linked=linked,
            )
            if new.score > state.score or self._leads_on(profile, available - used):
                yield new

    def _leads_on(self, profile: Profile, words: frozenset[str]) -> bool:
        """Whether terms under the subterm hold one of the words."""
        return any(words & self.profiles[sub.id].below for sub in self.graph.subterms(profile.term))

    def _reaching_children(self, term: int, words: frozenset[str]) -> list[int]:
        """The subterms of `term` with a word of `words` in or under them, in Index order."""
        by_word = self._reaching.get(term)
        if by_word is None:
            collected: dict[str, list[int]] = {}
            for sub in self.graph.subterms(term):
                for word in self.profiles[sub.id].below:
                    collected.setdefault(word, []).append(sub.id)
            by_word = {word: tuple(ids) for word, ids in collected.items()}
            self._reaching[term] = by_word
        found: set[int] = set()
        for word in words:
            found.update(by_word.get(word, ()))
        return sorted(found)

    # -- candidates -----------------------------------------------------------------------------

    def _candidates(self, phrase: PhraseWords, states: Sequence[_State]) -> list[Candidate]:
        ranked: list[tuple[tuple[float, int, bool, int, int], _State, int]] = []
        for state in states:
            at = self._coded_term(state)
            if at is None or (at < len(state.path) - 1 and self._sends_on(state.term)):
                continue  # a "see" the coder must follow, not a default code (task 5.6)
            direct = at == len(state.path) - 1
            key = (-round(state.score, 6), state.entry, not direct, -len(state.path), state.term)
            ranked.append((key, state, at))
        ranked.sort(key=lambda item: item[0])
        out: list[Candidate] = []
        seen: set[tuple[str, str | None]] = set()
        for _, state, at in ranked:
            candidate = self._candidate(phrase, state, at)
            if (candidate.code, candidate.manifestation) in seen:
                continue
            seen.add((candidate.code, candidate.manifestation))
            out.append(candidate)
            if len(out) == TOP:
                break
        return out

    def _coded_term(self, state: _State) -> int | None:
        """The position on the path of the deepest term that gives a code, itself or through its
        "unspecified" subterm."""
        for at in range(len(state.path) - 1, -1, -1):
            if self._code_of(state.path[at][0]) is not None:
                return at
        return None

    def _sends_on(self, term: int) -> bool:
        return any(xref.kind == "see" for xref in self.graph.xrefs(term))

    def _code_of(self, term: int) -> "_Coded | None":
        """The code a term gives; for a term without one, the code of its subterm for the
        unspecified case (`Sarcoma > Kaposi's > unspecified site`)."""
        if term not in self._coded:
            coded = self._own_code(term)
            if coded is None:
                coded = next(
                    (
                        found
                        for sub in self.graph.subterms(term)
                        if _UNSPECIFIED.match(sub.title)
                        and (found := self._own_code(sub.id, unspecified=True)) is not None
                    ),
                    None,
                )
            self._coded[term] = coded
        return self._coded[term]

    def _own_code(self, term: int, unspecified: bool = False) -> "_Coded | None":
        pointers = self.graph.pointers(term)
        code = next((p for p in pointers if p.field in CODE_FIELDS and p.target is not None), None)
        if code is None:
            return None
        manifestation = next(
            (p for p in pointers if p.field == "manifestation" and p.target is not None), None
        )
        return _Coded(code, manifestation, term, unspecified)

    def _candidate(self, phrase: PhraseWords, state: _State, at: int) -> Candidate:
        coded = self._code_of(state.path[at][0])
        assert coded is not None
        code, manifestation = coded.code, coded.manifestation
        coded_at = self.graph.term(coded.term)
        flags: list[str] = []
        trace: list[TraceStep] = []
        for depth, (term_id, took) in enumerate(state.path):
            term = self.graph.term(term_id)
            uses = f" (uses {', '.join(took)})" if took else ""
            action = "main_term" if depth == 0 else "subterm"
            trace.append(TraceStep(layer="index", action=action, detail=f"{term.title}{uses}"))
        if at == 0 or coded.unspecified:
            flags.append(DEFAULT_CODE)
            source = self.graph.term(state.path[at][0])
            detail = (
                f"no subterm of {source.title} the phrase documents gives a code: the "
                f"unspecified case, {code.written}"
                if coded.unspecified
                else f"no subterm of {source.title} gives a code for the rest: {code.written}"
            )
            trace.append(
                TraceStep(layer="index", action="default_code", detail=detail, guideline="I.A.18")
            )
        if code.field == "see_category":
            flags.append("see_category")
        written = code.written + (f" [{manifestation.written}]" if manifestation else "")
        trace.append(
            TraceStep(layer="index", action="code", detail=f"{written} at {coded_at.path}")
        )
        used = state.essential | state.nonessential
        return Candidate(
            code=str(code.target),
            partial=code.partial,
            manifestation=manifestation.target if manifestation else None,
            score=round(state.score, 6),
            book=coded_at.book,
            column=code.column or None,
            path=self.graph.term(state.term).path,
            coded_at=coded_at.path,
            used=tuple(word for word in phrase.forms if word in used),
            unused=tuple(word for word in phrase.forms if word not in used),
            flags=tuple(flags),
            trace=tuple(trace),
        )


def _covers(joined: DiagnosisWalk, first: DiagnosisWalk, second: DiagnosisWalk) -> bool:
    if joined.best is None:
        return False
    used = set(joined.best.used)
    before = set(first.best.used) if first.best else set()
    after = set(second.best.used) if second.best else set()
    return before | after <= used and used > before


def diagnosis_text(parsed: ParsedPhrase, members: Sequence[int]) -> str:
    """The words of one diagnosis: its conditions joined by their connectors, without the wording
    only a 7th character carries."""
    connector: dict[tuple[int, int], str] = {}
    for relation in parsed.relations:
        connector[(relation.head, relation.linked)] = relation.text
        connector.setdefault((relation.linked, relation.head), relation.text)
    parts = [_walked(parsed, parsed.conditions[members[0]])]
    for before, index in pairwise(members):
        link = connector.get((before, index)) or next(
            (r.text for r in parsed.relations if r.linked == index), "with"
        )
        parts += [link, _walked(parsed, parsed.conditions[index])]
    return " ".join(" ".join(part.split()) for part in parts if part.strip()).strip(" ,")


def _walked(parsed: ParsedPhrase, condition: Condition) -> str:
    text = condition.text
    for modifier in condition.modifiers:
        if modifier.kind != "episode":
            continue
        trailing = parsed.normalized[: modifier.start].rstrip().endswith(",")
        if modifier.value in SEVENTH_ONLY or trailing:
            text = text.replace(modifier.text, " ", 1)
    return text.strip(" ,")


def describe_walk(result: WalkResult, top: int = 5) -> list[str]:
    """Readable lines: each diagnosis walked, its candidates and the trace of the best."""
    lines = [f"normalized: {result.parsed.normalized}"]
    if not result.diagnoses:
        lines.append("no diagnosis to walk")
    for number, walk in enumerate(result.diagnoses, start=1):
        lines.append(f"diagnosis {number}: {walk.text}")
        lines.append(f"  words: {' '.join(walk.words)}")
        for step in walk.trace:
            lines.append(f"  {step.action}: {step.detail}")
        if not walk.candidates:
            lines.append("  no candidate")
            continue
        for rank, candidate in enumerate(walk.candidates[:top], start=1):
            code = candidate.written
            if candidate.manifestation:
                code += f" [{candidate.manifestation}]"
            flags = f"  [{', '.join(candidate.flags)}]" if candidate.flags else ""
            lines.append(f"  {rank}. {code:<16} {candidate.score:5.2f}  {candidate.path}{flags}")
        best = walk.candidates[0]
        lines.append("  trace of 1:")
        for step in best.trace:
            guideline = f"  ({step.guideline})" if step.guideline else ""
            lines.append(f"    {step.action}: {step.detail}{guideline}")
    return lines
