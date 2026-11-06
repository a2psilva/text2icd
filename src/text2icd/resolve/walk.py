"""Subterm navigation in the Alphabetic Index: from a main term down to the code a coder would
take, as ranked candidates with their paths (Guidelines I.A.7, I.A.18).

A phrase is parsed first (`text.parse`); each diagnosis, the conditions that linking relations
join, is walked on its own. A condition that is not current is looked up under History or
Status ("family history of ...", I.C.21.c). The router (`route`) adds start states in the
Neoplasm Table and the Table of Drugs and Chemicals, in the column the phrase calls for, and an
External Cause Index walk when the phrase states a mechanism of injury (`external_causes`).
A history or family history is answered only by codes of its kind: a family history code for a
family history; for a personal history a Z code, a code the History main term leads to, or a
path that says "previous" itself, never the condition found elsewhere. Under History, "cancer"
reads "malignant neoplasm". Wording a 7th character carries ("initial encounter", a trailing ",
sequela") is left to the Tabular List and not walked, and so is the drug of a diabetes (a Z79
status code). Two neighbouring diagnoses are joined when one walk of both uses every word their
separate walks use, but for a stage of the second that the combination leaves to a `use
additional code` note ("type 2 diabetes, CKD stage 3a", I.A.15): the parser takes a comma before a
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
  no subterm may use a word the phrase puts after `without`; nor may a main term, unless the
  phrase uses the word elsewhere too ("sepsis without septic shock" is no "Shock, septic");
- a subterm the phrase does not document is passed through, at its cost, when terms under it hold
  a phrase word;
- a connector the Index presumes, "with" or "in", links the conditions even when the phrase only
  lists them (I.A.15: "diabetes, CKD" -> `Diabetes > with > chronic kidney disease`), traced as
  `presumed_link`; a causal one, "due to", needs the phrase to say so ("due to", "secondary to",
  "from", "-induced"), and costs `MISSING` otherwise, flagged `undocumented_link`. Diagnoses are
  not joined through an undocumented link;
- cross-references are followed (I.A.16): a `see` jumps to its target, a `see also` adds the
  target as another branch, and the walk goes on there with the words not used yet and those a
  partial reference leaves. The target's title costs what the phrase does not document, and at
  equal scores a code given directly wins. A reference into the Neoplasm Table carries its
  column, whose cells give the codes. No term is visited twice; at most `MAX_JUMPS` per path.

The beam keeps the states whose score plus the phrase words still below them is highest. Every
state is a candidate, coded by its own term or the nearest term above it that gives a code; a term
without a code gives that of its "unspecified" subterm. The main term's code, or an "unspecified"
subterm's, is the default code (I.A.18), flagged `default_code_used`. A subterm that only says
"see" gives no candidate of its own: the coder must follow it. Candidates are ranked by score,
then by fewer cross-references, then by the rank of their main term, then deeper first.

Codes are completed and verified in the Tabular List in Phase 6.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from text2icd.graph.model import Graph, Pointer
from text2icd.ontology.codes import undot
from text2icd.resolve.main_terms import Entry, MainTermFinder, PhraseWords, index_speller
from text2icd.resolve.profiles import MISSING, NONESSENTIAL, OTHER_PART, Profile, match
from text2icd.resolve.route import Route, route
from text2icd.text.model import Condition, ParsedPhrase, Setting
from text2icd.text.parse import Parser
from text2icd.trace import TraceStep, change

WIDTH = 5  # states kept per level, and main terms walked from
MAX_EXPANSIONS = 400  # states expanded per diagnosis
TOP = 10  # candidates kept per diagnosis
MAX_JUMPS = 3  # cross-references followed per path
CODE_FIELDS = ("code", "subcategory", "see_category")  # pointers that give the code itself
SEVENTH_ONLY = frozenset({"initial", "subsequent"})  # episode values only a 7th character carries
DEFAULT_CODE = "default_code_used"
BOOKS = {
    "index": "Index",
    "neoplasm": "Neoplasm Table",
    "drug": "Table of Drugs and Chemicals",
    "eindex": "External Cause Index",
}
# How a coder looks up a condition that is not current: under History or Status (I.C.21.c).
FAMILY_HISTORY = ("Z80", "Z81", "Z82", "Z83", "Z84")
_HISTORY_WORDS = re.compile(r"\b(?:previous|prior|old|past|history)\b")
_CANCER = re.compile(r"\b(?:cancers?|malignanc(?:y|ies))\b")
STATUS_LOOKUP = {
    "history": ("personal history of", "I.C.21.c.4"),
    "resolved": ("personal history of", "I.C.21.c.4"),
    "family_history": ("family history of", "I.C.21.c.4"),
    "status_post": ("status post", "I.C.21.c.3"),
}
WITH = frozenset({"with", "associated with", "complicated by"})  # connectors a `without` undoes
PRESUMED = frozenset({"with", "in", "associated with"})  # links the Index presumes (I.A.15)
CAUSAL = frozenset({"due to", "secondary to", "resulting from"})  # links the phrase must state
_CAUSAL = re.compile(
    r"\b(?:due to|secondary to|resulting from|result of|caused by|because of|related to|"
    r"attributed to|from|induced by)\b|-induced\b"
)
_CONNECTOR_WORDS = re.compile(r"\b(?:associated with|with|in)\b")
_WITHOUT = re.compile(r"\bwithout\b([^,;]*?)(?=[,;]|\b(?:with|due to|secondary to|and)\b|$)")
_UNSPECIFIED = re.compile(r"unspecified\b|NOS$", re.IGNORECASE)
_DIABETES = re.compile(r"\bdiabet")
_DRUG_LINK = r"(?:on|with|using|taking|requiring|treated with|controlled (?:with|on|by)|and)"


class PathStep(BaseModel):
    """One term on a candidate's path through the Index (task 9.11)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    term: int  # the term's id in the release's graph
    title: str
    took: tuple[str, ...] = ()  # the phrase words it uses, in the forms the walk compares
    via: Literal["", "see", "see_also"] = ""  # reached by this cross-reference, if any
    note: str = ""  # that cross-reference, as written
    column: int | None = None  # the table column read from here on


class Candidate(BaseModel):
    """A code the Index leads to for one diagnosis, possibly partial (written with a trailing
    "-"), with the path that leads there."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    partial: bool = False
    manifestation: str | None = None  # the code in brackets, sequenced after `code` (I.A.13)
    score: float
    penalty: float = 0.0  # what the path's undocumented slots cost (`MISSING` each)
    book: str = "index"
    column: int | None = None  # the table column of a cell
    path: str  # the deepest term reached
    coded_at: str  # the term that gives the code: the deepest term or one above it
    used: tuple[str, ...] = ()  # phrase words the path uses
    unused: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()
    role: Literal["diagnosis", "external_cause"] = "diagnosis"
    steps: tuple[PathStep, ...] = ()  # the path, term by term from the main term
    coded_step: int | None = None  # the step whose term (or its "unspecified" subterm) gives it

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
    external_causes: tuple[Candidate, ...] = ()  # when the phrase states one (I.C.20)
    trace: tuple[TraceStep, ...] = ()

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


class WalkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parsed: ParsedPhrase
    diagnoses: tuple[DiagnosisWalk, ...] = ()


@dataclass(frozen=True, slots=True)
class _Context:
    """What the walk of one diagnosis compares against."""

    words: frozenset[str]  # the phrase words, in the forms the walk compares
    absent: frozenset[str]  # words the phrase puts after `without`
    causal: bool  # whether the phrase says one condition causes another ("due to", "from")
    said: frozenset[str]  # the presumed-link connectors the phrase writes
    routes: tuple[Route, ...] = ()
    negated: frozenset[str] = frozenset()  # absent words the phrase uses nowhere else


@dataclass(frozen=True, slots=True)
class _Coded:
    code: Pointer
    manifestation: Pointer | None
    term: int  # the term that gives it
    column: int | None = None  # the table column of a cell
    unspecified: bool = False  # given by an "unspecified" subterm of a term without a code


class _Step(NamedTuple):
    term: int
    took: tuple[str, ...]  # phrase words the term uses
    via: str = ""  # "see" or "see_also" when a cross-reference led here
    column: int | None = None  # the table column the walk reads, in the Neoplasm Table
    note: str = ""  # the cross-reference as written


@dataclass(frozen=True, slots=True)
class _State:
    term: int
    essential: frozenset[str]  # phrase words essential titles on the path use
    nonessential: frozenset[str]  # phrase words only nonessential modifiers use
    penalty: float
    path: tuple[_Step, ...]  # from the main term
    entry: int  # the rank of the main term walked from
    linked: bool = False  # under a `with` connector
    given: frozenset[str] = frozenset()  # words a partial cross-reference leaves to continue with
    jumps: int = 0  # cross-references followed
    route: int = -1  # the table route it started from, or -1 for the main Index

    @property
    def score(self) -> float:
        return len(self.essential) + NONESSENTIAL * len(self.nonessential) - self.penalty

    @property
    def column(self) -> int | None:
        return self.path[-1].column


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
        self._coded: dict[tuple[int, int | None], _Coded | None] = {}

    @classmethod
    def from_graph(
        cls, graph: Graph, setting: Setting = "inpatient", width: int = WIDTH
    ) -> "IndexWalker":
        return cls(graph, Parser.from_graph(graph, setting), width=width)

    def walk(self, phrase: str, setting: Setting | None = None) -> WalkResult:
        """`setting`: inpatient or outpatient for this phrase (default: the parser's)."""
        parsed = self.parser.parse(phrase, setting)
        walks = []
        for members in parsed.diagnoses:
            if not any(parsed.conditions[i].reportable for i in members):
                continue
            text = diagnosis_text(parsed, members)
            if not text:
                continue
            status = parsed.conditions[members[0]].status
            lookup = STATUS_LOOKUP.get(status)
            if lookup is None:
                walks.append(self.walk_text(text, members))
                continue
            words, guideline = lookup
            if status != "status_post":
                # "Cancer, see Neoplasm, malignant": under History the Index says "malignant
                # neoplasm".
                text = _CANCER.sub("malignant neoplasm", text)
            walk = self.walk_text(f"{words} {text}", members)
            if status in ("history", "resolved") and self.finder.forms.form("family") in walk.words:
                status = "family_history"  # "famizy history of", read through its typo
            kept = tuple(c for c in walk.candidates if _status_code(status, c))
            walk = walk.model_copy(update={"candidates": kept})
            step = TraceStep(
                layer="index",
                action="status",
                detail=f'{status.replace("_", " ")}: looked up as "{words} {text}"',
                guideline=guideline,
            )
            walks.append(walk.model_copy(update={"trace": (step, *walk.trace)}))
        return WalkResult(parsed=parsed, diagnoses=tuple(self.join(walks, parsed)))

    def join(
        self, walks: Sequence[DiagnosisWalk], parsed: ParsedPhrase | None = None
    ) -> list[DiagnosisWalk]:
        """Join neighbouring diagnoses that one Index path codes: when the walk of both together
        uses every word the separate walks use ("abuse with psychosis, delusions"), but for the
        stage of the second, which the combination code leaves to a `use additional code` note
        ("type 2 diabetes, CKD stage 3a": E11.22, then N18.31; I.A.15)."""
        out: list[DiagnosisWalk] = []
        for walk in walks:
            if out:
                joined = self.walk_text(f"{out[-1].text}, {walk.text}", out[-1].conditions)
                found = _covering(joined, out[-1], walk, self._stage_words(parsed, walk))
                if found is not None:
                    step = TraceStep(
                        layer="index",
                        action="joined",
                        detail=f'"{out[-1].text}" and "{walk.text}": one Index path codes both',
                    )
                    others = tuple(c for c in joined.candidates if c is not found)
                    out[-1] = joined.model_copy(
                        update={
                            "conditions": out[-1].conditions + walk.conditions,
                            "candidates": (found, *others),
                            "trace": (step, *joined.trace),
                        }
                    )
                    continue
            out.append(walk)
        return out

    def _stage_words(self, parsed: ParsedPhrase | None, walk: DiagnosisWalk) -> frozenset[str]:
        if parsed is None:
            return frozenset()
        return frozenset(
            word
            for i in walk.conditions
            if i < len(parsed.conditions)
            for m in parsed.conditions[i].modifiers
            if m.kind == "stage"
            for word in self.finder.forms.content(m.text)
        )

    def walk_text(self, text: str, conditions: tuple[int, ...] = (0,)) -> DiagnosisWalk:
        """Walk the words of one diagnosis."""
        phrase = self.finder.phrase_words(text)
        trace = [
            TraceStep(layer="index", action="spelling", detail=change(c.written, c.corrected))
            for c in phrase.corrections
        ]
        routes = tuple(route(_corrected(text, phrase)))
        entries = self.finder.candidates(phrase)[: self.width]
        if entries:
            considered = ", ".join(f"{e.term.title} ({e.score:g})" for e in entries)
            trace.append(TraceStep(layer="index", action="main_terms", detail=considered))
        absent = self._absent(text)
        elsewhere = self.finder.forms.content(_WITHOUT.sub(" ", text.casefold()))
        context = _Context(
            words=phrase.remaining,
            absent=absent,
            causal=bool(_CAUSAL.search(text.casefold())),
            said=frozenset(_CONNECTOR_WORDS.findall(text.casefold())),
            routes=routes,
            negated=absent - frozenset(elsewhere),
        )
        starts = self._starts(entries)
        external: list[Candidate] = []
        for number, found in enumerate(routes):
            trace.append(self._route_step(found))
            if found.book == "eindex":
                external = self._external(phrase, found, number, context)
            else:
                starts += self._routed(phrase, found, number)
        states = self._search(starts, context)
        candidates = self._candidates(phrase, states, context)
        return DiagnosisWalk(
            conditions=conditions,
            text=text,
            words=phrase.forms,
            candidates=tuple(candidates),
            external_causes=tuple(external),
            trace=tuple(trace),
        )

    def _route_step(self, found: Route) -> TraceStep:
        heading = self.graph.column_heading(found.book, found.column) if found.column else None
        into = BOOKS[found.book] + (f" [{heading}]" if heading else "")
        return TraceStep(
            layer="index",
            action="route",
            detail=f'"{found.cue}": {into}',
            guideline=found.guideline,
        )

    def _starts(self, entries: Sequence[Entry], route: int = -1) -> list[_State]:
        return [
            _State(
                term=entry.term.id,
                essential=entry.match.used,
                nonessential=entry.match.nonessential,
                penalty=MISSING * entry.match.missing + OTHER_PART * entry.match.empty_parts,
                path=(_Step(entry.term.id, tuple(sorted(entry.match.consumed))),),
                entry=rank,
                route=route,
            )
            for rank, entry in enumerate(entries)
        ]

    def _routed(self, phrase: PhraseWords, found: Route, number: int) -> list[_State]:
        """Start states in a table: the Neoplasm Table's main term, or the substances of the
        Table of Drugs and Chemicals the rest of the phrase names. The cue words count as used."""
        cue = frozenset(self.finder.forms.content(found.cue)) & phrase.remaining
        if found.book == "neoplasm":
            main = self.graph.main_terms("neoplasm")[0].id
            terms = [(main, cue | (match(self.profiles[main], phrase.remaining).used), 0.0)]
        else:
            substance = self.finder.phrase_words(found.text)
            terms = [
                (
                    entry.term.id,
                    cue | entry.match.used,
                    MISSING * entry.match.missing + OTHER_PART * entry.match.empty_parts,
                )
                for entry in self.finder.candidates(substance, found.book)[: self.width]
            ]
        return [
            _State(
                term=term,
                essential=essential,
                nonessential=frozenset(),
                penalty=penalty,
                path=(_Step(term, tuple(sorted(essential)), column=found.column),),
                entry=-1,  # a routed start goes first at equal scores
                route=number,
            )
            for term, essential, penalty in terms
        ]

    def _external(
        self, phrase: PhraseWords, found: Route, number: int, context: _Context
    ) -> list[Candidate]:
        """External cause codes for the mechanism the phrase states, from the External Cause
        Index; only candidates that use the mechanism's words."""
        cue = frozenset(self.finder.forms.content(found.cue))
        entries = self.finder.candidates(phrase, "eindex")[: self.width]
        states = self._search(self._starts(entries, number), context)
        return [
            candidate.model_copy(update={"role": "external_cause"})
            for candidate in self._candidates(phrase, states, context)
            if cue & set(candidate.used)
        ]

    # -- search ---------------------------------------------------------------------------------

    def _absent(self, text: str) -> frozenset[str]:
        """The words the phrase says are absent: those after `without`, up to the next comma or
        connector ("cough without pneumonia")."""
        found: set[str] = set()
        for segment in _WITHOUT.finditer(text.casefold()):
            found.update(self.finder.forms.content(segment.group(1)))
        return frozenset(found)

    def _search(self, starts: Sequence[_State], context: _Context) -> list[_State]:
        words = context.words
        frontier = list(starts)
        visited = list(frontier)
        expanded = 0
        while frontier and expanded < MAX_EXPANSIONS:
            children: dict[tuple[int, int | None], tuple[float, _State]] = {}
            for state in frontier:
                expanded += 1
                for child in (*self._children(state, context), *self._jumps(state, words)):
                    priority = child.score + len(
                        (words - child.essential - child.nonessential)
                        & self.profiles[child.term].below
                    )
                    key = (child.term, child.column)
                    best = children.get(key)
                    if best is None or (priority, child.score) > (best[0], best[1].score):
                        children[key] = (priority, child)
            ranked = sorted(
                children.values(), key=lambda pair: (-pair[0], -pair[1].score, pair[1].term)
            )
            frontier = [state for _, state in ranked[: self.width]]
            visited += frontier
        return visited

    def _children(self, state: _State, context: _Context) -> Iterable[_State]:
        words = context.words
        available = words - state.essential  # words a subterm may still use essentially
        fresh = available - state.nonessential  # words nothing has used yet
        for child in self._reaching_children(state.term, available | state.given):
            profile = self.profiles[child]
            if "without" in profile.words and "without" not in available:
                continue
            # Words used above, and words a cross-reference gives, fill the subterm's slots; only
            # phrase words not used yet score.
            found = match(profile, words | state.given)
            used = (found.used - state.essential) & words
            linked = state.linked or profile.connector in WITH
            if linked and used & context.absent:
                continue  # "with pneumonia" when the phrase says "without pneumonia"
            # "with" and "in" presume a link (I.A.15); "due to" needs one documented.
            undocumented = profile.connector in CAUSAL and not context.causal
            taken = found.nonessential & fresh
            essential = state.essential | used
            new = _State(
                term=child,
                essential=essential,
                nonessential=(state.nonessential | taken) - essential,
                penalty=state.penalty
                + MISSING * (found.missing + undocumented)
                + OTHER_PART * found.empty_parts,
                path=(*state.path, _Step(child, tuple(sorted(used | taken)), column=state.column)),
                entry=state.entry,
                linked=linked,
                given=state.given - found.used,
                jumps=state.jumps,
                route=state.route,
            )
            follows = bool(found.used & state.given)  # the words a cross-reference gives
            onward = (available - used) | (state.given - found.used)
            if new.score > state.score or follows or self._leads_on(profile, onward):
                yield new

    def _jumps(self, state: _State, words: frozenset[str]) -> Iterable[_State]:
        """Where the term's cross-references lead (I.A.16): a `see` the coder must follow, a `see
        also` as another branch. The walk goes on there with the phrase words not used yet, and
        with the words a partial reference leaves (`see Abortion, by type complicated by,
        sepsis`). The target's own title is matched like a subterm's: what the phrase does not
        document costs. A reference into the Neoplasm Table carries its column. No term is
        visited twice, and at most `MAX_JUMPS` references are followed."""
        if state.jumps >= MAX_JUMPS:
            return
        on_path = {step.term for step in state.path}
        for xref in self.graph.xrefs(state.term):
            if xref.target is None or xref.status not in ("resolved", "partial"):
                continue
            if xref.target in on_path:
                continue
            target = self.graph.term(xref.target)
            if target.book == "drug":
                continue  # the Table of Drugs and Chemicals comes with task 5.9
            column = xref.column if xref.column is not None else state.column
            if target.book != "neoplasm":
                column = None
            elif column is None:
                continue
            given = state.given | frozenset(
                self.finder.forms.content(xref.remainder) if xref.remainder else ()
            )
            found = match(self.profiles[target.id], words | given)
            used = (found.used - state.essential) & words
            essential = state.essential | used
            # A `see` from a term the phrase documents, or that a `see` led to, names the same
            # thing; a `see also`, or a `see` from a term the phrase only partly documents, is
            # charged for what the target adds.
            source = match(self.profiles[state.term], words | state.given)
            partly = bool(source.missing or source.empty_parts) and state.path[-1].via != "see"
            charged = xref.kind == "see_also" or partly
            cost = MISSING * found.missing * charged + OTHER_PART * found.empty_parts
            yield replace(
                state,
                term=target.id,
                essential=essential,
                nonessential=state.nonessential - essential,
                penalty=state.penalty + cost,
                path=(
                    *state.path,
                    _Step(target.id, tuple(sorted(used)), xref.kind, column, xref.written),
                ),
                given=given,
                jumps=state.jumps + 1,
            )

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

    def _candidates(
        self, phrase: PhraseWords, states: Sequence[_State], context: _Context
    ) -> list[Candidate]:
        ranked: list[tuple[tuple[float, int, int, bool, int, int], _State, int]] = []
        for state in states:
            at = self._coded_term(state)
            if at is None or (at < len(state.path) - 1 and self._sends_on(state.term)):
                continue  # a "see" the coder must follow, not a default code (task 5.6)
            if set(state.path[0].took) & context.negated:
                continue  # a main term the phrase says is absent: "sepsis without septic shock"
            direct = at == len(state.path) - 1
            # At equal scores: the better main term, then a code given directly before one a
            # cross-reference leads to.
            key = (
                -round(state.score, 6),
                state.entry,
                state.jumps,
                not direct,
                -len(state.path),
                state.term,
            )
            ranked.append((key, state, at))
        ranked.sort(key=lambda item: item[0])
        # One candidate per code, in rank order; of the paths to a code at its best score, the
        # one with the fewest cross-references.
        chosen: dict[tuple[str, str | None], tuple[_State, int]] = {}
        for key, state, at in ranked:
            coded = self._code_of(state.path[at].term, state.path[at].column)
            assert coded is not None
            code = (str(coded.code.target), coded.manifestation and coded.manifestation.target)
            first = chosen.get(code)
            fewer = first is not None and state.jumps < first[0].jumps
            if first is None or (fewer and round(first[0].score, 6) == -key[0]):
                chosen[code] = (state, at)
        return [
            self._candidate(phrase, state, at, context) for state, at in list(chosen.values())[:TOP]
        ]

    def _coded_term(self, state: _State) -> int | None:
        """The position on the path of the deepest term that gives a code, itself or through its
        "unspecified" subterm."""
        for at in range(len(state.path) - 1, -1, -1):
            if self._code_of(state.path[at].term, state.path[at].column) is not None:
                return at
        return None

    def _sends_on(self, term: int) -> bool:
        return any(xref.kind == "see" for xref in self.graph.xrefs(term))

    def _code_of(self, term: int, column: int | None = None) -> _Coded | None:
        """The code a term gives (in a table, the cell of the column); for a term without one,
        the code of its subterm for the unspecified case (`Sarcoma > Kaposi's > unspecified
        site`)."""
        key = (term, column)
        if key not in self._coded:
            coded = self._own_code(term, column)
            if coded is None:
                coded = next(
                    (
                        found
                        for sub in self.graph.subterms(term)
                        if _UNSPECIFIED.match(sub.title)
                        and (found := self._own_code(sub.id, column, unspecified=True)) is not None
                    ),
                    None,
                )
            self._coded[key] = coded
        return self._coded[key]

    def _own_code(self, term: int, column: int | None, unspecified: bool = False) -> _Coded | None:
        pointers = self.graph.pointers(term)
        if column is None:
            fields: tuple[str, ...] = CODE_FIELDS
        else:
            fields = ("cell",)
        code = next(
            (
                p
                for p in pointers
                if p.field in fields
                and p.target is not None
                and (column is None or p.column == column)
            ),
            None,
        )
        if code is None:
            return None
        manifestation = next(
            (p for p in pointers if p.field == "manifestation" and p.target is not None), None
        )
        return _Coded(code, manifestation, term, column, unspecified)

    def _candidate(
        self, phrase: PhraseWords, state: _State, at: int, context: _Context
    ) -> Candidate:
        coded = self._code_of(state.path[at].term, state.path[at].column)
        assert coded is not None
        code, manifestation = coded.code, coded.manifestation
        coded_at = self.graph.term(coded.term)
        flags: list[str] = []
        trace: list[TraceStep] = []
        if state.route >= 0 and context.routes[state.route].book != "eindex":
            routed = context.routes[state.route]
            flags.extend(routed.flags)
            trace.append(self._route_step(routed))
        steps: list[PathStep] = []
        for depth, step in enumerate(state.path):
            term = self.graph.term(step.term)
            via = "see" if step.via == "see" else "see_also" if step.via == "see_also" else ""
            steps.append(
                PathStep(
                    term=step.term,
                    title=term.title,
                    took=step.took,
                    via=via,
                    note=step.note,
                    column=step.column,
                )
            )
            if step.via:
                column = self.graph.column_heading(term.book, step.column) if step.column else None
                into = f" [{column}]" if column else ""
                detail = f"{step.note} -> {term.path}{into}"
                trace.append(
                    TraceStep(
                        layer="index",
                        action=step.via,
                        detail=detail,
                        guideline="I.A.16",
                        term=step.term,
                    )
                )
                continue
            uses = f" (uses {', '.join(step.took)})" if step.took else ""
            action = "main_term" if depth == 0 else "subterm"
            trace.append(
                TraceStep(
                    layer="index", action=action, detail=f"{term.title}{uses}", term=step.term
                )
            )
            connector = self.profiles[step.term].connector if depth else None
            if connector in PRESUMED and connector not in context.said:
                trace.append(
                    TraceStep(
                        layer="index",
                        action="presumed_link",
                        detail=f'"{connector}" in the Index links the conditions',
                        guideline="I.A.15",
                    )
                )
            elif connector in CAUSAL and not context.causal:
                flags.append("undocumented_link")
                trace.append(
                    TraceStep(
                        layer="index",
                        action="undocumented_link",
                        detail=f'"{connector}": the phrase does not say one causes the other',
                        guideline="I.A.15",
                    )
                )
        if any(step.via for step in state.path):
            flags.append("cross_reference")
        if (coded_at.parent is None and coded_at.book != "drug") or coded.unspecified:
            flags.append(DEFAULT_CODE)
            source = self.graph.term(state.path[at].term)
            detail = (
                f"no subterm of {source.title} the phrase documents gives a code: the "
                f"unspecified case, {code.written}"
                if coded.unspecified
                else f"no subterm of {source.title} gives a code for the rest: {code.written}"
            )
            trace.append(
                TraceStep(
                    layer="index",
                    action="default_code",
                    detail=detail,
                    guideline="I.A.18",
                    term=state.path[at].term,
                    code=code.target,
                )
            )
        if code.field == "see_category":
            flags.append("see_category")
        written = code.written + (f" [{manifestation.written}]" if manifestation else "")
        trace.append(
            TraceStep(
                layer="index",
                action="code",
                detail=f"{written} at {coded_at.path}",
                term=coded.term,
                code=code.target,
            )
        )
        used = state.essential | state.nonessential
        return Candidate(
            code=str(code.target),
            partial=code.partial,
            manifestation=manifestation.target if manifestation else None,
            score=round(state.score, 6),
            penalty=round(state.penalty, 6),
            book=coded_at.book,
            column=coded.column,
            path=self.graph.term(state.term).path,
            coded_at=coded_at.path,
            used=tuple(word for word in phrase.forms if word in used),
            unused=tuple(word for word in phrase.forms if word not in used),
            flags=tuple(flags),
            trace=tuple(trace),
            steps=tuple(steps),
            coded_step=at,
        )


def _status_code(status: str, candidate: Candidate) -> bool:
    """Whether a candidate can answer a status lookup: a family history only a family history
    code; a personal history a Z code, one the History main term leads to (I25.2 for "history
    of MI") or a path that says "previous" itself (O34.219 for "previous cesarean delivery"),
    never the condition itself found elsewhere (C34.90 for "history of lung cancer") nor a
    family history."""
    bare = undot(candidate.code)
    if status == "family_history":
        return bare.startswith(FAMILY_HISTORY)
    if bare.startswith(FAMILY_HISTORY):
        return False
    if status in ("history", "resolved"):
        main = next((s.detail for s in candidate.trace if s.action == "main_term"), "")
        # a Z code, History (see ...), or a path that says "previous" itself (O34.219)
        said = _HISTORY_WORDS.search(candidate.path.casefold()) is not None
        return bare.startswith("Z") or main.startswith("History") or said
    return True


def _corrected(text: str, phrase: PhraseWords) -> str:
    """The text with its typos corrected, so that misspelled cue words still route."""
    for correction in phrase.corrections:
        text = re.sub(
            rf"\b{re.escape(correction.written)}\b", correction.corrected, text, flags=re.IGNORECASE
        )
    return text


def _covering(
    joined: DiagnosisWalk,
    first: DiagnosisWalk,
    second: DiagnosisWalk,
    left: frozenset[str] = frozenset(),
) -> Candidate | None:
    """The joined walk's best candidate that uses every word the separate walks use, but
    `left` (words of the second the combination leaves to a companion code), with no
    undocumented slot or link."""
    before = set(first.best.used) if first.best else set()
    after = set(second.best.used) if second.best else set()
    for rank, candidate in enumerate(joined.candidates):
        used = set(candidate.used)
        if "undocumented_link" in candidate.flags or (rank and candidate.penalty >= MISSING):
            continue
        if before | (after - left) <= used and used > before:
            return candidate
    return None


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
    diabetes = _DIABETES.search(parsed.normalized) is not None
    for modifier in condition.modifiers:
        if modifier.kind == "diabetes_drug" and diabetes:
            # The drug of a diabetes is a status code (Z79, I.C.4.a.3), not Index words.
            text = re.sub(rf"(?:\b{_DRUG_LINK}\s+)?{re.escape(modifier.text)}", " ", text, count=1)
            continue
        if modifier.kind != "episode":
            continue
        trailing = parsed.normalized[: modifier.start].rstrip().endswith(",")
        if modifier.value in SEVENTH_ONLY or trailing:
            text = text.replace(modifier.text, " ", 1)
    return " ".join(text.split()).strip(" ,")


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
        for candidate in walk.external_causes[:1]:
            lines.append(f"  external cause: {candidate.written}  {candidate.path}")
        best = walk.candidates[0]
        lines.append("  trace of 1:")
        for step in best.trace:
            guideline = f"  ({step.guideline})" if step.guideline else ""
            lines.append(f"    {step.action}: {step.detail}{guideline}")
    return lines
