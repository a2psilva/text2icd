"""L3 in the coder (task 8.3): for a phrase L1 and L2 escalate, the LLM navigator chooses where the
graph offers several ways forward that the coder could not tell apart.

The decisions come in a coder's order, each only when it arises, within the navigator's limit of
calls per phrase:

1. **Abbreviation.** An abbreviation has several meanings and no word of the phrase chose one
   (`ambiguous_abbreviation`). Options: its meanings. Another meaning than the lexicon's first
   rewrites the phrase with it, and L1 and L2 code it again; a code set no longer escalated ends
   L3 there.
2. **Main term.** The Index entries of the primary diagnosis (its walk's candidates, completed to
   billable codes) come from several main terms and are more than one question holds
   (`INDEX_OPTIONS`). Options: the main terms, each with its best entry's code. The next question
   keeps the chosen main term's entries.
3. **Index entry.** Options: the Index entries, then the codes L2's retrieval found that no entry
   gives. The chosen one is coded as the first candidate of its diagnosis.
4. **Tabular child.** After an Index entry is chosen, the primary code was completed through a
   node where the phrase does not settle the child (`TabularVerifier.open_choice`). Options: the
   node's children, completed. The chosen child is coded in place of the Index code, and its own
   children may be asked about next.

The coder verifies every choice as it verifies its own candidates, so L3 never writes a code.
When the navigator gives no option (none fits, a limit, an error, no provider), the coder's code
set stays and is flagged `needs_review`; a chosen Index entry, or a meaning that made the code
set confident, clears `escalate`. The confidence stays the one the calibrated layers gave (L3's
choices are not calibrated), and `escalated` stays as L1 and L2 set it, so an evaluation compares
the same escalated phrases with and without L3. The trace records every decision, and the code
set the calls, cost and token usage.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from text2icd.llm.cost import Usage
from text2icd.llm.decisions import Choice, Decision, Offer, Option
from text2icd.llm.navigator import Navigator
from text2icd.resolve.choice import BOOK
from text2icd.resolve.tabular import REVIEW, Completion
from text2icd.resolve.walk import Candidate, DiagnosisWalk, WalkResult
from text2icd.text.model import ParsedPhrase
from text2icd.trace import TraceStep

if TYPE_CHECKING:
    from text2icd.resolve.coder import Coder, CodeSet

INDEX_OPTIONS = 8  # Index entries in one question
INDEX_DEPTH = 16  # Index entries read, to be grouped by main term
RETRIEVED_OPTIONS = 5  # codes found by retrieval, after the Index entries
CHILD_OPTIONS = 12  # children in one question; a node with more is not asked about
SEPARATOR = " > "


@dataclass
class _Run:
    """One phrase's decisions: what they cost and the trace they leave."""

    navigator: Navigator
    phrase: str
    calls: int = 0
    cost_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)
    steps: list[TraceStep] = field(default_factory=list)

    def ask(self, decision: Decision) -> Option | None:
        made = self.navigator.choose(decision, calls=self.calls)
        self.calls += made.calls
        self.cost_usd += made.cost_usd
        self.usage = self.usage + Usage(**made.usage)
        option = decision.option(made.option)
        self.steps.append(_step(decision, made, option))
        return option


@dataclass
class _State:
    result: WalkResult
    code_set: "CodeSet"
    origin: int


def navigate(
    coder: "Coder", navigator: Navigator, result: WalkResult, first: "CodeSet", origin: int
) -> "CodeSet":
    """L3 on an escalated code set: `result`, the walk it was assembled from, and `origin`, the
    diagnosis of the walk its primary code codes."""
    run = _Run(navigator, result.parsed.original)
    state = _State(result, first, origin)
    settled_by_text = _meanings(coder, run, state)
    settled = settled_by_text or _index_entry(coder, run, state)
    if settled and not settled_by_text:
        _tabular_children(coder, run, state)
    current = state.code_set
    flags = current.flags if settled else tuple(sorted({*current.flags, REVIEW}))
    resolver = "llm" if current.codes[:1] != first.codes[:1] else first.resolver
    l2 = [s for s in first.trace if s.layer == "retrieval" and s not in current.trace]
    return current.model_copy(
        update={
            "flags": flags,
            "escalate": not settled,
            "escalated": first.escalated,
            "confidence": current.confidence if settled_by_text else first.confidence,
            "resolver": resolver,
            "trace": (*current.trace, *l2, *run.steps),
            "llm_calls": run.calls,
            "cost_usd": run.cost_usd,
            "usage": run.usage.as_dict(),
        }
    )


def _meanings(coder: "Coder", run: _Run, state: _State) -> bool:
    """Decision 1; True when a meaning made the code set confident."""
    for expansion in state.result.parsed.expansions:
        if expansion.chosen_by != "default":
            continue
        meanings = (expansion.expansion, *expansion.alternatives)
        decision = Decision.of(
            "abbreviation", run.phrase, [Offer(m) for m in meanings], subject=expansion.written
        )
        option = run.ask(decision)
        if option is None or option.text == expansion.expansion:
            continue
        rewritten = rewrite(run.phrase, expansion.written, option.text)
        if rewritten == run.phrase:
            continue
        run.phrase = rewritten
        state.result, state.code_set, state.origin = coder.deterministic(rewritten)
        if not state.code_set.escalate:
            return True
    return False


def _index_entry(coder: "Coder", run: _Run, state: _State) -> bool:
    """Decisions 2 and 3; True when an Index entry or a retrieved code was chosen."""
    if not state.result.diagnoses:
        return False
    parsed = state.result.parsed
    walk = state.result.diagnoses[min(state.origin, len(state.result.diagnoses) - 1)]
    entries = _entries(coder, parsed, walk)
    codes = {completion.code for _, completion in entries}
    retrieved = [
        option
        for option in coder.options(state.code_set)
        if option.node is not None and option.code not in codes
    ][:RETRIEVED_OPTIONS]
    path: tuple[str, ...] = ()
    mains = list(dict.fromkeys(main_term(candidate) for candidate, _ in entries))
    if len(entries) > INDEX_OPTIONS and len(mains) > 1:
        best = {main: next(e for e in entries if main_term(e[0]) == main) for main in mains}
        offers = [Offer(main, c.code, c.description) for main, (_, c) in best.items()]
        main = run.ask(Decision.of("main_term", run.phrase, offers))
        if main is not None:
            entries = [e for e in entries if main_term(e[0]) == main.text]
            path = (main.text,)
    entries = entries[:INDEX_OPTIONS]
    offers = [Offer(c.path, done.code, done.description) for c, done in entries]
    offers += [Offer(o.description, o.code, note="retrieval") for o in retrieved]
    if not offers:
        return False
    chosen = run.ask(Decision.of("subterm", run.phrase, offers, path=path))
    if chosen is None:
        return False
    number = int(chosen.id) - 1
    if number < len(entries):
        candidate = entries[number][0]
    else:
        candidate, _ = coder.first_candidate(parsed, walk, retrieved[number - len(entries)])
    return _recode(coder, run, state, walk, candidate, "subterm")


def _tabular_children(coder: "Coder", run: _Run, state: _State) -> None:
    """Decision 4, as long as the descent has an open node and the navigator chooses."""
    while state.code_set.diagnoses and state.result.diagnoses:
        coded = state.code_set.diagnoses[0]
        walk = state.result.diagnoses[0]
        if coded.candidate not in walk.candidates:
            return
        evidence = coder.evidence(state.result.parsed, walk)
        found = coder.verifier.open_choice(coded.candidate.code, evidence)
        if found is None:
            return
        node, children, taken = found
        offers: list[Offer] = []
        keys: list[str] = []
        for child in children:
            done = coder.verifier.complete(child.key, evidence)
            if done is None or any(done.code == offer.code for offer in offers):
                continue
            offers.append(Offer(child.description, done.code))
            keys.append(child.key)
        if not 1 < len(offers) <= CHILD_OPTIONS:
            return
        decision = Decision.of(
            "tabular_child",
            run.phrase,
            offers,
            subject=f"{node.key} {node.description}",
            path=(coded.candidate.path,),
        )
        chosen = run.ask(decision)
        if chosen is None:
            return
        key = keys[int(chosen.id) - 1]
        if key == taken.key:
            return
        candidate = coded.candidate.model_copy(update={"code": key, "partial": False})
        if not _recode(coder, run, state, walk, candidate, "tabular_child"):
            return


def _recode(
    coder: "Coder", run: _Run, state: _State, walk: DiagnosisWalk, candidate: Candidate, kind: str
) -> bool:
    origin = state.result.diagnoses.index(walk)
    recoded = coder.recode(state.result, origin, candidate)
    if recoded is None:
        detail = f"{candidate.code}: the notes take it out; the coder's code set stays"
        run.steps.append(TraceStep(layer="llm", action=kind, detail=detail))
        return False
    state.result, state.code_set = recoded
    state.origin = 0
    return True


def _entries(
    coder: "Coder", parsed: ParsedPhrase, walk: DiagnosisWalk
) -> list[tuple[Candidate, Completion]]:
    """The walk's Index candidates that complete, one per billable code, best first (not a
    candidate L2 made from retrieval: retrieval's codes are offered as such)."""
    evidence = coder.evidence(parsed, walk)
    out: list[tuple[Candidate, Completion]] = []
    for candidate in walk.candidates:
        if candidate.book == BOOK:
            continue
        done = coder.verifier.complete(candidate.code, evidence)
        if done is None or any(done.code == c.code for _, c in out):
            continue
        out.append((candidate, done))
        if len(out) == INDEX_DEPTH:
            break
    return out


def main_term(candidate: Candidate) -> str:
    return candidate.path.split(SEPARATOR)[0]


def rewrite(phrase: str, written: str, meaning: str) -> str:
    """The phrase with the abbreviation, as written, replaced by its meaning."""
    pattern = rf"(?<![A-Za-z0-9]){re.escape(written)}(?![A-Za-z0-9])"
    return re.sub(pattern, lambda _: meaning, phrase)


def _step(decision: Decision, made: Choice, option: Option | None) -> TraceStep:
    count = f"{len(decision.options)} options"
    if option is None:
        detail = f"{count}: none ({made.reason or 'no option fits'})"
    else:
        cached = " (cached)" if made.by == "cache" else ""
        detail = f"{count}: chose {option.id}{cached}, {option.text}"
        if option.code:
            detail += f": {option.code}"
    return TraceStep(layer="llm", action=decision.kind, detail=detail)
