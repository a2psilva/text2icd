"""Tabular verification: from an Index candidate, often a category or a partial code, down to a
billable code of the release (Guidelines I.A.4, I.A.5, I.A.9, I.B.1, I.B.2, I.B.18, I.C.19).

The coder never codes from the Index alone: the Tabular List decides. From the candidate's node
the verifier descends one level at a time until the code is billable:

- each child is scored by the words that set it apart from its parent, in its description or in
  one of its inclusion terms: 1 for each such word the phrase holds, minus `MISSING` for each it
  does not ("Malignant neoplasm of upper lobe, *right* bronchus or lung");
- when no child scores above zero, the phrase gives no further detail: the "unspecified" child is
  taken (I.A.9.b, I.B.18), flagged `unspecified_chosen`; when there is none, the best scoring
  child, flagged `needs_review`;
- a fracture not documented as displaced or nondisplaced is coded as displaced (I.C.19.c.1);
- the 7th character comes from the definition the code inherits (I.A.5): for injuries the
  episode of care the phrase states (initial, subsequent, sequela), for fractures also open or
  closed (closed when not stated) and the healing (routine when not stated); when the phrase
  states no episode, the configured default, flagged `seventh_character_assumed`. Other 7th
  characters (eye, fetus) are chosen by their words like any child. Placeholder `X`s (I.A.4) are
  part of the generated codes, so the descent writes them.

A code that cannot be completed to a billable one gives no completion: the caller tries the next
candidate (task 6.5), and never returns an invalid code.
"""

import re
from collections import defaultdict
from collections.abc import Sequence, Set
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from text2icd.graph.model import Code, Graph
from text2icd.resolve.profiles import MISSING
from text2icd.resolve.words import WordForms, default_word_forms
from text2icd.text.model import Modifier
from text2icd.trace import TraceStep

Episode = Literal["initial", "subsequent", "sequela"]
EPISODES: tuple[Episode, ...] = ("initial", "subsequent", "sequela")
UNSPECIFIED = "unspecified_chosen"
ASSUMED = "seventh_character_assumed"
REVIEW = "needs_review"
TIE = "tabular_tie"  # children tied and the first was taken
_HEALING = {
    "routine_healing": "routine healing",
    "delayed_healing": "delayed healing",
    "nonunion": "nonunion",
    "malunion": "malunion",
}
_TYPE_III = re.compile(r"\b(?:type )?(?:iii[abc]?|3[abc]?)\b")


@dataclass(frozen=True, slots=True)
class Evidence:
    """What the phrase says about one diagnosis: its words, in the forms the walk compares, and
    its modifiers (laterality, episode of care, fracture type)."""

    words: frozenset[str]
    modifiers: tuple[Modifier, ...] = ()
    text: str = ""

    def modifier(self, kind: str) -> list[str]:
        return [m.value for m in self.modifiers if m.kind == kind]


class Completion(BaseModel):
    """A billable code completed from a candidate, with how it was reached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    description: str
    flags: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()


@dataclass
class _Descent:
    flags: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)

    def step(self, action: str, detail: str, guideline: str | None = None) -> None:
        self.trace.append(
            TraceStep(layer="tabular", action=action, detail=detail, guideline=guideline)
        )


class TabularVerifier:
    """Completes Index candidates to billable codes, in one graph."""

    def __init__(
        self, graph: Graph, forms: WordForms | None = None, default_episode: Episode = "initial"
    ) -> None:
        self.graph = graph
        self.forms = forms or default_word_forms()
        self.default_episode = default_episode
        self._texts: dict[str, list[str]] = defaultdict(list)
        for text in graph.tabular_texts():
            if text.kind == "inclusion":
                self._texts[text.node].append(text.text)
        self._words: dict[str, frozenset[str]] = {}

    def inclusions(self, code: str) -> tuple[str, ...]:
        """The inclusion terms of `code`."""
        return tuple(self._texts.get(code, ()))

    def complete(self, code: str, evidence: Evidence) -> Completion | None:
        """The billable code the candidate leads to, or None when it leads to none."""
        if code not in self.graph:
            return None
        node = self.graph.code(code)
        descent = _Descent()
        while not node.billable:
            children = self.graph.children(node.key)
            if not children:
                return None
            if all(child.kind == "generated" for child in children):
                chosen = self._seventh(node, children, evidence, descent)
            else:
                chosen = self._child(node, children, evidence, descent)
            if chosen is None:
                return None
            node = chosen
        if node.key != code:
            descent.step("complete", f"{code} -> {node.key} {node.description}", "I.B.1")
        return Completion(
            code=node.key,
            description=node.description,
            flags=tuple(dict.fromkeys(descent.flags)),
            trace=tuple(descent.trace),
        )

    # -- children -------------------------------------------------------------------------------

    def _child(
        self, parent: Code, children: Sequence[Code], evidence: Evidence, descent: _Descent
    ) -> Code | None:
        above = self._forms(parent.description)
        scored = [(self._score(child, above, evidence.words), child) for child in children]
        best_score = max(score for score, _ in scored)
        if best_score > 0:
            best = [child for score, child in scored if score == best_score]
            return _displaced(best, descent) if len(best) > 1 else best[0]
        unspecified = [  # "unspecified" more often than in the parent: "... of unspecified femur"
            child for child in children if _unspecified(child) > _unspecified(parent)
        ]
        if unspecified:
            chosen = _displaced(unspecified, descent) if len(unspecified) > 1 else unspecified[0]
            descent.flags.append(UNSPECIFIED)
            descent.step(
                "unspecified",
                f"{parent.key}: the phrase gives no further detail: {chosen.key} "
                f"{chosen.description}",
                "I.A.9.b, I.B.18",
            )
            return chosen
        best = [child for score, child in scored if score == best_score]
        chosen = _displaced(best, descent) if len(best) > 1 else best[0]
        descent.flags.append(REVIEW)
        descent.step(
            "no_detail",
            f"{parent.key}: no child is unspecified and none matches; took {chosen.key}",
        )
        return chosen

    def _score(self, child: Code, above: frozenset[str], words: Set[str]) -> float:
        best: float | None = None
        for text in (child.description, *self._texts.get(child.key, ())):
            own = self._forms(text) - above
            score = len(own & words) - MISSING * len(own - words)
            if best is None or score > best:
                best = score
        assert best is not None
        return best

    def _forms(self, text: str) -> frozenset[str]:
        found = self._words.get(text)
        if found is None:
            found = frozenset(self.forms.content(text))
            self._words[text] = found
        return found

    # -- 7th character --------------------------------------------------------------------------

    def _seventh(
        self, parent: Code, children: Sequence[Code], evidence: Evidence, descent: _Descent
    ) -> Code | None:
        meanings = dict(self.graph.seven_char_def(parent.key))
        by_meaning = {meanings.get(child.key[-1], "").casefold(): child for child in children}
        if not any(meaning.startswith(EPISODES) for meaning in by_meaning):
            return self._child(parent, children, evidence, descent)  # eye, fetus, ...
        values = evidence.modifier("episode")
        stated = [episode for episode in EPISODES if episode in values]
        healing = [value for value in values if value in _HEALING]
        episode: Episode
        if stated:
            episode = stated[0]
        elif healing:
            episode = "subsequent"  # healing is only said of a later encounter
        else:
            episode = self.default_episode
            descent.flags.append(ASSUMED)
            descent.step(
                "seventh_character",
                f"{parent.key}: no episode of care stated: {episode} encounter assumed",
                "I.C.19.a",
            )
        wanted = [meaning for meaning in by_meaning if meaning.startswith(episode)]
        if not wanted:
            return None
        fracture = any("fracture" in meaning for meaning in wanted)
        if fracture:
            wanted = self._fracture(wanted, evidence, descent)
        chosen = by_meaning[wanted[0]]
        descent.step("seventh_character", f"{chosen.key[-1]}: {wanted[0]}", "I.A.5")
        return chosen

    def _fracture(self, meanings: list[str], evidence: Evidence, descent: _Descent) -> list[str]:
        """The 7th characters of a fracture, narrowed by open or closed and the healing."""
        kinds = evidence.modifier("fracture")
        if "open" in kinds:
            open_ = [m for m in meanings if "open fracture" in m]
            three = _TYPE_III.search(evidence.text.casefold()) is not None
            typed = [m for m in open_ if ("iii" in m) == three]
            meanings = typed or open_ or meanings
        else:
            closed = [m for m in meanings if "closed fracture" in m]
            if closed and "closed" not in kinds:
                descent.step(
                    "closed", "a fracture not stated as open is coded as closed", "I.C.19.c.1"
                )
            meanings = closed or meanings
        healing = [_HEALING[v] for v in evidence.modifier("episode") if v in _HEALING]
        healed = [m for m in meanings if healing and healing[0] in m]
        if healed:
            return healed
        routine = [m for m in meanings if "routine healing" in m]
        return routine or meanings


def _unspecified(code: Code) -> int:
    return len(re.findall(r"\bunspecified\b", code.description.casefold()))


def _displaced(children: Sequence[Code], descent: _Descent) -> Code:
    """Of children that differ only as displaced or not, the displaced one (I.C.19.c.1)."""
    displaced = [c for c in children if re.search(r"\bdisplaced\b", c.description.casefold())]
    nondisplaced = [c for c in children if "nondisplaced" in c.description.casefold()]
    if displaced and nondisplaced:
        descent.step(
            "displaced",
            "a fracture not stated as displaced or nondisplaced is coded as displaced",
            "I.C.19.c.1",
        )
        return displaced[0]
    descent.flags.append(TIE)
    descent.step("tie", f"{', '.join(c.key for c in children)} tie: took {children[0].key}")
    return children[0]
