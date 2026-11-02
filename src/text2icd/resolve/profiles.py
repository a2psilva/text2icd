"""Index terms as the walk reads them, and how well a term matches the words of a phrase.

A term's essential title is split into comma parts (`Diabetes, diabetic`; `syphilitic,
congenital`), and each part into slots, one per content word. Words joined by "or" share a slot,
since either one fills it (`inhalation of food or vomit`: inhalation, then food or vomit). A
subterm title can open with a connector (`with`, `due to`, `in`), kept apart from its words. The
nonessential modifiers are kept as a set of words, and `below` holds every word of the term and of
the terms under it, which bounds what a walk through it can still use; `size` counts the terms.

`match` scores a term against the phrase words not used yet (Guideline I.A.7):

- each phrase word an essential word uses counts 1;
- each word a nonessential modifier uses counts `NONESSENTIAL` (less than 1, so that when a
  nonessential modifier and a subterm are mutually exclusive the subterm wins), and an absent
  nonessential modifier costs nothing;
- each essential slot of the best part that the phrase does not fill costs `MISSING`: the phrase
  does not document that modifier;
- each other part the phrase leaves empty costs `OTHER_PART`, a little, because commas join both
  variants of one word (`apex, apical`) and qualifiers (`syphilitic, congenital`).
"""

import re
from collections.abc import Iterator, Set
from dataclasses import dataclass, replace

from text2icd.graph.model import Graph, Term
from text2icd.resolve.words import STOPWORDS, WordForms
from text2icd.text.terms import words

NONESSENTIAL = 0.8
MISSING = 0.6
OTHER_PART = 0.05
# Connectors a subterm title can open with, longest first.
CONNECTORS = (
    "associated with",
    "complicated by",
    "resulting from",
    "secondary to",
    "due to",
    "complicating",
    "following",
    "affecting",
    "involving",
    "during",
    "after",
    "from",
    "with",
    "in",
    "of",
    "on",
    "by",
)
_PART = re.compile(r",\s+")


Slot = frozenset[str]


@dataclass(frozen=True, slots=True)
class Profile:
    term: int
    parts: tuple[tuple[Slot, ...], ...]
    nonessential: frozenset[str]
    connector: str | None  # the connector the title opens with
    below: frozenset[str]  # every word of the term and of the terms under it
    size: int = 1  # the term and the terms under it

    @property
    def words(self) -> frozenset[str]:
        return frozenset(word for part in self.parts for slot in part for word in slot)


@dataclass(frozen=True, slots=True)
class Match:
    used: frozenset[str]  # phrase words the essential title uses
    nonessential: frozenset[str]  # phrase words its nonessential modifiers use
    missing: int  # essential slots of the best part the phrase leaves empty
    empty_parts: int  # other parts it leaves empty

    @property
    def score(self) -> float:
        return (
            len(self.used)
            + NONESSENTIAL * len(self.nonessential)
            - MISSING * self.missing
            - OTHER_PART * self.empty_parts
        )

    @property
    def consumed(self) -> frozenset[str]:
        return self.used | self.nonessential


NO_MATCH = Match(frozenset(), frozenset(), 0, 0)


def match(profile: Profile, remaining: Set[str]) -> Match:
    """How the term uses the phrase words in `remaining`."""
    if not profile.parts:
        nonessential = profile.nonessential & remaining
        return Match(frozenset(), frozenset(nonessential), 0, 0)
    best: tuple[float, int, frozenset[str], int] | None = None
    filled_parts: list[frozenset[str]] = []
    for i, part in enumerate(profile.parts):
        filled = frozenset(word for slot in part for word in slot if word in remaining)
        empty = sum(1 for slot in part if not slot & remaining)
        filled_parts.append(filled)
        value = (len(part) - empty) - MISSING * empty
        if best is None or value > best[0]:
            best = (value, i, filled, empty)
    assert best is not None
    _, chosen, used, missing = best
    others = [filled for i, filled in enumerate(filled_parts) if i != chosen]
    used = used.union(*others)
    empty_parts = sum(1 for filled in others if not filled)
    nonessential = (profile.nonessential & remaining) - used
    return Match(used, frozenset(nonessential), missing, empty_parts)


def slots(text: str, forms: WordForms) -> tuple[Slot, ...]:
    """The slots of one part: `inhalation of food or vomit` -> inhalation | food, vomit."""
    found: list[set[str]] = []
    join = False
    for word in words(text):
        if word == "or":
            join = bool(found)
            continue
        if word in STOPWORDS:
            join = False
            continue
        form = forms.form(word)
        if join:
            found[-1].add(form)
        else:
            found.append({form})
        join = False
    return tuple(frozenset(slot) for slot in found)


def split_connector(title: str) -> tuple[str | None, str]:
    """`due to food` -> ("due to", "food"); `with` -> ("with", ""); `aspiration` -> (None, ...)."""
    lowered = title.casefold()
    for connector in CONNECTORS:
        if lowered == connector or lowered.startswith(connector + " "):
            return connector, title[len(connector) :].strip()
    return None, title


def profile(term: Term, forms: WordForms) -> Profile:
    connector, rest = split_connector(term.title) if term.level else (None, term.title)
    parts = tuple(found for part in _PART.split(rest) if (found := slots(part, forms)))
    nonessential = frozenset(
        forms.form(word)
        for modifier in term.nonessential
        for word in words(modifier)
        if word not in STOPWORDS
    )
    return Profile(term.id, parts, nonessential, connector, frozenset())


class Profiles:
    """The profile of every Index term of a graph, and the main terms each word appears in."""

    def __init__(self, graph: Graph, forms: WordForms) -> None:
        self.graph = graph
        self.forms = forms
        own = {term.id: profile(term, forms) for term in graph.terms()}
        below: dict[int, set[str]] = {}
        sizes: dict[int, int] = {}
        for term_id in sorted(own, reverse=True):  # subterms have larger ids than their parents
            found = below.setdefault(term_id, set())
            found |= own[term_id].words | own[term_id].nonessential
            sizes[term_id] = sizes.get(term_id, 0) + 1
            parent = graph.term(term_id).parent
            if parent is not None:
                below.setdefault(parent, set()).update(found)
                sizes[parent] = sizes.get(parent, 0) + sizes[term_id]
        self._profiles = {
            term_id: replace(found, below=frozenset(below[term_id]), size=sizes[term_id])
            for term_id, found in own.items()
        }
        self.vocabulary = frozenset().union(*below.values())  # every form a term uses
        self._main: dict[str, dict[str, list[int]]] = {}
        for term in graph.terms():
            if term.parent is None:
                books = self._main.setdefault(term.book, {})
                for word in self._profiles[term.id].words:
                    books.setdefault(word, []).append(term.id)

    def __getitem__(self, term_id: int) -> Profile:
        return self._profiles[term_id]

    def __iter__(self) -> Iterator[Profile]:
        return iter(self._profiles.values())

    def main_terms_with(self, word: str, book: str = "index") -> tuple[int, ...]:
        """The main terms of `book` whose essential title holds `word`, in Index order."""
        return tuple(self._main.get(book, {}).get(word, ()))
