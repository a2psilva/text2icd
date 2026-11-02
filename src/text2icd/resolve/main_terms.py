"""Main-term selection: where in the Alphabetic Index a coder starts for a phrase.

The candidates are the main terms whose essential title holds a word of the phrase, after typo
correction (`words.Speller`; a word whose form an Index term uses is never corrected). When no
main term holds one, Index search (FTS5 BM25 over titles and nonessential modifiers) supplies
them: the main terms of the best-matching terms, in order.

A coder looks up the condition, not the anatomical site (`Fracture, femur`, not `Femur`), and
picks the main term from which the subterms can take up the rest of the statement. So:

- main terms that only send the coder to the condition (`Kidney - see condition`) are dropped;
- the rest are ranked by what the main term itself uses (`profiles.match`) plus the phrase words
  its subterms could still use (`reach`);
- ties go to the main term that holds the head noun of the phrase, the last word before the first
  comma or connector ("aspiration *pneumonia* due to food", "*fracture* of femur"), then to the
  one whose own title uses more, then to the larger entry (`Fracture, traumatic` before
  `Fracture, pathological` when the phrase says neither).

The ranking is an estimate: the subterm walk (`resolve.walk`) settles what the subterms do use.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from text2icd.graph.model import Graph, Term
from text2icd.resolve.profiles import Match, Profiles, match
from text2icd.resolve.words import STOPWORDS, Speller, WordForms, default_word_forms
from text2icd.text.terms import words

FoundBy = Literal["word", "search"]
SEARCH_DEPTH = 50  # terms read from Index search when no main term holds a phrase word
# Where the first segment of a phrase, which holds its head noun, ends.
_HEAD_END = re.compile(
    r"[,;:(]|\b(?:of|with|without|due|in|from|secondary|complicated|and|or|on|following|after|"
    r"during|by|for|associated|caused|involving|affecting|to|vs|versus)\b"
)


@dataclass(frozen=True, slots=True)
class Correction:
    written: str
    corrected: str


@dataclass(frozen=True, slots=True)
class PhraseWords:
    """The content words of a phrase in the forms the walk compares, with typo corrections."""

    text: str
    forms: tuple[str, ...]
    corrections: tuple[Correction, ...] = ()
    head: str | None = None  # the form of the head noun

    @property
    def remaining(self) -> frozenset[str]:
        return frozenset(self.forms)


@dataclass(frozen=True, slots=True)
class Entry:
    """A main term to start from."""

    term: Term
    match: Match  # what the main term's own title uses
    reach: int  # phrase words its subterms could still use
    found_by: FoundBy
    holds_head: bool = False
    size: int = 1  # Index terms in the entry

    @property
    def score(self) -> float:
        return self.match.score + self.reach


class MainTermFinder:
    """Finds and ranks main terms for phrases, in one graph."""

    def __init__(
        self,
        graph: Graph,
        forms: WordForms | None = None,
        profiles: Profiles | None = None,
        speller: Speller | None = None,
    ) -> None:
        self.graph = graph
        self.forms = forms or default_word_forms()
        self.profiles = profiles or Profiles(graph, self.forms)
        self.speller = speller or index_speller(graph)
        self._site_only = frozenset(
            term.id
            for term in graph.terms()
            if term.parent is None
            and not graph.pointers(term.id)
            and any(
                xref.kind == "see" and xref.instruction == "condition"
                for xref in graph.xrefs(term.id)
            )
        )

    def phrase_words(self, text: str) -> PhraseWords:
        forms: list[str] = []
        corrections: list[Correction] = []
        head_end = _HEAD_END.search(text.casefold())
        first = len(words(text[: head_end.start()])) if head_end else None
        head: str | None = None
        for i, word in enumerate(words(text)):
            if word in STOPWORDS:
                continue
            form = self.forms.form(word)
            corrected = None if form in self.profiles.vocabulary else self.speller.correct(word)
            if corrected is not None:
                corrections.append(Correction(word, corrected))
                form = self.forms.form(corrected)
            forms.append(form)
            if first is None or i < first:
                head = form
        return PhraseWords(text, tuple(dict.fromkeys(forms)), tuple(corrections), head)

    def candidates(self, phrase: PhraseWords, book: str = "index") -> list[Entry]:
        """Every main term of `book` to start from, best first."""
        remaining = phrase.remaining
        seen: dict[int, Entry] = {}
        for word in phrase.forms:
            for term_id in self.profiles.main_terms_with(word, book):
                if term_id not in seen and term_id not in self._site_only:
                    seen[term_id] = self._entry(term_id, remaining, phrase.head, "word")
        if not seen and phrase.forms:
            for term_id in self._searched(phrase.text, book):
                seen.setdefault(term_id, self._entry(term_id, remaining, phrase.head, "search"))
        return sorted(seen.values(), key=_rank)

    def _entry(
        self, term_id: int, remaining: frozenset[str], head: str | None, found_by: FoundBy
    ) -> Entry:
        profile = self.profiles[term_id]
        found = match(profile, remaining)
        reach = len((remaining - found.consumed) & profile.below)
        holds_head = head is not None and head in found.consumed
        return Entry(self.graph.term(term_id), found, reach, found_by, holds_head, profile.size)

    def _searched(self, text: str, book: str) -> list[int]:
        roots: list[int] = []
        for hit in self.graph.search_index(text, SEARCH_DEPTH, book=book):
            root = self._root(hit.term)
            if root not in roots and root not in self._site_only:
                roots.append(root)
        return roots

    def _root(self, term_id: int) -> int:
        term = self.graph.term(term_id)
        while term.parent is not None:
            term = self.graph.term(term.parent)
        return term.id


def _rank(entry: Entry) -> tuple[float, bool, float, int, int]:
    return (-entry.score, not entry.holds_head, -entry.match.score, -entry.size, entry.term.id)


def index_speller(graph: Graph, known: Iterable[str] = ()) -> Speller:
    """A speller over the words of every Index title and nonessential modifier; `known` words
    (the Tabular List's, say) are never corrected."""

    def texts() -> list[str]:
        out: list[str] = []
        for term in graph.terms():
            out.append(term.title)
            out.extend(term.nonessential)
        return out

    return Speller.from_texts(texts(), frozenset(known))
