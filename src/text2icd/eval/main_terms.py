"""Main-term recall: does Index search put a right main term near the top for a phrase?

A coder starts from a main term of the Alphabetic Index. For a phrase whose gold code is G, the
right main terms are those with a subterm (or themselves) that gives G or a code above G, from
which the coder continues in the Tabular List. Recall@k is the share of phrases for which one of
the first k main terms found is right. Three ways of finding them are measured:

- `main`: search the main terms only;
- `any`: search every term and take the main term each hit belongs to, in order;
- `coder`: the main-term selection of the Index walk (`resolve.main_terms`) on the normalised
  phrase, in the main Index only (the tables come with the Index router).

Used to choose the FTS5 tokenizer of the graph (task 2.8) and to measure main-term selection.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from text2icd.eval.datasets import Example
from text2icd.eval.mimic.audit import wilson
from text2icd.graph.model import Graph
from text2icd.resolve.main_terms import MainTermFinder, index_speller
from text2icd.text.parse import Parser

Mode = Literal["main", "any", "coder"]
MODES: tuple[Mode, ...] = ("main", "any", "coder")
DEPTHS = (1, 5, 10)
_SEARCH_DEPTH = 50  # terms read in `any` mode to find the first ten main terms


@dataclass(frozen=True, slots=True)
class Recall:
    mode: Mode
    k: int
    found: int
    total: int

    def line(self) -> str:
        if not self.total:
            return f"{self.mode} recall@{self.k}: no examples"
        low, high = wilson(self.found, self.total)
        return (
            f"{self.mode:<5} recall@{self.k:<2} {self.found / self.total:.1%} "
            f"(95% CI {low:.1%}-{high:.1%}), {self.found:,} of {self.total:,}"
        )


class MainTerms:
    """Which main terms lead to which codes, for one graph."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._root: dict[int, int] = {}
        self.leading: dict[str, set[int]] = {}
        for term in graph.terms():
            for pointer in graph.pointers(term.id):
                if pointer.target is not None:
                    self.leading.setdefault(pointer.target, set()).add(self.root(term.id))

    def root(self, term_id: int) -> int:
        if term_id not in self._root:
            parent = self.graph.term_parent(term_id)
            self._root[term_id] = term_id if parent is None else self.root(parent.id)
        return self._root[term_id]

    def right(self, example: Example) -> set[int]:
        """The main terms that lead to the example's primary gold code."""
        gold = example.primary
        codes = (gold.code,) if gold.status == "valid" else gold.mapped_to
        found: set[int] = set()
        for code in codes:
            if code not in self.graph:
                continue
            for node in (self.graph.code(code), *self.graph.ancestors(code)):
                found |= self.leading.get(node.key, set())
        return found

    @cached_property
    def coder(self) -> tuple[Parser, MainTermFinder]:
        parser = Parser.from_graph(self.graph)
        known = parser.terms.vocabulary if parser.terms is not None else frozenset()
        return parser, MainTermFinder(self.graph, speller=index_speller(self.graph, known))

    def ranked(self, phrase: str, mode: Mode, k: int) -> list[int]:
        if mode == "coder":
            parser, finder = self.coder
            words = finder.phrase_words(parser.parse(phrase).normalized)
            return [entry.term.id for entry in finder.candidates(words)[:k]]
        if mode == "main":
            return [hit.term for hit in self.graph.search_index(phrase, k, main_terms=True)]
        roots: list[int] = []
        for hit in self.graph.search_index(phrase, _SEARCH_DEPTH):
            root = self.root(hit.term)
            if root not in roots:
                roots.append(root)
            if len(roots) == k:
                break
        return roots


def main_term_recall(
    graph: Graph, examples: Iterable[Example], depths: Sequence[int] = DEPTHS
) -> list[Recall]:
    """Recall at each depth and mode, over the examples whose gold code some main term leads
    to (the others cannot be found through the Index at all)."""
    index = MainTerms(graph)
    deepest = max(depths)
    found = {(mode, k): 0 for mode in MODES for k in depths}
    total = 0
    for example in examples:
        right = index.right(example)
        if not right:
            continue
        total += 1
        for mode in MODES:
            ranked = index.ranked(example.phrase, mode, deepest)
            for k in depths:
                found[(mode, k)] += bool(right & set(ranked[:k]))
    return [Recall(mode, k, found[(mode, k)], total) for mode in MODES for k in depths]
