"""What text understanding needs from the knowledge graph.

- `vocabulary(graph)`: the case-folded words of the Alphabetic Index (titles and nonessential
  modifiers of the four books) and the Tabular List (descriptions and inclusion terms), without
  diacritics. A token outside it is a misspelling, a truncation or an abbreviation.
- `Terms`, for the condition splitter:
  - **conditions**: lenient forms of the main-term titles of the Index that name conditions. Main
    terms that send the coder elsewhere ("see condition", as anatomical sites and adjectives do)
    are left out, and so are forms made only of body-site words (the words of the Neoplasm
    Table's sites, such as lung or extremity), so that `, left lower extremity` refines the
    condition before it while `, hyperlipidemia` is another diagnosis.
  - **joined**: word sequences of CMS terms written with "and" or "or" (`hand, foot and mouth
    disease`, `abortion or miscarriage`), which are never split.
  - **combinations**: a main term and a subterm under its "with" (`Nausea > with vomiting`,
    `Diabetes > with > chronic kidney disease`): "nausea and vomiting" is one diagnosis.
"""

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from text2icd.graph.model import Graph
from text2icd.graph.xrefs import title_forms

_WORD = re.compile(r"[a-z0-9]+")
_PARENS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_UNSPECIFIC = re.compile(r"\b(?:nec|nos)\b")
_JOINERS = frozenset({"and", "or"})
MAX_JOINED = 12  # longer CMS terms are not looked for in phrases
MAX_NGRAM = 4
# Words of the Neoplasm Table's sites that name conditions, not sites.
CONDITIONS_AMONG_SITES = frozenset(
    {
        "carcinoid",
        "carcinoma",
        "disease",
        "diverticulum",
        "fibrosarcoma",
        "hepatoblastoma",
        "hepatoma",
        "kaposi",
        "lesion",
        "melanoma",
        "mesothelioma",
        "metastatic",
        "neoplasm",
        "neoplastic",
        "nevus",
        "sarcoma",
        "scar",
    }
)
STOPWORDS = frozenset({"a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "and", "or"})
# Main-term words that never name a condition of their own in a phrase ("subsequent encounter").
NOT_CONDITIONS = frozenset({"encounter", "encounters"})


def words(text: str) -> list[str]:
    """The case-folded words of a text, without diacritics: `Schönlein's` is `schonlein`, `s`."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return _WORD.findall("".join(char for char in decomposed if not unicodedata.combining(char)))


def vocabulary(graph: Graph) -> frozenset[str]:
    found: set[str] = set()
    for term in graph.terms():
        found.update(words(term.title))
        for modifier in term.nonessential:
            found.update(words(modifier))
    for text in graph.tabular_texts():
        found.update(words(text.text))
    return frozenset(found)


@dataclass(frozen=True)
class Terms:
    conditions: frozenset[str] = frozenset()
    sites: frozenset[str] = frozenset()
    joined: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)  # by 1st word
    combinations: frozenset[tuple[str, str]] = frozenset()
    vocabulary: frozenset[str] = frozenset()
    # Forms of the titles below each main term, by the main term's forms; tables apart.
    descendants: Mapping[str, frozenset[str]] = field(default_factory=dict)
    table_descendants: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        conditions: Iterable[str] = (),
        joined: Iterable[str] = (),
        combinations: Iterable[tuple[str, str]] = (),
        sites: Iterable[str] = (),
        vocabulary: Iterable[str] = (),
        paths: Iterable[tuple[str, str]] = (),
        table_paths: Iterable[tuple[str, str]] = (),
    ) -> "Terms":
        """Terms from plain lists, for tests and small lexicons. `paths` are (main term, title
        of a term below it) pairs of the Index, `table_paths` those of the two tables."""
        site_words = frozenset(sites)
        # A title and each of its comma parts: "Diabetes, diabetic" gives "diabetes" too.
        forms = {
            form
            for text in conditions
            for part in (text, *text.split(","))
            for form in title_forms(part)
        }
        return cls(
            conditions=frozenset(f for f in forms if not _only_sites(f, site_words)),
            sites=site_words,
            joined=_by_first_word(_sequence(text) for text in joined),
            combinations=frozenset(
                (a, b)
                for left, right in combinations
                for a in title_forms(left)
                for b in title_forms(right)
            ),
            vocabulary=frozenset(vocabulary),
            descendants=_below(paths),
            table_descendants=_below(table_paths),
        )

    @classmethod
    def from_graph(cls, graph: Graph) -> "Terms":
        sites = {word for term in graph.terms("neoplasm") for word in words(term.title)}
        sites -= CONDITIONS_AMONG_SITES
        conditions: list[str] = []
        combinations: list[tuple[str, str]] = []
        for term in graph.main_terms("index"):
            heads = [term.title, *term.title.split(",")]
            if not any(xref.instruction == "condition" for xref in graph.xrefs(term.id)):
                conditions.append(term.title)
            for sub in graph.subterms(term.id):
                lowered = sub.title.casefold()
                if lowered.startswith("with "):
                    combinations += [(head, sub.title[5:]) for head in heads]
                elif lowered == "with":
                    combinations += [
                        (head, below.title) for below in graph.subterms(sub.id) for head in heads
                    ]
        joined = [term.title for term in graph.terms() if _has_joiner(term.title)]
        joined += [text.text for text in graph.tabular_texts() if _has_joiner(text.text)]
        paths: list[tuple[str, str]] = []
        table_paths: list[tuple[str, str]] = []
        for term in graph.terms():
            if term.parent is None:
                continue
            main = graph.term(_root(graph, term.id))
            target = table_paths if term.book in ("drug", "neoplasm") else paths
            target.append((main.title, term.title))
        return cls.build(
            conditions, joined, combinations, sites, vocabulary(graph), paths, table_paths
        )

    def is_condition(self, text: str) -> bool:
        """Whether some words of `text` form the title of a condition main term."""
        found = [w for w in words(text) if w not in STOPWORDS and w not in NOT_CONDITIONS]
        for size in range(min(MAX_NGRAM, len(found)), 0, -1):
            for start in range(len(found) - size + 1):
                gram = found[start : start + size]
                if all(word in self.sites for word in gram):
                    continue
                if title_forms(" ".join(gram)) & self.conditions:
                    return True
        return False

    def continues(self, previous: str, fragment: str) -> bool:
        """Whether `fragment` reads as the next step of an Index path that `previous` starts
        (`anemia, hemolytic`: Anemia > hemolytic) or, for the two tables, ends (`poisoning by
        albumin, human serum`: Albumin > human serum)."""
        before = [word for word in words(previous) if word not in STOPWORDS]
        after = [word for word in words(fragment) if word not in STOPWORDS]
        if not before or not after:
            return False
        steps = {
            form
            for size in range(1, min(4, len(after)) + 1)
            for form in title_forms(" ".join(after[:size]))
        }
        for size in range(1, min(3, len(before)) + 1):
            for form in title_forms(" ".join(before[:size])):
                if steps & self.descendants.get(form, frozenset()):
                    return True
            for form in title_forms(" ".join(before[-size:])):
                if steps & self.table_descendants.get(form, frozenset()):
                    return True
        return False

    def joins(self, sequence: Sequence[str], at: int) -> bool:
        """Whether a joined CMS term covers the words on both sides of position `at`, which lies
        between `sequence[at - 1]` and `sequence[at]` (a comma), or is an "and" or "or"."""
        for begin in range(at):
            for term in self.joined.get(sequence[begin], ()):
                end = begin + len(term)
                if end > at and tuple(sequence[begin:end]) == term:
                    return True
        return False

    def combines(self, left: str, right: str) -> bool:
        """Whether the Index lists one under the other's "with" (`Nausea > with vomiting`)."""
        lefts, rights = title_forms(left), title_forms(right)
        return any(
            (a, b) in self.combinations or (b, a) in self.combinations
            for a in lefts
            for b in rights
        )


def _root(graph: Graph, term_id: int) -> int:
    parent = graph.term_parent(term_id)
    while parent is not None:
        term_id = parent.id
        parent = graph.term_parent(term_id)
    return term_id


def _below(paths: Iterable[tuple[str, str]]) -> dict[str, frozenset[str]]:
    found: dict[str, set[str]] = {}
    for main, title in paths:
        forms = title_forms(title)
        for head in (main, *main.split(",")):
            for form in title_forms(head):
                found.setdefault(form, set()).update(forms)
    return {form: frozenset(below) for form, below in found.items()}


def _only_sites(form: str, sites: frozenset[str]) -> bool:
    found = words(form)
    return bool(found) and all(word in sites or word in STOPWORDS for word in found)


def _has_joiner(text: str) -> bool:
    return bool(re.search(r"\s(?:and|or)\s", text, re.IGNORECASE))


def _sequence(text: str) -> tuple[str, ...]:
    return tuple(words(_UNSPECIFIC.sub(" ", _PARENS.sub(" ", text.casefold()))))


def _by_first_word(sequences: Iterable[tuple[str, ...]]) -> dict[str, tuple[tuple[str, ...], ...]]:
    found: dict[str, set[tuple[str, ...]]] = {}
    for sequence in sequences:
        if 2 < len(sequence) <= MAX_JOINED and _JOINERS & set(sequence[1:-1]):
            found.setdefault(sequence[0], set()).add(sequence)
    return {word: tuple(sorted(items)) for word, items in found.items()}
