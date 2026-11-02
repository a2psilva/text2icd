"""What text understanding needs from the knowledge graph: the words of the CMS files.

`vocabulary(graph)` is the set of case-folded words of the Alphabetic Index (titles and
nonessential modifiers of the four books) and the Tabular List (descriptions and inclusion terms),
without diacritics. A token outside it is a misspelling, a truncation or an abbreviation.
"""

import re
import unicodedata

from text2icd.graph.model import Graph

_WORD = re.compile(r"[a-z0-9]+")


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
