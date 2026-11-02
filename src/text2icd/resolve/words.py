"""Words as the Index walk compares them: a phrase and an Index title meet word by word.

Both sides are reduced to the same forms (`WordForms.form`):

- case folded and without diacritics (`text.terms.words`): `Schönlein's` gives `schonlein`;
- singular, by suffix rules that only have to treat both sides alike (`fractures`, `injuries`,
  `abscesses`, `vertebrae`, `stenoses`), and a short list of irregular plurals (`calculi`);
- the noun of an anatomical adjective (`resources/word_forms.toml`): `renal` and `kidney` are one
  word, so "renal failure" meets `Failure, renal` and "kidney failure" meets it too.

Function words and connectors (`of`, `with`, `due to`) carry no content of their own: the walk
reads connector subterms through the relations of the parsed phrase. `without`, `not` and `no`
are kept, because they decide between the `with` and `without` subterms. `NEC` and `NOS` mark
residual subterms and are never written in a phrase.

A phrase word that no CMS text uses is taken for a typo when an Index word is within one edit of
it (two for words of nine letters or more; swapping two neighbouring letters is one edit), and is
replaced by it, the more frequent Index word winning a tie (`Speller`). Shorter words than five
letters are never corrected.
"""

import tomllib
from collections import Counter
from collections.abc import Container, Iterable, Mapping
from functools import lru_cache
from importlib import resources
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import OSA

from text2icd.text.terms import words

WORD_FORMS = "word_forms.toml"
SCHEMA = 1
STOPWORDS = frozenset(
    {"a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "and", "or", "with", "due"}
    | {"as", "from", "s", "nec", "nos", "is"}
)
MIN_CORRECTED = 5  # letters
LONG_WORD = 9  # letters: two edits allowed from here on
_SINGULAR_AS = frozenset({"pancreas", "atlas", "alias"})  # other -as words are plurals: hernias


def singular(word: str) -> str:
    """`fractures` -> `fracture`, `injuries` -> `injury`, `abscesses` -> `abscess`, `hernias` ->
    `hernia`; words of three letters or fewer, and words ending like a singular (`-ss`, `-us`,
    `-is`, `pancreas`), unchanged."""
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("aches"):  # headaches
        return word[:-1]
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if len(word) >= 7 and word.endswith(("oses", "tases")):  # stenoses, metastases
        return word[:-2] + "is"
    if word.endswith("ae"):  # vertebrae
        return word[:-1]
    if word.endswith(("ss", "us", "is")) or word in _SINGULAR_AS:
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


class WordFormsError(ValueError):
    """The word-forms file is malformed."""


class WordForms:
    """Maps words to the forms the walk compares: singular, and nouns for anatomical adjectives."""

    def __init__(self, adjectives: Mapping[str, str], plurals: Mapping[str, str]) -> None:
        self.adjectives = dict(adjectives)
        self.plurals = dict(plurals)
        self._forms: dict[str, str] = {}

    @classmethod
    def load(cls, path: Path | None = None) -> "WordForms":
        raw = (
            path.read_bytes()
            if path is not None
            else resources.files("text2icd.resources").joinpath(WORD_FORMS).read_bytes()
        )
        data = tomllib.loads(raw.decode())
        if data.get("schema") != SCHEMA:
            raise WordFormsError(f"{WORD_FORMS}: schema must be {SCHEMA}")
        tables = {}
        for name in ("adjectives", "plurals"):
            table = data.get(name, {})
            if not isinstance(table, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in table.items()
            ):
                raise WordFormsError(f"{WORD_FORMS}: [{name}] must map words to words")
            tables[name] = {k.casefold(): v.casefold() for k, v in table.items()}
        return cls(tables["adjectives"], tables["plurals"])

    def form(self, word: str) -> str:
        """The form of one case-folded word: `Kidneys` and `renal` are both `kidney`."""
        found = self._forms.get(word)
        if found is None:
            base = self.plurals.get(word) or singular(word)
            found = self.adjectives.get(base, base)
            self._forms[word] = found
        return found

    def content(self, text: str) -> list[str]:
        """The forms of the words of `text` that carry content, in order."""
        return [self.form(word) for word in words(text) if word not in STOPWORDS]


@lru_cache(maxsize=1)
def default_word_forms() -> WordForms:
    return WordForms.load()


class Speller:
    """Corrects words that no CMS text uses to the nearest Index word."""

    def __init__(self, counts: Mapping[str, int], known: Container[str] = ()) -> None:
        self._choices = sorted(counts, key=lambda word: (-counts[word], word))
        self._index = frozenset(counts)
        self._known = known
        self._cache: dict[str, str | None] = {}

    @classmethod
    def from_texts(cls, texts: Iterable[str], known: Container[str] = ()) -> "Speller":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(word for word in words(text) if word.isalpha())
        return cls(counts, known)

    def known(self, word: str) -> bool:
        return word in self._index or word in self._known

    def correct(self, word: str) -> str | None:
        """The Index word `word` was meant to be, or None if it is known, too short, or has no
        single nearest Index word."""
        if len(word) < MIN_CORRECTED or not word.isalpha() or self.known(word):
            return None
        if word not in self._cache:
            limit = 2 if len(word) >= LONG_WORD else 1
            best = process.extractOne(word, self._choices, scorer=OSA.distance, score_cutoff=limit)
            self._cache[word] = None if best is None else best[0]
        return self._cache[word]
