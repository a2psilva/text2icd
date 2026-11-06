"""Clinical abbreviations: the lexicon in `resources/abbreviations.toml` and its use on phrases.

The lexicon is written from medical knowledge (the file header gives the rules). An entry has one
or more expansions; with more than one it is ambiguous, and words of the phrase (hints) choose
the meaning, or the first expansion is taken and the choice is flagged. Entries marked `cased`
match only when written in capitals, because their lower-case forms are English words.

Expansion runs on the cleaned phrase before case folding, so that `ALL` and `all` differ.
Abbreviation-like tokens the lexicon does not know are reported, for the LLM step and for
deciding which entries to write next.
"""

import re
import tomllib
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from text2icd.text.tracked import Tracked
from text2icd.trace import TraceStep, change, text_step

LEXICON = "abbreviations.toml"
SCHEMA = 1
_SEPARATORS = re.compile(r"[\s\-./']")
_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?![A-Za-z0-9])")
_ROMAN = re.compile(r"[IVXL]+[ab]?")
_NOT_ABBREVIATIONS = frozenset({"NOS", "NEC"})
_SHOUTING_LENGTH = 5  # an all-capitals phrase holds a word this long that is not a key

ChosenBy = Literal["only", "hint", "default"]


class LexiconError(ValueError):
    pass


def fold(key: str) -> str:
    """The form keys are compared in: `A-fib`, `AFib` and `a fib` are all `afib`."""
    return _SEPARATORS.sub("", key.casefold())


@dataclass(frozen=True, slots=True)
class Abbreviation:
    key: str
    expansions: tuple[str, ...]
    hints: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    cased: bool = False

    @property
    def ambiguous(self) -> bool:
        return len(self.expansions) > 1

    def choose(self, context: str) -> tuple[str, ChosenBy, str | None]:
        """The expansion for a phrase (case-folded): the first whose hint the phrase holds,
        else the first expansion."""
        if not self.ambiguous:
            return self.expansions[0], "only", None
        for expansion in self.expansions:
            for hint in self.hints.get(expansion, ()):
                if re.search(rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])", context):
                    return expansion, "hint", hint
        return self.expansions[0], "default", None


class Expansion(BaseModel):
    """One abbreviation expanded in a phrase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    written: str  # as in the phrase: "MS"
    key: str  # the lexicon entry
    expansion: str
    alternatives: tuple[str, ...] = ()  # the other meanings of an ambiguous entry
    chosen_by: ChosenBy = "only"
    hint: str | None = None
    start: int | None = None  # where `written` is in the original phrase (task 9.11)
    end: int | None = None

    @property
    def ambiguous(self) -> bool:
        return self.chosen_by != "only"

    def step(self) -> TraceStep:
        detail = change(self.written, self.expansion)
        others = ", ".join(self.alternatives)
        if self.chosen_by == "hint":
            return text_step("abbreviation", f'{detail} (hint "{self.hint}"; also: {others})')
        if self.chosen_by == "default":
            return text_step("ambiguous_abbreviation", f"{detail} (no hint; also: {others})")
        return text_step("abbreviation", detail)


class Lexicon:
    def __init__(self, entries: Iterable[Abbreviation]) -> None:
        self.entries: dict[str, Abbreviation] = {}
        for entry in entries:
            folded = fold(entry.key)
            if folded in self.entries:
                other = self.entries[folded].key
                raise LexiconError(f"{entry.key!r} and {other!r} are the same key")
            self.entries[folded] = entry
        patterns = sorted((_key_pattern(e.key) for e in self.entries.values()), key=len)
        alternation = "|".join(reversed(patterns))
        self._pattern = re.compile(
            rf"(?<![A-Za-z0-9])(?P<key>{alternation})(?P<plural>s)?"
            rf"(?![A-Za-z0-9]|-[0-9])(?P<space>(?=\s))?",
            re.IGNORECASE,
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "Lexicon":
        raw = (
            path.read_bytes()
            if path is not None
            else resources.files("text2icd.resources").joinpath(LEXICON).read_bytes()
        )
        return cls.from_toml(raw.decode("utf-8"))

    @classmethod
    def from_toml(cls, text: str) -> "Lexicon":
        data = tomllib.loads(text)
        if data.get("schema") != SCHEMA:
            raise LexiconError(f"unsupported lexicon schema {data.get('schema')!r}")
        return cls(_entry(key, value) for key, value in data.get("abbreviations", {}).items())

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and fold(key) in self.entries

    def get(self, key: str) -> Abbreviation | None:
        return self.entries.get(fold(key))

    def expand(self, text: str, *, shouting: bool | None = None) -> tuple[str, list[Expansion]]:
        """The text with every abbreviation replaced by its expansion, and the expansions.
        `shouting` says whether the phrase is written all in capitals (computed from `text`
        when not given)."""
        expanded, found = self.expand_tracked(Tracked.of(text), shouting=shouting)
        return expanded.text, found

    def expand_tracked(
        self, text: Tracked, *, shouting: bool | None = None
    ) -> tuple[Tracked, list[Expansion]]:
        """`expand`, keeping track of where each character came from; each expansion records
        the span of the original phrase it was written at."""
        if shouting is None:
            shouting = self.shouting(text.text)
        context = text.text.casefold()
        found: list[Expansion] = []

        def replace(match: re.Match[str]) -> str:
            written = match["key"]
            entry = self.entries[fold(written)]
            capitals = written == written.upper()
            if match["plural"] and not capitals:
                return match[0]
            if entry.cased and (shouting or not capitals):
                return match[0]
            if len(fold(written)) == 1 and match["space"] is None:
                return match[0]
            expansion, chosen_by, hint = entry.choose(context)
            others = tuple(e for e in entry.expansions if e != expansion)
            start, end = text.origin(*match.span())
            found.append(
                Expansion(
                    written=match[0],
                    key=entry.key,
                    expansion=expansion,
                    alternatives=others,
                    chosen_by=chosen_by,
                    hint=hint,
                    start=start,
                    end=end,
                )
            )
            return expansion

        return text.sub(self._pattern, replace), found

    def unknown(
        self,
        text: str,
        *,
        shouting: bool | None = None,
        vocabulary: Container[str] | None = None,
    ) -> tuple[str, ...]:
        """Tokens of a phrase before expansion that look like abbreviations (two or more
        capitals) and have no entry. Tokens in `vocabulary` (case-folded words of the CMS files,
        where the Index lists AIDS, GERD or STEMI as written) are known too."""
        if self.shouting(text) if shouting is None else shouting:
            return ()
        tokens = []
        for match in _TOKEN.finditer(text):
            token = match[0]
            if sum(char.isupper() for char in token) < 2 or len(token) > 8:
                continue
            if token in _NOT_ABBREVIATIONS or _ROMAN.fullmatch(token) or self._knows(token):
                continue
            if vocabulary is not None and token.casefold() in vocabulary:
                continue
            if token not in tokens:
                tokens.append(token)
        return tuple(tokens)

    def _knows(self, token: str) -> bool:
        return token in self or (token.endswith("s") and token[:-1] in self)

    def shouting(self, text: str) -> bool:
        """Whether the phrase is written all in capitals, where `AS` may be the word "as"."""
        if any(char.islower() for char in text):
            return False
        return any(
            len(word) >= _SHOUTING_LENGTH and word not in self
            for word in re.findall(r"[A-Z]+", text)
        )


def _entry(key: str, value: Any) -> Abbreviation:
    if isinstance(value, str):
        value = {"expansions": [value]}
    if not isinstance(value, dict):
        raise LexiconError(f"{key}: an entry is a string or a table")
    unknown = set(value) - {"expansions", "hints", "cased"}
    if unknown:
        raise LexiconError(f"{key}: unknown fields {sorted(unknown)}")
    expansions = tuple(value.get("expansions", ()))
    if not expansions or not all(isinstance(e, str) and e.strip() for e in expansions):
        raise LexiconError(f"{key}: needs at least one expansion")
    if len(set(expansions)) != len(expansions):
        raise LexiconError(f"{key}: repeated expansion")
    hints = {name: tuple(words) for name, words in value.get("hints", {}).items()}
    if extra := set(hints) - set(expansions):
        raise LexiconError(f"{key}: hints for unknown expansions {sorted(extra)}")
    if hints and len(expansions) == 1:
        raise LexiconError(f"{key}: hints need more than one expansion")
    for words in hints.values():
        if any(word != word.casefold() or not word.strip() for word in words):
            raise LexiconError(f"{key}: hints are written in lower case")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .\-/']*", key):
        raise LexiconError(f"{key!r}: a key is letters and digits, and spaces, dots, hyphens or /")
    return Abbreviation(key, expansions, hints, bool(value.get("cased", False)))


def _key_pattern(key: str) -> str:
    """The pattern of a key: separators inside it are optional, and match any separator."""
    parts = re.split(r"[\s\-./']+", key)
    return r"[\s\-./']?\s?".join(re.escape(part) for part in parts if part)


@lru_cache(maxsize=1)
def default_lexicon() -> Lexicon:
    """The lexicon shipped with text2icd."""
    return Lexicon.load()
