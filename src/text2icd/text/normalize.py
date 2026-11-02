"""Normalise a diagnosis phrase: the first step of text understanding.

Steps, in order; each rewrite is recorded in the trace:

1. `clean`: Unicode NFKC, dashes and quotes, invisible characters, list markers, spacing around
   punctuation, whitespace, and punctuation at the ends.
2. `shorthand`: clinical shorthand written with slashes or short forms (`w/`, `w/o`, `s/p`, `c/b`,
   `2/2`, `h/o`, `r/o`, `hx`, ...) replaced by words.
3. Case folding.
4. `numeral`: Roman numerals of stages (`stage IIIb` to `stage 3b`, the form the CMS files use),
   CKD stages written without the word (`CKD 3` to `CKD stage 3`), and diabetes types
   (`type II diabetes`, `diabetes mellitus 2` to `type 2`). Other Roman numerals stay: the Index
   writes `class I`, `factor VIII` and `type I` glycogen storage disease that way.

The original string is kept unchanged alongside the normalised text.
"""

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from text2icd.trace import TraceStep, change, text_step

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2043\u2212\ufe58\ufe63\uff0d"
_SINGLE_QUOTES = "\u2018\u2019\u201a\u201b\u2032\u00b4`"
_DOUBLE_QUOTES = "\u201c\u201d\u201e\u201f\u2033"
_INVISIBLE = "\u00ad\u200b\u200c\u200d\u2060\ufeff"
_TRANSLATION = {
    **dict.fromkeys(map(ord, _DASHES), "-"),
    **dict.fromkeys(map(ord, _SINGLE_QUOTES), "'"),
    **dict.fromkeys(map(ord, _DOUBLE_QUOTES), '"'),
    **dict.fromkeys(map(ord, _INVISIBLE), None),
}
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022\u00b7#>]+|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+", re.I)
_SPACE_BEFORE = re.compile(r"\s+([,;:)])")
_SPACE_AFTER_PAREN = re.compile(r"\(\s+")
_SEPARATOR = re.compile(r"([,;])(?=\S)")
_SPACES = re.compile(r"\s+")
_EDGES = re.compile(r"^[\s.,;:]+|[\s.,;:]+$")

Rule = tuple[re.Pattern[str], Callable[[re.Match[str]], str]]


def _word(pattern: str) -> str:
    """A shorthand that stands alone: not inside a word or a longer slash form."""
    return rf"(?<![\w/])(?:{pattern})(?![\w/])"


def _lead(pattern: str) -> str:
    """A shorthand ending in a slash, which may touch the next word: `w/sepsis`."""
    return rf"(?<![\w/])(?:{pattern})"


def _fixed(replacement: str) -> Callable[[re.Match[str]], str]:
    return lambda _: replacement


_SHORTHAND_TABLE: tuple[tuple[str, str], ...] = (
    (_lead(r"assoc\.?\s*w/") + "|" + _word(r"assoc\.?\s+with|a/w"), "associated with "),
    (_word(r"assoc\.?"), "associated"),
    (_word(r"w/o(?:ut)?"), "without"),
    (_lead(r"w/"), "with "),
    (_word(r"c/b"), "complicated by"),
    (_word(r"s/p|s\.p\.|status[\s-]+post"), "status post"),
    (_word(r"f/h|fhx|fam(?:ily)?\.?\s+hx"), "family history"),
    (_word(r"h/o|h/x|hx\.?\s+of"), "history of"),
    (_word(r"hx"), "history"),
    (_word(r"r/o"), "rule out"),
    (_word(r"2/2(?:\s+to)?|d/t"), "due to"),
    (r"(?<![\w/])(?:2['\u00b0]|2nd(?:ary)?|sec\.?)\s*to(?![\w/])", "secondary to"),
    (_word(r"b/l|bilat\.?"), "bilateral"),
    (_word(r"c/w"), "consistent with"),
    (_word(r"r/t"), "related to"),
    (_word(r"d/o"), "disorder"),
    (_word(r"v\.?s\.|v/s"), "vs"),
    (r"(?<=\s)&(?=\s)", "and"),
    (_word(r"neg\.?"), "negative"),
)
SHORTHAND: tuple[Rule, ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), _fixed(replacement))
    for pattern, replacement in _SHORTHAND_TABLE
)

_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
_ROMAN_STAGE = r"iv|v|i{1,3}"
_NOT_TIME = r"(?!\s*(?:years?|yrs?|months?|mos?|weeks?|wks?|days?)\b)"


def _stage(match: re.Match[str]) -> str:
    return f"stage {_ROMAN[match[1]]}{match[2] or ''}"


def _ckd_stage(match: re.Match[str]) -> str:
    stage = _ROMAN.get(match[2], match[2])
    return f"{match[1]} stage {stage}{match[3] or ''}"


def _diabetes_type(match: re.Match[str]) -> str:
    return f"{match[1]} type {_ROMAN.get(match[2], match[2])}"


def _type_before_diabetes(match: re.Match[str]) -> str:
    return f"type {_ROMAN[match[1]]}"


NUMERALS: tuple[Rule, ...] = (
    (re.compile(rf"\bstage\s+({_ROMAN_STAGE})([ab])?\b"), _stage),
    (
        re.compile(
            rf"\b(chronic kidney disease|ckd)[\s,-]*(?:stage\s*)?({_ROMAN_STAGE}|[1-5])([ab])?\b"
            + _NOT_TIME
        ),
        _ckd_stage,
    ),
    (
        re.compile(rf"\b(diabetes(?: mellitus)?|dm)[\s,-]*(?:type[\s-]*)?(ii|i|2|1)\b{_NOT_TIME}"),
        _diabetes_type,
    ),
    (re.compile(r"\btype[\s-]*(ii|i)\b(?=[\s,-]*(?:diabetes|dm)\b)"), _type_before_diabetes),
)


@dataclass(frozen=True, slots=True)
class Normalized:
    original: str
    text: str
    trace: tuple[TraceStep, ...]


def normalize(phrase: str) -> Normalized:
    trace: list[TraceStep] = []
    text = clean(phrase)
    if text != phrase:
        trace.append(text_step("clean", change(phrase, text)))
    text = clean(rewrite(text, SHORTHAND, "shorthand", trace))
    text = rewrite(text.casefold(), NUMERALS, "numeral", trace)
    return Normalized(original=phrase, text=text, trace=tuple(trace))


def clean(text: str) -> str:
    """NFKC, one kind of dash and quote, no list marker, and single spaces after `,` and `;`."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATION)
    text = "".join(" " if unicodedata.category(char) == "Cc" else char for char in text)
    text = _LIST_MARKER.sub("", text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER_PAREN.sub("(", text)
    text = _SEPARATOR.sub(_separate, text)
    return _EDGES.sub("", _SPACES.sub(" ", text))


def _separate(match: re.Match[str]) -> str:
    text, at = match.string, match.start()
    if match[1] == "," and at > 0 and text[at - 1].isdigit() and text[at + 1].isdigit():
        return ","  # 1,000
    return match[1] + " "


def rewrite(text: str, rules: Sequence[Rule], action: str, trace: list[TraceStep]) -> str:
    """Apply each rule in turn, recording every match that changes the text."""
    for pattern, replace in rules:

        def record(match: re.Match[str], replace: Callable[[re.Match[str]], str] = replace) -> str:
            result = replace(match)
            if result.strip() != match[0]:
                trace.append(text_step(action, change(match[0], result.strip())))
            return result

        text = pattern.sub(record, text)
    return text
