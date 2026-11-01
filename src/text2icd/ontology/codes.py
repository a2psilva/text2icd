"""ICD-10-CM code strings: formatting, validation, 7th characters and code references.

A code starts with a three-character category: a letter followed by two letters or digits, at least
one of them a digit (`E11`, `C4A`, and since FY2026 `QA0`). It is optionally followed by a dot and
one to four letters or digits (`E11.22`, `T51.3X1A`, `QA0.0101`).
MIMIC-IV and the CMS order files store codes without the dot (`E1122`).
"""

import re
from dataclasses import dataclass

# A category has a digit in its second or third position: E11, C4A, O9A, QA0. Words such as HIV
# or SCFE are not codes.
_CATEGORY = r"[A-Z](?:[0-9][0-9A-Z]|[A-Z][0-9])"
_UNDOTTED = re.compile(rf"^{_CATEGORY}[0-9A-Z]{{0,4}}$")
_DOTTED = re.compile(rf"^{_CATEGORY}(?:\.[0-9A-Z]{{1,4}})?$")
_REFERENCE = re.compile(rf"^({_CATEGORY}(?:\.?[0-9A-Z]{{1,4}})?)(?:\.?-)?$")

PLACEHOLDER = "X"
CODE_LENGTH_BEFORE_SEVENTH = 6


def undot(code: str) -> str:
    """`E11.22` -> `E1122`. Validates the result."""
    result = code.strip().upper().replace(".", "")
    if not _UNDOTTED.fullmatch(result):
        raise ValueError(f"not an ICD-10-CM code: {code!r}")
    return result


def dot(code: str) -> str:
    """`E1122` or `E11.22` -> `E11.22`; a three-character category stays as it is."""
    bare = undot(code)
    return bare if len(bare) == 3 else f"{bare[:3]}.{bare[3:]}"


def is_code(code: str) -> bool:
    """True for a well-formed code, dotted or not. Says nothing about whether it exists."""
    value = code.strip().upper()
    return bool(_DOTTED.fullmatch(value) or _UNDOTTED.fullmatch(value))


def category(code: str) -> str:
    """The three-character category of a code: `T51.3X1A` -> `T51`."""
    return undot(code)[:3]


def with_seventh_character(code: str, character: str) -> str:
    """Append a 7th character, padding with the `X` placeholder first (Guidelines I.A.4, I.A.5).

    `T51.3X1` + `A` -> `T51.3X1A`; `S72.001` + `A` -> `S72.001A`; `T07` + `A` -> `T07.XXXA`.
    """
    if not re.fullmatch(r"[0-9A-Z]", character):
        raise ValueError(f"a 7th character is one letter or digit, got {character!r}")
    bare = undot(code)
    if len(bare) > CODE_LENGTH_BEFORE_SEVENTH:
        raise ValueError(f"{code!r} already has seven characters")
    return dot(bare.ljust(CODE_LENGTH_BEFORE_SEVENTH, PLACEHOLDER) + character)


def is_partial(code: str) -> bool:
    """Alphabetic Index codes ending in `-` need more characters (Guideline I.B.1)."""
    return code.strip().endswith("-")


def strip_partial(code: str) -> str:
    """`C79.8-` -> `C79.8`; `E10.-` -> `E10`."""
    return dot(code.strip().rstrip("-").rstrip("."))


@dataclass(frozen=True, slots=True)
class CodeReference:
    """A code or range as written in instructional notes: `N18.1-N18.6`, `J09-J18`, `E10.-`.

    Both ends are prefixes: `N18.1-N18.6` contains `N18.31`, and a single code such as `K70.3`
    contains its subdivisions (`K70.30`, `K70.31`), as the notes intend.
    """

    start: str
    end: str

    def __contains__(self, code: object) -> bool:
        if not isinstance(code, str) or not is_code(code):
            return False
        bare = undot(code)
        return self.start <= bare[: len(self.start)] and bare[: len(self.end)] <= self.end

    def __str__(self) -> str:
        if self.start == self.end:
            return dot(self.start)
        return f"{dot(self.start)}-{dot(self.end)}"


def parse_reference(text: str) -> CodeReference:
    """Parse one code reference: a code, a partial code (`C79.8-`, `E10.-`) or a range (`A-B`)."""
    value = text.strip().upper().replace("\N{EN DASH}", "-")
    first, sep, last = value.partition("-")
    if sep and last and not last.startswith("-"):
        start, end = _reference_prefix(first, text), _reference_prefix(last, text)
        if start > end:
            raise ValueError(f"range end comes before its start: {text!r}")
        return CodeReference(start, end)
    prefix = _reference_prefix(value, text)
    return CodeReference(prefix, prefix)


def _reference_prefix(part: str, original: str) -> str:
    match = _REFERENCE.fullmatch(part.strip())
    if not match:
        raise ValueError(f"not a code reference: {original!r}")
    return match.group(1).replace(".", "")
