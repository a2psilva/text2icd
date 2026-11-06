"""Text that remembers where each of its characters came from (task 9.11).

Normalisation rewrites a phrase many times (Unicode, spacing, shorthand, abbreviations, case,
numerals), and every later span (conditions, modifiers, cues) points into the result. A
`Tracked` text carries, for each character, the span of the original phrase it comes from, so
a span of the normalised text maps back to the words the user wrote: `origin(start, end)`.

A rewrite maps what it keeps to where it was, and what it inserts to the span it replaced: the
common prefix and suffix of the old and new text map one to one, and the rest to the part of
the match they replace. `"w/"` to `"with "` maps `w` to `w` and `ith ` to `/`;
`"DM"` to `"diabetes mellitus"` maps every character to `DM`.
"""

import re
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

Replace = str | Callable[[re.Match[str]], str]


@dataclass(frozen=True, slots=True)
class Tracked:
    text: str
    starts: tuple[int, ...]  # per character: where its source starts in the original
    ends: tuple[int, ...]  # and where it ends

    @classmethod
    def of(cls, text: str) -> "Tracked":
        """The original text, each character its own source."""
        return cls(text, tuple(range(len(text))), tuple(range(1, len(text) + 1)))

    def origin(self, start: int, end: int) -> tuple[int, int]:
        """The span of the original text that `text[start:end]` comes from."""
        return source_span(self.starts, self.ends, start, end)

    def sub(self, pattern: re.Pattern[str], replace: Replace) -> "Tracked":
        """`pattern.sub(replace, text)`, tracked. `replace` is called once per match, in order,
        exactly as `re.sub` calls it, so callbacks that record what they do work unchanged."""
        text = self.text
        pieces: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        last = 0
        for match in pattern.finditer(text):
            new = replace(match) if callable(replace) else match.expand(replace)
            old = match[0]
            if new == old:
                continue  # copied with the untouched text around it
            begin, finish = match.span()
            pieces.append(text[last:begin])
            starts += self.starts[last:begin]
            ends += self.ends[last:begin]
            self._splice(pieces, starts, ends, begin, finish, old, new)
            last = finish
        if not pieces:
            return self
        pieces.append(text[last:])
        starts += self.starts[last:]
        ends += self.ends[last:]
        return Tracked("".join(pieces), tuple(starts), tuple(ends))

    def map_chars(self, convert: Callable[[str], str]) -> "Tracked":
        """`convert` applied to each character, for conversions that work one character at a
        time (`str.translate`, `str.casefold`). Each output character comes from the character it
        was made from. If converting the whole text gives another result, every character maps to
        the whole span, which stays correct, if coarse."""
        whole = convert(self.text)
        if whole == self.text:
            return self
        pieces: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for i, char in enumerate(self.text):
            new = convert(char)
            pieces.append(new)
            starts += [self.starts[i]] * len(new)
            ends += [self.ends[i]] * len(new)
        if "".join(pieces) != whole:
            return self._coarse(whole)
        return Tracked(whole, tuple(starts), tuple(ends))

    def nfkc(self) -> "Tracked":
        """Unicode NFKC, tracked by cluster: a character with the combining marks after it."""
        whole = unicodedata.normalize("NFKC", self.text)
        if whole == self.text:
            return self
        pieces: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for begin, finish in _clusters(self.text):
            new = unicodedata.normalize("NFKC", self.text[begin:finish])
            source = self.origin(begin, finish)
            pieces.append(new)
            starts += [source[0]] * len(new)
            ends += [source[1]] * len(new)
        if "".join(pieces) != whole:
            return self._coarse(whole)
        return Tracked(whole, tuple(starts), tuple(ends))

    def _splice(
        self,
        pieces: list[str],
        starts: list[int],
        ends: list[int],
        begin: int,
        finish: int,
        old: str,
        new: str,
    ) -> None:
        """Append `new`, which replaces `text[begin:finish]` (`old`)."""
        prefix = _common_prefix(old, new)
        suffix = _common_prefix(old[prefix:][::-1], new[prefix:][::-1])
        pieces.append(new)
        starts += self.starts[begin : begin + prefix]
        ends += self.ends[begin : begin + prefix]
        middle = len(new) - prefix - suffix
        source = self.origin(begin + prefix, finish - suffix)
        starts += [source[0]] * middle
        ends += [source[1]] * middle
        starts += self.starts[finish - suffix : finish]
        ends += self.ends[finish - suffix : finish]

    def _coarse(self, text: str) -> "Tracked":
        source = self.origin(0, len(self.text))
        return Tracked(text, (source[0],) * len(text), (source[1],) * len(text))


def source_span(
    starts: Sequence[int], ends: Sequence[int], start: int, end: int
) -> tuple[int, int]:
    """The span of the original text that characters `start` to `end` of a tracked text come
    from (`starts` and `ends` per character). An empty span maps to an empty span where it
    sits, and a span past the end is cut at the end."""
    end = min(end, len(starts))
    if start < end:
        return min(starts[start:end]), max(ends[start:end])
    if start < len(starts):
        return starts[start], starts[start]
    return (ends[-1], ends[-1]) if ends else (0, 0)


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _clusters(text: str) -> Iterator[tuple[int, int]]:
    """Spans of a base character and the combining marks after it."""
    begin = 0
    for i in range(1, len(text) + 1):
        if i == len(text) or not unicodedata.combining(text[i]):
            yield begin, i
            begin = i
