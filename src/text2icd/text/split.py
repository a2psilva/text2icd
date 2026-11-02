"""Split a normalised phrase into conditions and the relations between them.

The coder separates the reportable conditions of a statement (Guideline I.B.4-I.B.6) and keeps
how they are linked, because the Index's "with", "due to" and "in" subterms use the linked
condition (I.A.15). The phrase is cut in this order:

1. `;` separates list items.
2. `,` separates fragments of an item. A fragment that names no condition, that describes
   treatment (`requiring`, `on`), or that continues the Index path of the fragment before
   (`pneumonia, aspiration`: Pneumonia > aspiration) refines the one before it
   (`pneumonia, right lower lobe`, `fracture, initial encounter`, `UTI, resolved`). A
   fragment that starts with a connector, or with `in` as the Index's "in" subterms do, links
   to the condition before it (`pneumonia, with sepsis`, `ulcer, in Behcet's disease`).
3. Linking connectors join conditions into one diagnosis: `with`, `without`, `due to`,
   `secondary to`, `caused by`, `complicated by`, `in the setting of`, `on` (`AKI on CKD`).
   A part after a connector that names no condition qualifies the one before it
   (`UTI due to E. coli`).
4. `and`, `or`, `vs` and `/` list conditions side by side, unless the words form a CMS term
   (`hand, foot and mouth disease`) or the Index lists one under the other's "with"
   (`nausea and vomiting`: Nausea > with vomiting). A part that names no condition joins the
   next one (`acute and chronic respiratory failure`).

Modifier and cue spans are never cut (`consistent with`, `with complications`,
`acute on chronic`). Whether a part names a condition comes from the graph (`terms.Terms`):
some of its words, apart from modifier and cue words, form the title of a condition main term.
Without a graph, a part names a condition when it has any such words besides stopwords. Every
cut, refinement and qualifier is a trace step.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from text2icd.text.model import Cue, Modifier, RelationKind
from text2icd.text.terms import STOPWORDS, Terms, words
from text2icd.trace import TraceStep, text_step

LINKS: dict[str, RelationKind] = {
    "with": "with",
    "associated with": "with",
    "in association with": "with",
    "accompanied by": "with",
    "without": "without",
    "due to": "due_to",
    "secondary to": "due_to",
    "caused by": "due_to",
    "because of": "due_to",
    "induced by": "due_to",
    "related to": "due_to",
    "attributed to": "due_to",
    "attributable to": "due_to",
    "as a result of": "due_to",
    "resulting from": "due_to",
    "complicated by": "complicated_by",
    "in the setting of": "in",
    "in setting of": "in",
    "in the context of": "in",
    "in context of": "in",
    "on": "on",
}
LISTS: dict[str, RelationKind] = {
    "and/or": "and",
    "and": "and",
    "or": "or",
    "vs": "versus",
    "versus": "versus",
    "/": "and",
}
_TREATMENT = re.compile(r"(?:requiring|required|requires|needing|treated with|on|receiving)\b")


def _alternation(keys: Sequence[str]) -> str:
    ordered = sorted(keys, key=len, reverse=True)
    return "|".join(re.escape(key).replace(r"\ ", r"\s+") for key in ordered)


_LINK = re.compile(rf"(?<![a-z0-9])(?:{_alternation(list(LINKS))})(?![a-z0-9])")
_LIST = re.compile(
    rf"(?<![a-z0-9])(?:{_alternation([key for key in LISTS if key != '/'])})(?![a-z0-9])"
    r"|(?<=[a-z])/(?=[a-z])"
)
_WORD = re.compile(r"[a-z0-9]+")
_IN = re.compile(r"in(?=\s)")
_EDGE = " ,;:-"


def connector_kind(text: str) -> RelationKind:
    key = " ".join(text.split())
    if key == "in":
        return "in"
    return LINKS.get(key) or LISTS[key]


@dataclass
class Piece:
    """A part of the phrase that becomes a condition, or merges into one."""

    start: int
    end: int
    condition: bool = False


@dataclass
class Link:
    kind: RelationKind
    head: Piece
    linked: Piece
    text: str
    start: int  # the connector's span
    end: int


@dataclass
class Fragment:
    start: int
    end: int
    pieces: list[Piece] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    leading: re.Match[str] | None = None  # a linking connector the fragment starts with

    @property
    def names_condition(self) -> bool:
        return any(piece.condition for piece in self.pieces)


class Splitter:
    def __init__(
        self,
        text: str,
        modifiers: Sequence[Modifier] = (),
        cues: Sequence[Cue] = (),
        terms: Terms | None = None,
    ) -> None:
        self.text = text
        self.terms = terms
        self.protected = [(m.start, m.end) for m in modifiers] + [(c.start, c.end) for c in cues]
        self.status_starts = [
            c.start for c in cues if c.family == "status" and c.position == "before"
        ]
        self.trace: list[TraceStep] = []

    def split(self) -> list[Fragment]:
        fragments: list[Fragment] = []
        for start, end in self._cuts(
            0, len(self.text), [m.start() for m in re.finditer(";", self.text)]
        ):
            fragments += self._item(start, end)
        return fragments

    # List items: fragments separated by commas

    def _item(self, start: int, end: int) -> list[Fragment]:
        kept: list[Fragment] = []
        for begin, finish in self._commas(start, end):
            fragment = self._fragment(begin, finish)
            if not fragment.pieces:
                if kept and fragment.leading:
                    kept[-1].pieces[-1].end = fragment.end  # a dangling connector
                    kept[-1].end = fragment.end
                continue
            if kept and fragment.leading:
                self._attach(kept[-1], fragment)
            elif kept and (
                not fragment.names_condition
                or _TREATMENT.match(self.text, fragment.start)
                or self._continues(kept[-1], fragment)
            ):
                last = kept[-1].pieces[-1]
                self._step("refinement", f'"{self._show(fragment)}" refines "{self._show(last)}"')
                last.end = kept[-1].end = fragment.end
            else:
                kept.append(fragment)
        while len(kept) > 1 and not kept[0].names_condition:
            first, second = kept[0], kept[1]
            self._step("refinement", f'"{self._show(first)}" refines "{self._show(second)}"')
            second.pieces[0].start = second.start = first.start
            kept.pop(0)
        if kept and not any(fragment.names_condition for fragment in kept):
            for piece in (piece for fragment in kept for piece in fragment.pieces):
                piece.condition = True  # nothing names a condition: keep what there is
        for left, right in pairwise(kept):
            self._step("split", f'",": "{self._show(left)}" | "{self._show(right)}"')
        return kept

    def _attach(self, previous: Fragment, fragment: Fragment) -> None:
        """A fragment that starts with a connector links to the condition before it."""
        assert fragment.leading is not None
        connector = fragment.leading
        head = previous.pieces[-1]
        if fragment.names_condition:
            self._link_step(connector[0], head, fragment.pieces)
            for piece in fragment.pieces:
                previous.links.append(self._link(connector, head, piece))
            previous.pieces += fragment.pieces
            previous.links += fragment.links
        else:
            self._step("qualifier", f'"{self._show(fragment)}" qualifies "{self._show(head)}"')
            head.end = fragment.end
        previous.end = fragment.end

    def _continues(self, previous: Fragment, fragment: Fragment) -> bool:
        """Whether the fragment continues the Index path of the one before (`anemia, hemolytic`)."""
        if self.terms is None:
            return False
        before = self._free(Piece(previous.start, previous.end))
        return self.terms.continues(before, self._free(Piece(fragment.start, fragment.end)))

    def _commas(self, start: int, end: int) -> list[tuple[int, int]]:
        sequence, offsets = self._tokens(start, end)
        cuts = []
        for match in re.finditer(",", self.text[start:end]):
            at = start + match.start()
            if self._protected(at) or self._between_digits(at):
                continue
            if self._joined(sequence, offsets, at):
                continue
            cuts.append(at)
        return self._cuts(start, end, cuts)

    # A fragment: segments joined by linking connectors

    def _fragment(self, start: int, end: int) -> Fragment:
        fragment = Fragment(start, end)
        connectors = [
            m for m in _LINK.finditer(self.text, start, end) if not self._protected(m.start())
        ]
        bounds = [start, *(b for m in connectors for b in (m.start(), m.end())), end]
        segments = [self._segment(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2)]
        before: list[re.Match[str] | None] = [None, *connectors]
        if connectors and not segments[0][0]:
            fragment.leading = connectors[0]
            before[1] = None
            segments, before = segments[1:], before[1:]
        elif (leading := _IN.match(self.text, start)) and not self._protected(start):
            # "ulcer, vulva, in Behcet's disease": the Index's "in" subterms
            fragment.leading = leading
            segments[0] = self._segment(
                leading.end(), segments[0][0][-1].end if segments[0][0] else end
            )
        for (pieces, links), connector in zip(segments, before, strict=True):
            if not pieces:
                continue
            names = any(piece.condition for piece in pieces)
            if not fragment.pieces:
                fragment.pieces, fragment.links = pieces, links
            elif not names:
                head = fragment.pieces[-1]
                shown = self._text(pieces[0].start, pieces[-1].end)
                self._step("qualifier", f'"{shown}" qualifies "{self._show(head)}"')
                head.end = pieces[-1].end
            elif not fragment.names_condition:
                # What came before qualifies what follows ("left, with fracture").
                pieces[0].start = fragment.pieces[0].start
                fragment.pieces, fragment.links = pieces, links
            else:
                assert connector is not None
                head = fragment.pieces[-1]
                self._link_step(connector[0], head, pieces)
                fragment.links += [self._link(connector, head, piece) for piece in pieces]
                fragment.pieces += pieces
                fragment.links += links
        if fragment.pieces and not fragment.names_condition:
            fragment.pieces = [Piece(fragment.pieces[0].start, fragment.pieces[-1].end)]
            fragment.links = []
        return fragment

    def _segment(self, start: int, end: int) -> tuple[list[Piece], list[Link]]:
        """The conditions a segment lists side by side, and their `and`, `or`, `vs` links. A
        status cue inside the segment also starts a new condition, without a link: in "CAD s/p
        CABG" the bypass graft is a status and the coronary artery disease current."""
        start, end = self._trim(start, end)
        if start >= end:
            return [], []
        sequence, offsets = self._tokens(start, end)
        cuts: list[tuple[int, int, re.Match[str] | None]] = [
            (cue_start, cue_start, None)
            for cue_start in self.status_starts
            if start < cue_start < end and words(self.text[start:cue_start])
        ]
        for listed in _LIST.finditer(self.text, start, end):
            if not self._protected(listed.start()):
                cuts.append((listed.start(), listed.end(), listed))
        pieces: list[Piece] = []
        joins: list[re.Match[str] | None] = []
        cursor = start
        for at, after, match in sorted(cuts, key=lambda cut: cut[0]):
            if at < cursor:
                continue
            if match is not None:
                if self._joined(sequence, offsets, at):
                    continue
                left, right = Piece(*self._trim(cursor, at)), self._after(match, end)
                if self.terms is not None and self._combined(left, right):
                    shown = self._text(left.start, right.end)
                    self._step(
                        "kept_together", f'"{shown}": the Index lists one under the other\'s "with"'
                    )
                    continue
            pieces.append(Piece(*self._trim(cursor, at)))
            joins.append(match)
            cursor = after
        pieces.append(Piece(*self._trim(cursor, end)))
        for piece in pieces:
            piece.condition = self._names_condition(piece)
        # A conjunct that names no condition joins the next one ("acute and chronic ...").
        kept: list[Piece] = []
        kept_joins: list[re.Match[str] | None] = []
        carry: tuple[int, re.Match[str] | None] | None = None
        for index, piece in enumerate(pieces):
            join = joins[index - 1] if index else None
            if carry is not None:
                (piece.start, join), carry = carry, None
            if not piece.condition and index < len(pieces) - 1:
                carry = (piece.start, join)
            elif not piece.condition and kept:
                kept[-1].end = piece.end
            else:
                if kept:
                    kept_joins.append(join)
                kept.append(piece)
        if not kept:
            kept = [Piece(start, end)]
        links = []
        for (left, right), join in zip(pairwise(kept), kept_joins, strict=False):
            if join is None:
                self._step("split", f'status: "{self._show(left)}" | "{self._show(right)}"')
                continue
            self._step("split", f'"{join[0]}": "{self._show(left)}" | "{self._show(right)}"')
            links.append(self._link(join, left, right))
        return kept, links

    def _after(self, match: re.Match[str], end: int) -> Piece:
        """The words after a list connector, up to the next one."""
        following = _LIST.search(self.text, match.end(), end)
        return Piece(*self._trim(match.end(), following.start() if following else end))

    def _combined(self, left: Piece, right: Piece) -> bool:
        assert self.terms is not None
        before = words(self._free(left))
        after = words(self._free(right))
        for size in range(1, min(3, len(before)) + 1):
            for count in range(1, min(4, len(after)) + 1):
                if self.terms.combines(" ".join(before[-size:]), " ".join(after[:count])):
                    return True
        return False

    def _names_condition(self, piece: Piece) -> bool:
        if piece.start in self.status_starts:
            return True  # "status post CABG": a status is coded even when no condition is named
        free = self._free(piece)
        if self.terms is not None:
            return self.terms.is_condition(free)
        return any(word not in STOPWORDS for word in words(free))

    def _joined(self, sequence: Sequence[str], offsets: Sequence[tuple[int, int]], at: int) -> bool:
        """Whether a CMS term written with "and" or "or" covers the cut at `at`."""
        if self.terms is None:
            return False
        index = sum(1 for _, token_end in offsets if token_end <= at)
        if 0 < index < len(sequence) and self.terms.joins(sequence, index):
            shown = " ".join(sequence[max(0, index - 3) : index + 3])
            self._step("kept_together", f'"{shown}" is one CMS term')
            return True
        return False

    # Helpers

    def _link(self, connector: re.Match[str], head: Piece, linked: Piece) -> Link:
        return Link(
            connector_kind(connector[0]),
            head,
            linked,
            connector[0],
            connector.start(),
            connector.end(),
        )

    def _link_step(self, connector: str, head: Piece, pieces: Sequence[Piece]) -> None:
        shown = self._text(pieces[0].start, pieces[-1].end)
        self._step("split", f'"{connector}": "{self._show(head)}" -> "{shown}"')

    def _free(self, piece: Piece) -> str:
        """A piece's text without the words modifiers and cues account for."""
        chars = list(self.text[piece.start : piece.end])
        for begin, finish in self.protected:
            for position in range(max(begin, piece.start), min(finish, piece.end)):
                chars[position - piece.start] = " "
        return "".join(chars)

    def _protected(self, position: int) -> bool:
        return any(begin <= position < finish for begin, finish in self.protected)

    def _between_digits(self, at: int) -> bool:
        text = self.text
        return 0 < at < len(text) - 1 and text[at - 1].isdigit() and text[at + 1].isdigit()

    def _cuts(self, start: int, end: int, cuts: Sequence[int]) -> list[tuple[int, int]]:
        bounds = [start, *cuts, end]
        spans = [
            self._trim(bounds[i] + (1 if i else 0), bounds[i + 1]) for i in range(len(bounds) - 1)
        ]
        return [(s, e) for s, e in spans if s < e]

    def _trim(self, start: int, end: int) -> tuple[int, int]:
        while start < end and self.text[start] in _EDGE:
            start += 1
        while end > start and self.text[end - 1] in _EDGE:
            end -= 1
        return start, end

    def _tokens(self, start: int, end: int) -> tuple[list[str], list[tuple[int, int]]]:
        found = list(_WORD.finditer(self.text, start, end))
        return [m[0] for m in found], [(m.start(), m.end()) for m in found]

    def _text(self, start: int, end: int) -> str:
        return self.text[start:end]

    def _show(self, part: Piece | Fragment) -> str:
        return self.text[part.start : part.end]

    def _step(self, action: str, detail: str) -> None:
        self.trace.append(text_step(action, detail))
