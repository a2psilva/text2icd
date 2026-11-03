"""The Tabular List's instructional notes inside a code set (Guidelines I.A.12, I.A.13, I.B.7).

- **Companion codes.** `code first`, `use additional code` and `code also` notes name codes that
  belong with a code (I.B.7). A note applies to the codes below the node that holds it, so a code
  reads its own notes and those above it. The items of a list ("code to identify control using:
  insulin (Z79.4)") are read with their heading, and each item is one instruction. Several
  notes of one kind at a node are *alternatives*: whichever applies is coded (I50: heart failure
  due to hypertension, following surgery, rheumatic, ...); unless each is a whole "code to
  identify ..." instruction (I13.0: the type of heart failure, and the stage of chronic kidney
  disease). An instruction is *conditional* when
  it says so ("if applicable", "if known", "any"), or when it only names external causes, which
  no national rule makes mandatory (I.C.20). An instruction that is neither is *mandatory*. The
  code of an optional instruction is added only when the phrase documents it beyond the words
  of the code that holds the note ("genetic susceptibility to malignant neoplasm" is no current
  neoplasm). "Use additional code" means "if known" (I.B.7), so a mandatory one left unresolved
  asks for review; it does not invalidate the set.
- **Excludes1.** Two codes of the set whose notes exclude one another (I.A.12.a) cannot both be
  kept: the more specific (more characters, not an unspecified or default code) and then the
  higher scoring one stays.
- **Sequencing.** `code first` puts the named code before the one that holds the note; `use
  additional code` puts it after; `code also` gives no order (I.B.7). An Index manifestation goes
  after its etiology, and a code "in diseases classified elsewhere" is never primary (I.A.13).
  The guideline rules (`rules`) add orders of their own (`Order`).
"""

import re
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from text2icd.graph.model import Graph
from text2icd.ontology.codes import CodeReference, undot
from text2icd.trace import TraceStep

Kind = Literal["codeFirst", "useAdditionalCode", "codeAlso"]
KINDS: tuple[Kind, ...] = ("codeFirst", "useAdditionalCode", "codeAlso")
Role = Literal[
    "diagnosis", "manifestation", "code_first", "use_additional", "code_also", "external_cause"
]
ROLES: dict[Kind, Role] = {
    "codeFirst": "code_first",
    "useAdditionalCode": "use_additional",
    "codeAlso": "code_also",
}
NAMES: dict[Kind, str] = {
    "codeFirst": "code first",
    "useAdditionalCode": "use additional code",
    "codeAlso": "code also",
}
_CONDITIONAL = re.compile(r"\b(?:if applicable|if known|if desired|if present|any)\b")
_ELSEWHERE = re.compile(r"\bclassified elsewhere\b")


@dataclass(frozen=True, slots=True)
class Instruction:
    """One `code first`, `use additional code` or `code also` instruction that applies to a code."""

    node: str  # the code or category whose note it is
    kind: Kind
    text: str  # the note, after the heading of its list when it is a list item
    references: tuple[CodeReference, ...]
    conditional: bool = False  # coded only when the phrase documents it on its own
    alternative: bool = False  # one of a list: whichever applies

    @property
    def mandatory(self) -> bool:
        return not (self.conditional or self.alternative)

    def covers(self, code: str) -> bool:
        return any(code in reference for reference in self.references)

    def reaches(self, code: str) -> bool:
        """Whether `code`, or a code the Tabular List completes it to, can be one it names."""
        bare = undot(code)
        for ref in self.references:
            low, high = min(len(bare), len(ref.start)), min(len(bare), len(ref.end))
            if ref.start[:low] <= bare[:low] and bare[:high] <= ref.end[:high]:
                return True
        return False


@dataclass(frozen=True, slots=True)
class Entry:
    """A code of the set before sequencing, with what decides Excludes1 conflicts."""

    code: str
    description: str
    role: Role
    source: str | None  # the code whose Index pair or note brought it in
    group: int  # the diagnosis it belongs to
    score: float  # the score of the Index path it came from
    specified: bool  # neither an unspecified nor a default code
    used: frozenset[str] = frozenset()  # the phrase words of that path


@dataclass(frozen=True, slots=True)
class Excluded:
    """A code an Excludes1 note took out of the set."""

    code: str
    kept: str
    note: str


class Notes:
    """The instructional notes of one release's codes, read once per code."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._instructions: dict[str, tuple[Instruction, ...]] = {}
        self._excludes: dict[str, tuple[tuple[CodeReference, str, str], ...]] = {}
        self._elsewhere: dict[str, bool] = {}

    def instructions(self, code: str) -> tuple[Instruction, ...]:
        """The companion instructions that apply to `code`: its own, then those above it."""
        found = self._instructions.get(code)
        if found is None:
            seen: set[tuple[str, tuple[CodeReference, ...]]] = set()
            out = []
            for node in self._lineage(code):
                for instruction in self._read(node):
                    key = (instruction.kind, instruction.references)
                    if key not in seen:  # "oral antidiabetic" and "oral hypoglycemic" (Z79.84)
                        seen.add(key)
                        out.append(instruction)
            found = self._instructions[code] = tuple(out)
        return found

    def excludes1(self, code: str) -> tuple[tuple[CodeReference, str, str], ...]:
        """The Excludes1 references that apply to `code`, with their note and the node holding
        it."""
        found = self._excludes.get(code)
        if found is None:
            found = self._excludes[code] = tuple(
                (_reference(edge.start, edge.end), edge.note, edge.source)
                for edge in self.graph.edges(code, "excludes1", inherited=True)
                if edge.status == "ok"
            )
        return found

    def excluding(self, first: str, second: str) -> tuple[str, str] | None:
        """The Excludes1 note, and the node holding it, by which one code excludes the other."""
        for holder, other in ((first, second), (second, first)):
            for reference, note, node in self.excludes1(holder):
                if other in reference:
                    return note, node
        return None

    def classified_elsewhere(self, code: str) -> bool:
        """A manifestation code: its title, or one above it, says "classified elsewhere"."""
        found = self._elsewhere.get(code)
        if found is None:
            found = self._elsewhere[code] = any(
                _ELSEWHERE.search(self.graph.code(node).description.casefold())
                for node in self._lineage(code)
            )
        return found

    def _lineage(self, code: str) -> list[str]:
        if code not in self.graph:
            return []
        return [code] + [a.key for a in self.graph.ancestors(code) if a.kind == "diag"]

    def _read(self, node: str) -> list[Instruction]:
        references: dict[int, list[CodeReference]] = {}
        for edge in self.graph.edges(node):
            if edge.kind in KINDS and edge.status == "ok":
                references.setdefault(edge.note_id, []).append(_reference(edge.start, edge.end))
        notes = [note for note in self.graph.notes(node) if note.kind in KINDS]
        sizes = Counter(note.kind for note in notes)
        # Several notes that each say "code to identify ..." are instructions of their own (I13:
        # the type of heart failure and the stage of CKD), not the items of one list.
        whole = {
            kind
            for kind in sizes
            if all(n.text.strip().casefold().startswith("code") for n in notes if n.kind == kind)
        }
        out = []
        heading: tuple[str, str] | None = None  # (kind, text) of the list being read
        for note in notes:
            kind = cast(Kind, note.kind)
            text = note.text.strip()
            if heading is not None and heading[0] != kind:
                heading = None
            if text.endswith(":"):
                heading = (kind, text)
            elif heading is not None:
                text = f"{heading[1]} {text}"
            found = tuple(references.get(note.id, ()))
            if not found:
                continue
            conditional = _CONDITIONAL.search(text.casefold()) is not None or all(
                ref.start[0] in "VWXY" for ref in found
            )
            alternative = sizes[kind] > 1 and (heading is not None or kind not in whole)
            out.append(Instruction(node, kind, text, found, conditional, alternative))
        return out


def _reference(start: str, end: str) -> CodeReference:
    return CodeReference(undot(start), undot(end))


def _step(action: str, detail: str, guideline: str | None) -> TraceStep:
    return TraceStep(layer="tabular", action=action, detail=detail, guideline=guideline)


def resolve_excludes1(
    entries: Sequence[Entry], notes: Notes, exempt: Collection[frozenset[int]] = ()
) -> tuple[list[Entry], list[Excluded], list[TraceStep]]:
    """Drop, of two codes of different diagnoses that an Excludes1 note keeps apart (I.A.12.a),
    the less specific one, with the codes it brought in. Diagnoses a guideline codes together
    (`exempt`: the acute and chronic codes of one condition, I.B.8) are left alone."""
    dropped: set[int] = set()
    excluded: list[Excluded] = []
    trace: list[TraceStep] = []
    for i, first in enumerate(entries):
        for j in range(i + 1, len(entries)):
            second = entries[j]
            if i in dropped or j in dropped or first.group == second.group:
                continue
            if frozenset({first.group, second.group}) in exempt:
                continue
            found = notes.excluding(first.code, second.code)
            if found is None:
                continue
            note, node = found
            keep, drop = (i, j) if _rank(first, i) >= _rank(second, j) else (j, i)
            dropped.add(drop)
            dropped |= _brought_in(entries, drop)
            kept, gone = entries[keep].code, entries[drop].code
            excluded.append(Excluded(gone, kept, note))
            trace.append(
                _step(
                    "excludes1",
                    f'{gone} and {kept}: Excludes1 at {node} "{note}": kept {kept}, the more '
                    "specific",
                    "I.A.12.a",
                )
            )
    return [e for k, e in enumerate(entries) if k not in dropped], excluded, trace


def _rank(entry: Entry, position: int) -> tuple[int, bool, float, int]:
    return (len(undot(entry.code)), entry.specified, entry.score, -position)


def _brought_in(entries: Sequence[Entry], index: int) -> set[int]:
    """The entries a code brought in (its manifestation, its companions), and theirs."""
    out: set[int] = set()
    group, sources = entries[index].group, {entries[index].code}
    while True:
        found = {
            k
            for k, e in enumerate(entries)
            if k not in out and k != index and e.group == group and e.source in sources
        }
        if not found:
            return out
        out |= found
        sources = {entries[k].code for k in found}


@dataclass(frozen=True, slots=True)
class Order:
    """A guideline's coding order for two codes of the set (`rules`)."""

    first: str
    then: str
    step: TraceStep


def sequence(
    entries: Sequence[Entry], notes: Notes, orders: Sequence[Order] = ()
) -> tuple[list[Entry], list[TraceStep]]:
    """The entries in coding order: the order of the phrase, moved as the guideline rules
    (`orders`), `code first` and `use additional code` notes and etiology/manifestation pairs
    require, with a code "in diseases classified elsewhere" never first (I.A.13)."""
    after: dict[int, dict[int, TraceStep]] = {i: {} for i in range(len(entries))}

    def require(first: int, then: int, step: TraceStep) -> None:
        if first != then:
            after[then].setdefault(first, step)

    positions: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        positions.setdefault(entry.code, []).append(i)
    for order in orders:
        for i in positions.get(order.first, ()):
            for j in positions.get(order.then, ()):
                require(i, j, order.step)
    for i, entry in enumerate(entries):
        if entry.role == "manifestation" and entry.source is not None:
            for j, other in enumerate(entries):
                if other.code == entry.source and other.group == entry.group:
                    require(
                        j,
                        i,
                        _step(
                            "sequence",
                            f"{other.code} before {entry.code}: etiology before manifestation",
                            "I.A.13",
                        ),
                    )
        for instruction in notes.instructions(entry.code):
            if instruction.kind == "codeAlso":
                continue  # gives no order (I.B.7)
            for j, other in enumerate(entries):
                if not instruction.covers(other.code):
                    continue
                name = NAMES[instruction.kind]
                if instruction.kind == "codeFirst":
                    detail = f'{other.code} before {entry.code}: {name} "{instruction.text}"'
                    require(j, i, _step("sequence", detail, "I.A.13"))
                else:
                    detail = f'{entry.code} before {other.code}: {name} "{instruction.text}"'
                    require(i, j, _step("sequence", detail, "I.B.7"))
    placed: list[int] = []
    remaining = list(range(len(entries)))
    while remaining:
        ready = next((i for i in remaining if set(after[i]) <= set(placed)), remaining[0])
        placed.append(ready)
        remaining.remove(ready)
    trace = [
        step
        for position, i in enumerate(placed)
        for first, step in after[i].items()
        if first > i and first in placed[:position]  # moved ahead of the phrase order
    ]
    if placed and notes.classified_elsewhere(entries[placed[0]].code):
        lead = next(
            (
                i
                for i in placed
                if not notes.classified_elsewhere(entries[i].code)
                and entries[i].role != "external_cause"
                and not after[i]
            ),
            None,
        )
        if lead is not None:
            first = entries[placed[0]].code
            placed.remove(lead)
            placed.insert(0, lead)
            trace.append(
                _step(
                    "sequence",
                    f"{entries[lead].code} first: {first} is a code in diseases classified "
                    "elsewhere, never primary",
                    "I.A.13",
                )
            )
    return [entries[i] for i in placed], trace
