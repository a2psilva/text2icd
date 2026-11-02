"""Text understanding end to end: a diagnosis phrase in, a `ParsedPhrase` out.

1. Normalise and expand abbreviations (`normalize`, `abbreviations`).
2. Find modifiers and cues in the normalised phrase (`modifiers`, `cues`).
3. Split it into conditions and relations (`split`).
4. Give each condition the cues that cover it, and judge its status, certainty and whether it is
   coded in the setting (`cues.judge`).
5. Group the conditions that linking relations join into diagnoses.

Which conditions a cue covers: one inside a condition's span covers it, except a bare `not` after
the condition's first words, which qualifies what follows ("epilepsy not intractable": the
epilepsy is affirmed, and "not" stays in its words for the Index). A cue before conditions
also covers those after it in its comma fragment, and a cue after them those before it, until a
cue of the same family (status, or certainty) takes over. Status cues stop at linking
connectors: in "status post fall with hip fracture" the fracture is current. Conditions listed
with `vs` or `or` are both uncertain, coded as if confirmed inpatient (II.D); a condition linked
by `without` is absent.
"""

from collections.abc import Sequence

from text2icd.graph.model import Graph
from text2icd.text.abbreviations import Expansion, Lexicon, default_lexicon
from text2icd.text.cues import cues as find_cues
from text2icd.text.cues import judge
from text2icd.text.model import Condition, Cue, CueKind, ParsedPhrase, Relation, Setting
from text2icd.text.modifiers import modifiers as find_modifiers
from text2icd.text.normalize import normalize
from text2icd.text.split import Fragment, Piece, Splitter
from text2icd.text.terms import Terms
from text2icd.trace import TraceStep, text_step

COMPARATIVE = frozenset({"versus", "or"})


class Parser:
    """Parses phrases with one lexicon, one set of graph terms and a default setting."""

    def __init__(
        self,
        terms: Terms | None = None,
        abbreviations: Lexicon | None = None,
        setting: Setting = "inpatient",
    ) -> None:
        self.terms = terms
        self.abbreviations = abbreviations if abbreviations is not None else default_lexicon()
        self.setting = setting

    @classmethod
    def from_graph(cls, graph: Graph, setting: Setting = "inpatient") -> "Parser":
        return cls(Terms.from_graph(graph), setting=setting)

    def parse(self, phrase: str, setting: Setting | None = None) -> ParsedPhrase:
        setting = setting or self.setting
        vocabulary = self.terms.vocabulary if self.terms is not None else None
        normalized = normalize(phrase, self.abbreviations, vocabulary)
        text = normalized.text
        modifiers = find_modifiers(text)
        cues = find_cues(text)
        splitter = Splitter(text, modifiers, cues, self.terms)
        fragments = splitter.split()
        pieces = [piece for fragment in fragments for piece in fragment.pieces]
        number = {id(piece): index for index, piece in enumerate(pieces)}
        relations = tuple(
            Relation(
                kind=link.kind,
                head=number[id(link.head)],
                linked=number[id(link.linked)],
                text=link.text,
            )
            for fragment in fragments
            for link in fragment.links
        )
        diagnoses = _groups(len(pieces), relations)
        group_of = {index: g for g, members in enumerate(diagnoses) for index in members}
        trace: list[TraceStep] = [*normalized.trace, *splitter.trace]
        conditions = []
        for fragment in fragments:
            for piece in fragment.pieces:
                index = number[id(piece)]
                implied = _implied(piece, fragment)
                covering = _covering(piece, fragment, cues) + implied
                judgement = judge(covering, setting)
                own = [cue for cue in covering if piece.start <= cue.start < piece.end]
                condition = Condition(
                    text=_without(text, piece, own),
                    start=piece.start,
                    end=piece.end,
                    modifiers=tuple(m for m in modifiers if piece.start <= m.start < piece.end),
                    cues=tuple(covering),
                    status=judgement.status,
                    certainty=judgement.certainty,
                    reportable=judgement.reportable,
                    diagnosis=group_of[index],
                )
                conditions.append(condition)
                trace += [_about(condition.text, step) for step in judgement.trace]
                comparative = any(cue.kind == "uncertain" for cue in implied)
                if comparative and setting == "inpatient" and condition.certainty == "uncertain":
                    trace.append(
                        text_step(
                            "comparative",
                            f'"{condition.text}": a comparative diagnosis, coded as if confirmed',
                            "II.D",
                        )
                    )
        return ParsedPhrase(
            original=phrase,
            normalized=text,
            setting=setting,
            conditions=tuple(conditions),
            relations=relations,
            diagnoses=diagnoses,
            expansions=normalized.expansions,
            unknown=normalized.unknown,
            flags=_flags(conditions, normalized.expansions, normalized.unknown),
            trace=tuple(trace),
        )


def parse(
    phrase: str,
    terms: Terms | None = None,
    abbreviations: Lexicon | None = None,
    setting: Setting = "inpatient",
) -> ParsedPhrase:
    return Parser(terms, abbreviations, setting).parse(phrase)


def _covering(piece: Piece, fragment: Fragment, cues: Sequence[Cue]) -> list[Cue]:
    """The cues of the fragment that cover the piece: its own, and those before or after it
    with no cue of the same family in between (and, for status cues, no linking connector)."""
    inside = [
        c for c in cues if fragment.start <= c.start < fragment.end and not _qualifies(c, piece)
    ]
    connectors = [
        (link.start, link.end) for link in fragment.links if link.kind not in COMPARATIVE | {"and"}
    ]
    own = {cue.family for cue in inside if piece.start <= cue.start < piece.end}
    covering = []
    for cue in inside:
        if piece.start <= cue.start < piece.end:
            covering.append(cue)
            continue
        if cue.family in own:
            continue  # the condition's own cue of that family decides
        if cue.position == "before" and cue.end <= piece.start:
            gap = (cue.end, piece.start)
        elif cue.position == "after" and cue.start >= piece.end:
            gap = (piece.end, cue.start)
        else:
            continue
        if any(gap[0] <= other.start < gap[1] and other.family == cue.family for other in inside):
            continue
        if cue.family == "status" and any(gap[0] <= start < gap[1] for start, _ in connectors):
            continue
        covering.append(cue)
    return covering


def _qualifies(cue: Cue, piece: Piece) -> bool:
    """A bare `not` after the first words of a condition qualifies what follows it, not the
    condition: "epilepsy not intractable", "hernia not specified as recurrent"."""
    return cue.kind == "negated" and cue.text == "not" and piece.start < cue.start < piece.end


def _implied(piece: Piece, fragment: Fragment) -> list[Cue]:
    """Cues the connectors imply: `vs` and `or` make both sides uncertain (II.D), and `without`
    makes the linked condition absent."""
    implied = []
    for link in fragment.links:
        sides = (link.head, link.linked)
        if link.kind in COMPARATIVE and any(side is piece for side in sides):
            implied.append(_cue("uncertain", link.text, link.start, link.end))
        elif link.kind == "without" and link.linked is piece:
            implied.append(_cue("negated", link.text, link.start, link.end))
    return implied


def _cue(kind: CueKind, text: str, start: int, end: int) -> Cue:
    return Cue(kind=kind, text=text, start=start, end=end, position="before")


def _without(text: str, piece: Piece, own: Sequence[Cue]) -> str:
    chars = list(text[piece.start : piece.end])
    for cue in own:
        for position in range(max(cue.start, piece.start), min(cue.end, piece.end)):
            chars[position - piece.start] = " "
    return " ".join("".join(chars).split()).strip(" ,;:-()")


def _about(condition: str, step: TraceStep) -> TraceStep:
    return step.model_copy(update={"detail": f'"{condition}": {step.detail}'})


def _groups(count: int, relations: Sequence[Relation]) -> tuple[tuple[int, ...], ...]:
    parent = list(range(count))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for relation in relations:
        if relation.links:
            parent[root(relation.linked)] = root(relation.head)
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    return tuple(tuple(members) for members in sorted(groups.values()))


def _flags(
    conditions: Sequence[Condition], expansions: Sequence[Expansion], unknown: Sequence[str]
) -> tuple[str, ...]:
    flags: set[str] = set()
    if any(expansion.chosen_by == "default" for expansion in expansions):
        flags.add("ambiguous_abbreviation")
    if unknown:
        flags.add("unknown_abbreviation")
    for condition in conditions:
        if condition.certainty != "affirmed":
            flags.add(condition.certainty)
        if condition.status != "current":
            flags.add(condition.status)
    if not conditions:
        flags.add("no_condition")
    return tuple(sorted(flags))


def describe_parsed(parsed: ParsedPhrase) -> list[str]:
    """Readable lines: the normalised phrase, its diagnoses and conditions, and the trace."""
    lines = [f"normalized: {parsed.normalized}", f"setting: {parsed.setting}"]
    for number, members in enumerate(parsed.diagnoses, start=1):
        lines.append(f"diagnosis {number}:")
        for index in members:
            condition = parsed.conditions[index]
            state = f"{condition.status}, {condition.certainty}"
            if not condition.reportable:
                state += ", not coded"
            lines.append(f"  [{index + 1}] {condition.text}  ({state})")
            if condition.modifiers:
                found = ", ".join(f"{m.kind}={m.value}" for m in condition.modifiers)
                lines.append(f"      modifiers: {found}")
    for relation in parsed.relations:
        lines.append(f"relation: [{relation.head + 1}] {relation.kind} [{relation.linked + 1}]")
    if parsed.flags:
        lines.append(f"flags: {', '.join(parsed.flags)}")
    lines.append("trace:")
    for step in parsed.trace:
        guideline = f"  ({step.guideline})" if step.guideline else ""
        lines.append(f"  {step.action}: {step.detail}{guideline}")
    return lines
