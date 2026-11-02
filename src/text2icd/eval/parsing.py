"""How text understanding reads a dataset: aggregate counts, and agreement on the number of
diagnoses each phrase names.

The expected number of diagnoses comes from the example: `meta.diagnoses` when the annotation
recorded it (`mimic-gold`, where the coder separates diagnoses with commas), or one for
`cms-synthetic`, whose phrases each come from one Index entry, Tabular term or table cell.
Agreement is the share of those phrases split into exactly that many diagnoses (Phase 4 exit
criterion: at least 95% on `mimic-gold` dev), with a Wilson interval.

Everything reported is an aggregate. Mismatched phrases are kept only on request, and the
command refuses to show them for restricted datasets.
"""

import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from text2icd.eval.datasets import Example
from text2icd.eval.mimic.audit import wilson
from text2icd.text.parse import Parser

ONE_DIAGNOSIS_SOURCES = frozenset({"cms-synthetic"})


def expected_diagnoses(example: Example) -> int | None:
    value = example.meta.get("diagnoses")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1 if example.source in ONE_DIAGNOSIS_SOURCES else None


@dataclass
class Mismatch:
    id: str
    phrase: str
    expected: int
    found: int
    conditions: tuple[str, ...]


@dataclass
class ParseStats:
    phrases: int = 0
    seconds: float = 0.0
    diagnoses: Counter[int] = field(default_factory=Counter)  # diagnoses found -> phrases
    expected: int = 0  # phrases with an expected number of diagnoses
    agreed: int = 0
    too_many: int = 0
    too_few: int = 0
    with_expansion: int = 0
    flags: Counter[str] = field(default_factory=Counter)  # phrases with each flag
    relations: Counter[str] = field(default_factory=Counter)  # relations by kind
    modifiers: Counter[str] = field(default_factory=Counter)  # phrases with each kind
    mismatches: list[Mismatch] = field(default_factory=list)

    def lines(self) -> list[str]:
        def share(n: int) -> str:
            return f"{n / self.phrases:.1%}" if self.phrases else "n/a"

        rate = self.phrases / self.seconds if self.seconds else 0.0
        counts = ", ".join(
            f"{n if n < 4 else '4+'}: {share(c)}"
            for n, c in sorted(_bucket(self.diagnoses).items())
        )
        lines = [
            f"phrases {self.phrases:,} ({rate:,.0f} per second)",
            f"diagnoses per phrase: {counts}",
        ]
        if self.expected:
            low, high = wilson(self.agreed, self.expected)
            lines.append(
                f"same number of diagnoses as expected: {self.agreed / self.expected:.1%} "
                f"(95% CI {low:.1%}-{high:.1%}), {self.agreed:,} of {self.expected:,}; "
                f"too many {self.too_many:,}, too few {self.too_few:,}"
            )
        else:
            lines.append("same number of diagnoses as expected: no phrase has an expected number")
        lines.append(f"phrases with an abbreviation expanded: {share(self.with_expansion)}")
        lines.append(
            "phrases flagged: "
            + (", ".join(f"{k} {share(v)}" for k, v in sorted(self.flags.items())) or "none")
        )
        lines.append(
            "relations: "
            + (", ".join(f"{k} {v:,}" for k, v in sorted(self.relations.items())) or "none")
        )
        lines.append(
            "phrases with a modifier: "
            + (", ".join(f"{k} {share(v)}" for k, v in sorted(self.modifiers.items())) or "none")
        )
        return lines


def _bucket(counts: Counter[int]) -> Counter[int]:
    out: Counter[int] = Counter()
    for n, c in counts.items():
        out[min(n, 4)] += c
    return out


def parse_stats(examples: Iterable[Example], parser: Parser, *, keep: int = 0) -> ParseStats:
    """Parse every phrase and count; keep up to `keep` mismatched phrases."""
    stats = ParseStats()
    start = time.perf_counter()
    for example in examples:
        parsed = parser.parse(example.phrase)
        stats.phrases += 1
        found = len(parsed.diagnoses)
        stats.diagnoses[found] += 1
        stats.with_expansion += bool(parsed.expansions)
        stats.flags.update(parsed.flags)
        stats.relations.update(relation.kind for relation in parsed.relations)
        stats.modifiers.update({m.kind for c in parsed.conditions for m in c.modifiers})
        expected = expected_diagnoses(example)
        if expected is None:
            continue
        stats.expected += 1
        if found == expected:
            stats.agreed += 1
            continue
        if found > expected:
            stats.too_many += 1
        else:
            stats.too_few += 1
        if len(stats.mismatches) < keep:
            stats.mismatches.append(
                Mismatch(
                    example.id,
                    example.phrase,
                    expected,
                    found,
                    tuple(c.text for c in parsed.conditions),
                )
            )
    stats.seconds = time.perf_counter() - start
    return stats
