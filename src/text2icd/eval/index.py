"""How well the Index walk alone codes a dataset, before Tabular verification.

For each phrase the walk (`resolve.walk`) gives ranked candidates per diagnosis; the system's
answer is the first diagnosis walked. A candidate matches the primary gold code when the gold
code starts with it: the Index often gives a partial code (`S72.9-`), which the Tabular List
completes (Phase 6), so this is the "prefix match before Tabular verification" of the Phase 5
exit criterion. Gold codes mapped to a release (`status: mapped`) match through any code they map
to. Also counted: the first candidate's category, how often the default code was used, and
phrases with no candidate. Counts are given in total and by generator (`meta.generator`, as
`cms-synthetic` records it).

Everything reported is an aggregate. Mismatched phrases are kept only on request, and the
command refuses to show them for restricted datasets.
"""

import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from text2icd.eval.datasets import Example
from text2icd.eval.mimic.audit import wilson
from text2icd.resolve.walk import DEFAULT_CODE, Candidate, IndexWalker, WalkResult

DEPTHS = (1, 5)
ALL = "all"


def gold_codes(example: Example) -> tuple[str, ...]:
    gold = example.primary
    return (gold.code,) if gold.status == "valid" else gold.mapped_to


def ranked(result: WalkResult) -> tuple[Candidate, ...]:
    """The candidates of the first diagnosis walked, best first."""
    return result.diagnoses[0].candidates if result.diagnoses else ()


def prefix_match(candidate: Candidate, codes: Sequence[str]) -> bool:
    return any(code.startswith(candidate.code) for code in codes)


@dataclass
class Tally:
    phrases: int = 0
    found: Counter[int] = field(default_factory=Counter)  # depth -> prefix matches
    category: int = 0  # first candidate in the gold category
    default_code: int = 0  # first candidate is a default code
    none: int = 0  # no candidate

    def add(self, candidates: Sequence[Candidate], codes: Sequence[str]) -> int | None:
        """Count one phrase; the rank of the first matching candidate, if any."""
        self.phrases += 1
        if not candidates:
            self.none += 1
            return None
        first = candidates[0]
        self.category += any(code[:3] == first.code[:3] for code in codes)
        self.default_code += DEFAULT_CODE in first.flags
        rank = next((i for i, c in enumerate(candidates) if prefix_match(c, codes)), None)
        for depth in DEPTHS:
            self.found[depth] += rank is not None and rank < depth
        return rank


@dataclass
class Miss:
    id: str
    phrase: str
    gold: str
    found: str  # the first candidate and its path, or "no candidate"


@dataclass
class IndexStats:
    seconds: float = 0.0
    tallies: dict[str, Tally] = field(default_factory=dict)  # "all" and each generator
    diagnoses: Counter[int] = field(default_factory=Counter)  # diagnoses walked -> phrases
    misses: list[Miss] = field(default_factory=list)

    def lines(self) -> list[str]:
        total = self.tallies.get(ALL, Tally())
        rate = total.phrases / self.seconds if self.seconds else 0.0
        walked = ", ".join(f"{n}: {c:,}" for n, c in sorted(self.diagnoses.items()))
        lines = [
            f"phrases {total.phrases:,} ({rate:,.0f} per second); diagnoses walked {walked}",
            "prefix match of the primary gold code (gold code starts with the candidate):",
        ]
        groups = [ALL, *sorted(name for name in self.tallies if name != ALL)]
        if len(groups) == 2:  # one generator: the total says it all
            groups = [ALL]
        for name in groups:
            lines.extend(_tally_lines(name, self.tallies[name]))
        return lines


def _tally_lines(name: str, tally: Tally) -> list[str]:
    if not tally.phrases:
        return [f"  {name}: no examples"]

    def share(n: int) -> str:
        return f"{n / tally.phrases:.1%}"

    first = tally.found[1]
    low, high = wilson(first, tally.phrases)
    return [
        f"  {name} ({tally.phrases:,}): top-1 {share(first)} (95% CI {low:.1%}-{high:.1%}), "
        f"top-5 {share(tally.found[5])}, category {share(tally.category)}, "
        f"default code {share(tally.default_code)}, no candidate {share(tally.none)}"
    ]


def index_stats(examples: Iterable[Example], walker: IndexWalker, *, keep: int = 0) -> IndexStats:
    """Walk every phrase and count; keep up to `keep` phrases whose first candidate misses."""
    stats = IndexStats()
    start = time.perf_counter()
    for example in examples:
        result = walker.walk(example.phrase)
        stats.diagnoses[len(result.diagnoses)] += 1
        candidates = ranked(result)
        codes = gold_codes(example)
        groups = [ALL, str(example.meta.get("generator", example.source))]
        ranks = [stats.tallies.setdefault(g, Tally()).add(candidates, codes) for g in groups]
        if ranks[0] != 0 and len(stats.misses) < keep:
            found = (
                f"{candidates[0].written} at {candidates[0].path}" if candidates else "no candidate"
            )
            stats.misses.append(Miss(example.id, example.phrase, ", ".join(codes), found))
    stats.seconds = time.perf_counter() - start
    return stats
