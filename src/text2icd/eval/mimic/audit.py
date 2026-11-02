"""Hand audit of silver labels: a seeded random sample of alignments, reviewed by a person, and
the label precision with a 95% Wilson interval.

The review file holds MIMIC-IV phrases, so it lives under the git-ignored data directory and is
reviewed locally (`text2icd eval mimic audit review` in a terminal, or any spreadsheet editor).
Verdicts: `y` (the code is right for the phrase), `n` (wrong), `?` (cannot tell); `?` is left
out of the precision and counted.
"""

import csv
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from text2icd.eval.datasets import Example

SAMPLE_SIZE = 200
COLUMNS = ("id", "phrase", "code", "description", "score", "verdict", "comment")
VERDICTS = ("y", "n", "?")
Z = 1.959964  # two-sided 95%


@dataclass(frozen=True, slots=True)
class Precision:
    right: int
    wrong: int
    unsure: int
    unreviewed: int

    @property
    def reviewed(self) -> int:
        return self.right + self.wrong

    @property
    def value(self) -> float:
        return self.right / self.reviewed if self.reviewed else math.nan

    def interval(self) -> tuple[float, float]:
        return wilson(self.right, self.reviewed)

    def line(self, label: str = "precision") -> str:
        if not self.reviewed:
            return f"{label}: nothing reviewed yet ({self.unreviewed} to review)"
        low, high = self.interval()
        return (
            f"{label}: {self.value:.1%} (95% CI {low:.1%}-{high:.1%}), {self.right} right of "
            f"{self.reviewed}; {self.unsure} unsure, {self.unreviewed} not reviewed"
        )


def wilson(successes: int, trials: int, z: float = Z) -> tuple[float, float]:
    """The Wilson score interval of a proportion."""
    if trials == 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def sample_for_review(
    examples: Sequence[Example], size: int = SAMPLE_SIZE, seed: int = 0
) -> list[Example]:
    """A seeded random sample of silver examples, from the dev split only (the test split is
    not looked at)."""
    dev = sorted((e for e in examples if e.split == "dev"), key=lambda e: e.id)
    return random.Random(f"{seed}:audit").sample(dev, min(size, len(dev)))


def write_review(path: Path, examples: Sequence[Example], describe: Callable[[str], str]) -> int:
    """Write the review file; refuses to overwrite one that already has verdicts."""
    if path.is_file() and any(row["verdict"].strip() for row in read_review(path)):
        raise FileExistsError(f"{path} already has verdicts; review or move it first")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, COLUMNS)
        writer.writeheader()
        for example in examples:
            code = example.primary.code
            writer.writerow(
                {
                    "id": example.id,
                    "phrase": example.phrase,
                    "code": code,
                    "description": describe(code),
                    "score": example.meta.get("score", ""),
                    "verdict": "",
                    "comment": "",
                }
            )
    return len(examples)


def read_review(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def save_review(
    path: Path, rows: Sequence[Mapping[str, str]], columns: Sequence[str] = COLUMNS
) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def precision(rows: Sequence[Mapping[str, str]]) -> Precision:
    verdicts = [row.get("verdict", "").strip().lower() for row in rows]
    return Precision(
        right=verdicts.count("y"),
        wrong=verdicts.count("n"),
        unsure=verdicts.count("?"),
        unreviewed=sum(v not in VERDICTS for v in verdicts),
    )


def by_score(rows: Sequence[Mapping[str, str]], edges: Sequence[float]) -> list[str]:
    """Precision per band of alignment score, to choose the thresholds."""
    lines = []
    bounds = [*edges, math.inf]
    for low, high in pairwise(bounds):
        band = [r for r in rows if r.get("score") and low <= float(r["score"]) < high]
        if band:
            label = f"score {low:.2f}+" if high == math.inf else f"score {low:.2f}-{high:.2f}"
            lines.append(precision(band).line(label))
    return lines


REVIEW_HELP = (
    "y: the code is right for the phrase, n: wrong, ?: cannot tell; add a comment after a space."
    " b: back, q: save and quit."
)


def _silver_lines(row: Mapping[str, str]) -> list[str]:
    return [
        f"score {row.get('score', '')}",
        f"  phrase: {row['phrase']}",
        f"  code:   {row['code']}  {row['description']}",
    ]


def review(
    rows: list[dict[str, str]],
    ask: Callable[[str], str],
    say: Callable[[str], None],
    save: Callable[[Sequence[Mapping[str, str]]], None],
    describe: Callable[[Mapping[str, str]], list[str]] = _silver_lines,
    help_text: str = REVIEW_HELP,
) -> int:
    """Ask for a verdict on each row without one, saving after each; returns how many were
    given this time. `describe` gives the lines shown for a row (a header, then details)."""
    given = 0
    position = next((i for i, r in enumerate(rows) if not r["verdict"].strip()), len(rows))
    while position < len(rows):
        row = rows[position]
        header, *details = describe(row)
        say("")
        say(f"[{position + 1}/{len(rows)}]  {header}")
        for line in details:
            say(line)
        answer = ask("> ").strip()
        key, _, comment = answer.partition(" ")
        key = key.lower()
        if key == "q":
            return given
        if key == "b":
            position = max(0, position - 1)
            continue
        if key not in VERDICTS:
            say(f"  {help_text}")
            continue
        row["verdict"] = key
        row["comment"] = comment.strip()
        save(rows)
        given += 1
        position += 1
    say("all rows reviewed")
    return given
