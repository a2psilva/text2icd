"""`mimic-gold`: a stratified sample of discharge diagnosis items, coded by hand.

1. `sample_candidates`: silver items sampled by frequency band of their code (head, torso, tail)
   and by chapter, each with its admission's billed codes as candidates.
2. `annotate`: a terminal session in which a person codes each phrase (docs/annotation-guide.md):
   candidate numbers or codes, primary first; `s` skips a phrase that cannot be coded, `?` marks
   it unsure; lookups search the Alphabetic Index and show a code. Answers are saved after each
   phrase, so a session can stop and resume.
3. `build_gold`: the answered phrases as a dataset.

Every file holds MIMIC-IV phrases and stays under the git-ignored data directory.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from text2icd.eval.datasets import Example, GoldCode
from text2icd.eval.mimic.splits import split_of_subject
from text2icd.ontology.codes import dot, is_code

SOURCE = "mimic-gold"
BANDS = ("head", "torso", "tail")
# Codes whose silver examples make up the first half of all examples are the head, the next 40%
# the torso, and the last 10% the tail.
HEAD_SHARE = 0.5
TORSO_SHARE = 0.9
SAMPLE_SIZE = 500


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    phrase: str
    subject_id: str
    hadm_id: str
    note_id: str
    silver: str  # the silver code, for comparison after annotation; never shown while coding
    band: str
    chapter: str
    choices: tuple[tuple[str, str], ...]  # the admission's billed codes and descriptions


Verdict = Literal["coded", "skip", "unsure"]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    verdict: Verdict
    codes: tuple[str, ...] = ()  # primary first
    comment: str = ""


def frequency_bands(examples: Sequence[Example]) -> dict[str, str]:
    """The band of each silver code by how much of the set its examples make up."""
    counts = Counter(e.primary.code for e in examples)
    total = sum(counts.values())
    bands: dict[str, str] = {}
    running = 0
    for code, n in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        share = running / total if total else 0
        bands[code] = "head" if share < HEAD_SHARE else "torso" if share < TORSO_SHARE else "tail"
        running += n
    return bands


def sample_candidates(
    examples: Sequence[Example],
    choices: Callable[[Example], Sequence[tuple[str, str]]],
    chapter_of: Callable[[Example], str],
    size: int = SAMPLE_SIZE,
    seed: int = 0,
) -> list[Candidate]:
    """A third of the sample from each frequency band; within a band, chapters in proportion
    to their examples, at least one each while the band's quota lasts. One item per note."""
    rng = random.Random(f"{seed}:gold")
    bands = frequency_bands(examples)
    by_band: dict[str, dict[str, list[Example]]] = defaultdict(lambda: defaultdict(list))
    for example in sorted(examples, key=lambda e: e.id):
        by_band[bands[example.primary.code]][chapter_of(example)].append(example)
    chosen: list[Example] = []
    notes: set[str] = set()
    for band in BANDS:
        chapters = by_band.get(band, {})
        quota = size // len(BANDS) + (1 if BANDS.index(band) < size % len(BANDS) else 0)
        for chapter, share in _shares(chapters, quota).items():
            pool = chapters[chapter][:]
            rng.shuffle(pool)
            taken = 0
            for example in pool:
                if taken >= share:
                    break
                note = str(example.meta.get("note_id"))
                if note in notes:
                    continue
                notes.add(note)
                chosen.append(example)
                taken += 1
    return [
        Candidate(
            id=e.id,
            phrase=e.phrase,
            subject_id=str(e.meta["subject_id"]),
            hadm_id=str(e.meta["hadm_id"]),
            note_id=str(e.meta["note_id"]),
            silver=e.primary.code,
            band=bands[e.primary.code],
            chapter=chapter_of(e),
            choices=tuple(choices(e)),
        )
        for e in sorted(chosen, key=lambda e: e.id)
    ]


def _shares(chapters: Mapping[str, Sequence[Example]], quota: int) -> dict[str, int]:
    """Each chapter's share of a band's quota: in proportion to its examples, at least one, and
    never more than it has; the largest shares give way when the minimums overshoot."""
    total = sum(len(v) for v in chapters.values())
    if not total or quota <= 0:
        return {}
    shares = {
        chapter: min(len(pool), max(1, round(quota * len(pool) / total)))
        for chapter, pool in chapters.items()
    }
    while sum(shares.values()) > quota:
        largest = max(shares, key=lambda c: (shares[c], c))
        shares[largest] -= 1
    return {chapter: n for chapter, n in sorted(shares.items()) if n > 0}


def write_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
    temporary.replace(path)


def read_candidates(path: Path) -> list[Candidate]:
    with path.open(encoding="utf-8") as fh:
        return [Candidate.model_validate_json(line) for line in fh if line.strip()]


def read_answers(path: Path) -> dict[str, Answer]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        answers = [Answer.model_validate_json(line) for line in fh if line.strip()]
    return {answer.id: answer for answer in answers}


# -- The coding session ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Parsed:
    answer: Answer | None = None
    command: str | None = None  # quit, back, help, lookup, show
    argument: str = ""
    error: str | None = None


HELP = """Enter the codes for the phrase, primary first: candidate numbers (1 3) or codes
(E11.22 N18.30), or both. Then:
  s [comment]   skip: not a codable diagnosis        ? [comment]  unsure
  l <words>     look up Index terms                  c <code>     show a code
  b             back to the previous phrase          q            save and quit
  h             this help"""


def parse_answer(text: str, candidate: Candidate, billable: Callable[[str], bool]) -> Parsed:
    """Read one line typed for a phrase."""
    line = text.strip()
    if not line:
        return Parsed(error="type codes, or h for help")
    head, _, rest = line.partition(" ")
    key = head.lower()
    if key in ("q", "b", "h"):
        return Parsed(command={"q": "quit", "b": "back", "h": "help"}[key])
    if key in ("l", "c"):
        if not rest.strip():
            return Parsed(error=f"{key} needs {'words' if key == 'l' else 'a code'}")
        return Parsed(command="lookup" if key == "l" else "show", argument=rest.strip())
    if key in ("s", "?"):
        verdict: Verdict = "skip" if key == "s" else "unsure"
        return Parsed(answer=Answer(id=candidate.id, verdict=verdict, comment=rest.strip()))
    codes: list[str] = []
    for token in line.replace(",", " ").split():
        if token.isdigit():
            number = int(token)
            if not 1 <= number <= len(candidate.choices):
                return Parsed(error=f"no candidate {number}")
            code = candidate.choices[number - 1][0]
        elif is_code(token):
            code = dot(token)
        else:
            return Parsed(error=f"{token!r} is neither a candidate number nor a code")
        if not billable(code):
            return Parsed(error=f"{code} is not a billable code of the release")
        if code not in codes:
            codes.append(code)
    return Parsed(answer=Answer(id=candidate.id, verdict="coded", codes=tuple(codes)))


@dataclass
class Session:
    """A coding session over the candidates, saving after each answer."""

    candidates: Sequence[Candidate]
    answers: dict[str, Answer]
    billable: Callable[[str], bool]
    save: Callable[[Mapping[str, Answer]], None]
    lookup: Callable[[str], list[str]]
    show: Callable[[str], list[str]]
    ask: Callable[[str], str]
    say: Callable[[str], None]
    position: int = field(default=0)

    def run(self) -> int:
        """Code the unanswered phrases in order; returns how many were answered this time."""
        done = 0
        self.position = next(
            (i for i, c in enumerate(self.candidates) if c.id not in self.answers),
            len(self.candidates),
        )
        while self.position < len(self.candidates):
            candidate = self.candidates[self.position]
            self._present(candidate)
            while True:
                parsed = parse_answer(self.ask("> "), candidate, self.billable)
                if parsed.error:
                    self.say(f"  {parsed.error}")
                    continue
                if parsed.command == "quit":
                    return done
                if parsed.command == "help":
                    self.say(HELP)
                    continue
                if parsed.command == "lookup":
                    self.say("\n".join(f"  {line}" for line in self.lookup(parsed.argument)))
                    continue
                if parsed.command == "show":
                    self.say("\n".join(f"  {line}" for line in self.show(parsed.argument)))
                    continue
                if parsed.command == "back":
                    self.position = max(0, self.position - 1)
                    break
                if parsed.answer is not None:
                    self.answers[candidate.id] = parsed.answer
                    self.save(self.answers)
                    done += 1
                    self.position += 1
                    break
        self.say("all phrases answered")
        return done

    def _present(self, candidate: Candidate) -> None:
        answered = sum(c.id in self.answers for c in self.candidates)
        self.say("")
        self.say(f"[{self.position + 1}/{len(self.candidates)}, {answered} answered]")
        self.say(f"  {candidate.phrase}")
        for number, (code, description) in enumerate(candidate.choices, start=1):
            self.say(f"  {number:>2}. {code:<9} {description}")
        previous = self.answers.get(candidate.id)
        if previous is not None:
            shown = " ".join(previous.codes) if previous.codes else previous.verdict
            self.say(f"  (answered: {shown})")


def build_gold(
    candidates: Sequence[Candidate], answers: Mapping[str, Answer], release: str
) -> tuple[list[Example], Counter[str]]:
    """The coded phrases as examples (all codes valid in `release`, checked when typed), and
    the count of each verdict."""
    verdicts: Counter[str] = Counter()
    examples = []
    for candidate in candidates:
        answer = answers.get(candidate.id)
        if answer is None:
            verdicts["unanswered"] += 1
            continue
        verdicts[answer.verdict] += 1
        if answer.verdict != "coded" or not answer.codes:
            continue
        gold = [GoldCode(code=answer.codes[0], role="primary")]
        gold += [GoldCode(code=code, role="companion") for code in answer.codes[1:]]
        examples.append(
            Example(
                id=candidate.id,
                phrase=candidate.phrase,
                gold=tuple(gold),
                release=release,
                source=SOURCE,
                split=split_of_subject(candidate.subject_id),
                meta={
                    "note_id": candidate.note_id,
                    "subject_id": candidate.subject_id,
                    "hadm_id": candidate.hadm_id,
                    "band": candidate.band,
                    "chapter": candidate.chapter,
                    "silver": candidate.silver,
                    "comment": answer.comment,
                },
            )
        )
    return examples, verdicts


def agreement_with_silver(examples: Sequence[Example]) -> str:
    """How often the hand-coded primary code equals the silver code (an aggregate check of the
    silver labels, alongside the audit)."""
    if not examples:
        return "no coded phrases"
    same = sum(e.primary.code == e.meta.get("silver") for e in examples)
    return f"primary code equals the silver code for {same / len(examples):.1%} of {len(examples)}"


def answers_to_jsonl(path: Path) -> Callable[[Mapping[str, Answer]], None]:
    def save(answers: Mapping[str, Answer]) -> None:
        write_jsonl(path, list(answers.values()))

    return save
