"""The local review mode (task 10.12): the silver-label audit (task 3.6) and the gold annotation
(task 3.7) in the browser, from `text2icd serve --review audit|gold`.

Both reviews keep their files and rules, so a review can move between the terminal and the
browser and resume either way:

- the audit's verdicts go to its CSV through `audit.save_review`;
- the gold answers are read by `annotate.parse_answer` (the terminal's grammar: candidate
  numbers or codes, commas between diagnoses, `s` and `?`) and go to `answers.jsonl`.

Each request reads the file, and each answer writes it back (atomically, under a lock), so the
server holds no copy that a terminal session could make stale.

The phrases are MIMIC-IV text. `text2icd serve` binds 127.0.0.1 only in this mode, with no
access log, and every `/v1/` response says `Cache-Control: no-store`. The gold annotation never
sends the silver code, its band or chapter, or the admission's ids: the person codes blind. The
audit shows the silver code, since that is what is judged, but not its alignment score.
"""

import csv
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path as FilePath
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field

from text2icd.eval.mimic import annotate, audit
from text2icd.graph.model import Graph

Mode = Literal["audit", "gold"]
MODES: tuple[Mode, ...] = ("audit", "gold")
Comment = Annotated[str, Field(max_length=500)]
Item = Annotated[int, Path(ge=0, description="The item's position in the file, from 0.")]


class ReviewError(ValueError):
    """The review file is missing or unreadable, or an answer cannot be read."""


class Progress(BaseModel):
    mode: Mode
    total: int
    answered: int
    next: int | None  # the first item without an answer, from 0


class AuditItem(BaseModel):
    n: int
    phrase: str
    code: str  # the silver code under review
    description: str
    verdict: str  # "" until given: y, n or ?
    comment: str


class AuditVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["y", "n", "?"]
    comment: Comment = ""


class Choice(BaseModel):
    code: str
    description: str


class GoldAnswer(BaseModel):
    verdict: annotate.Verdict
    codes: list[Choice]
    comment: str
    diagnoses: int | None


class GoldItem(BaseModel):
    n: int
    phrase: str
    choices: list[Choice]  # the admission's billed codes
    answer: GoldAnswer | None


class GoldLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line: Annotated[str, Field(min_length=1, max_length=500)]  # as typed in the terminal
    comment: Comment = ""


class Checked(BaseModel):
    """A line read without saving it: the codes it gives, or why it cannot be saved."""

    verdict: annotate.Verdict | None = None
    codes: list[Choice] = []
    diagnoses: int = 0
    error: str | None = None


def _next(answered: Sequence[bool]) -> int | None:
    return next((i for i, done in enumerate(answered) if not done), None)


def _missing(n: int) -> HTTPException:
    return HTTPException(404, f"no item {n}")


class AuditReview:
    """The silver-label audit's review file (`text2icd eval mimic audit sample`)."""

    mode: Mode = "audit"

    def __init__(self, path: FilePath) -> None:
        if not path.is_file():
            raise ReviewError(f"{path} not found; run `text2icd eval mimic audit sample`")
        self.path = path
        self._lock = threading.Lock()
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        missing = set(audit.COLUMNS) - set(header)
        if missing:
            raise ReviewError(f"{path} has no {', '.join(sorted(missing))} column")

    def _rows(self) -> list[dict[str, str]]:
        return audit.read_review(self.path)

    def progress(self) -> Progress:
        given = [bool(row["verdict"].strip()) for row in self._rows()]
        return Progress(mode=self.mode, total=len(given), answered=sum(given), next=_next(given))

    def item(self, n: int) -> AuditItem:
        return self._item(self._rows(), n)

    def answer(self, n: int, verdict: AuditVerdict) -> AuditItem:
        with self._lock:
            rows = self._rows()
            item = self._item(rows, n)  # 404 before anything is written
            comment = verdict.comment.strip()
            rows[n].update(verdict=verdict.verdict, comment=comment)
            audit.save_review(self.path, rows, list(rows[0]))  # keeps any column added by hand
            return item.model_copy(update={"verdict": verdict.verdict, "comment": comment})

    @staticmethod
    def _item(rows: Sequence[Mapping[str, str]], n: int) -> AuditItem:
        if n >= len(rows):
            raise _missing(n)
        row = rows[n]
        return AuditItem(
            n=n,
            phrase=row["phrase"],
            code=row["code"],
            description=row["description"],
            verdict=row["verdict"].strip().lower(),
            comment=row["comment"],
        )

    def router(self) -> APIRouter:
        api = APIRouter(prefix="/v1/review", tags=["review"])

        @api.get("")
        def progress() -> Progress:
            """How far the review is."""
            return self.progress()

        @api.get("/{n}")
        def item(n: Item) -> AuditItem:
            """A row: the phrase, its silver code, and the verdict so far."""
            return self.item(n)

        @api.put("/{n}")
        def answer(n: Item, verdict: AuditVerdict) -> AuditItem:
            """Give the row's verdict: y (the code is right), n (wrong) or ? (cannot tell)."""
            return self.answer(n, verdict)

        return api


class GoldReview:
    """The gold annotation's candidates (`text2icd eval mimic gold sample`), and the answers
    beside them in `answers.jsonl`, checked against the billable codes of `graph`."""

    mode: Mode = "gold"

    def __init__(self, candidates: FilePath, graph: Graph) -> None:
        if not candidates.is_file():
            raise ReviewError(f"{candidates} not found; run `text2icd eval mimic gold sample`")
        try:
            self.candidates = annotate.read_candidates(candidates)
        except ValueError as exc:
            raise ReviewError(f"{candidates} is not a candidates file: {exc}") from exc
        self.path = candidates.with_name("answers.jsonl")
        self.graph = graph
        self._lock = threading.Lock()

    def _answers(self) -> dict[str, annotate.Answer]:
        return annotate.read_answers(self.path)

    def _billable(self, code: str) -> bool:
        return code in self.graph and self.graph.code(code).billable

    def _choice(self, code: str) -> Choice:
        known = code in self.graph
        return Choice(code=code, description=self.graph.code(code).description if known else "")

    def _candidate(self, n: int) -> annotate.Candidate:
        if n >= len(self.candidates):
            raise _missing(n)
        return self.candidates[n]

    def progress(self) -> Progress:
        answers = self._answers()
        given = [c.id in answers for c in self.candidates]
        return Progress(mode=self.mode, total=len(given), answered=sum(given), next=_next(given))

    def item(self, n: int) -> GoldItem:
        candidate = self._candidate(n)
        return self._item(n, candidate, self._answers().get(candidate.id))

    def read(self, n: int, line: str, comment: str = "") -> annotate.Answer:
        """The answer a line gives, as the terminal reads it; `ReviewError` if there is none."""
        candidate = self._candidate(n)
        parsed = annotate.parse_answer(line, candidate, self._billable)
        if parsed.error:
            raise ReviewError(parsed.error)
        if parsed.answer is None:  # a terminal command: q, b, h, l or c
            raise ReviewError("type candidate numbers or codes; the page has the other commands")
        if parsed.answer.verdict == "coded" and not parsed.answer.codes:
            raise ReviewError("type at least one code")
        typed = comment.strip() or parsed.answer.comment
        return parsed.answer.model_copy(update={"comment": typed})

    def check(self, n: int, line: str) -> Checked:
        try:
            answer = self.read(n, line)
        except ReviewError as exc:
            return Checked(error=str(exc))
        return Checked(
            verdict=answer.verdict,
            codes=[self._choice(code) for code in answer.codes],
            diagnoses=answer.diagnoses or 0,
        )

    def answer(self, n: int, body: GoldLine) -> GoldItem:
        with self._lock:
            answer = self.read(n, body.line, body.comment)
            answers = self._answers()
            answers[answer.id] = answer
            annotate.write_jsonl(self.path, list(answers.values()))
            return self._item(n, self.candidates[n], answer)

    def _item(
        self, n: int, candidate: annotate.Candidate, answer: annotate.Answer | None
    ) -> GoldItem:
        # Only what the person codes from: never the silver code, its band or chapter, or ids.
        return GoldItem(
            n=n,
            phrase=candidate.phrase,
            choices=[Choice(code=code, description=text) for code, text in candidate.choices],
            answer=None
            if answer is None
            else GoldAnswer(
                verdict=answer.verdict,
                codes=[self._choice(code) for code in answer.codes],
                comment=answer.comment,
                diagnoses=answer.diagnoses,
            ),
        )

    def router(self) -> APIRouter:
        api = APIRouter(prefix="/v1/review", tags=["review"])

        @api.get("")
        def progress() -> Progress:
            """How far the annotation is."""
            return self.progress()

        @api.get("/{n}")
        def item(n: Item) -> GoldItem:
            """A phrase, the codes billed for its admission, and the answer so far."""
            return self.item(n)

        @api.post("/{n}/check")
        def check(n: Item, body: GoldLine) -> Checked:
            """Read a line without saving it: the codes it gives and their descriptions."""
            return self.check(n, body.line)

        @api.put("/{n}")
        def answer(n: Item, body: GoldLine) -> GoldItem:
            """Save the phrase's answer: codes (primary first, commas between diagnoses), or
            `s` (not a codable diagnosis) or `?` (unsure)."""
            try:
                return self.answer(n, body)
            except ReviewError as exc:
                raise HTTPException(422, str(exc)) from exc

        return api


Review = AuditReview | GoldReview
