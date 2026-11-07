"""`text2icd batch` (task 10.2): code a file of phrases, streamed, one JSON line per row.

Input, by extension or `--format`:
- `jsonl`: one JSON object per line, with `phrase` and optionally `id`;
- `csv`: a header row, then rows; `--column` names the phrase column, `--id-column` the id;
- `lines`: one phrase per line (blank lines are skipped). `-` reads standard input.

Output: one JSON object per row, in input order, `{"line": N, "id": ..., "result": {...}}`, the
result being a `CodingResult` (docs/schema/coding-result.json). A row that cannot be coded (no
phrase, a phrase over 1,000 characters, a line that is not a JSON object) gets `"error"` instead
of `"result"`, and the run goes on. Rows are coded `BATCH` at a time through
`Resolver.code_many`, so a phrase repeated in a batch is coded once.
"""

import csv
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, TextIO

from text2icd.resolve.resolver import BATCH, MAX_PHRASE, Resolver
from text2icd.resolve.tabular import Episode
from text2icd.text.model import Setting

Format = Literal["jsonl", "csv", "lines"]
FORMATS: tuple[Format, ...] = ("jsonl", "csv", "lines")
_EXTENSIONS: dict[str, Format] = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".txt": "lines",
}


class BatchError(ValueError):
    """The input cannot be read at all: an unknown format, or no phrase column."""


@dataclass(frozen=True, slots=True)
class Row:
    line: int  # where it is in the input, from 1 (the header is line 1 of a CSV file)
    id: str | int | None
    phrase: str | None
    error: str | None = None


@dataclass(slots=True)
class Summary:
    rows: int = 0
    coded: int = 0
    errors: int = 0
    repeats: int = 0  # answered from the cache or an earlier row of the batch
    seconds: float = 0.0

    def line(self) -> str:
        rate = f", {self.rows / self.seconds:,.0f} rows/s" if self.seconds else ""
        return (
            f"{self.rows:,} rows: {self.coded:,} coded ({self.repeats:,} repeats), "
            f"{self.errors:,} errors, in {self.seconds:.1f} s{rate}"
        )


def detect(name: str) -> Format:
    """The format of a file by its extension; standard input is read as lines."""
    for suffix, found in _EXTENSIONS.items():
        if name.lower().endswith(suffix):
            return found
    if name == "-":
        return "lines"
    raise BatchError(f"cannot tell the format of {name}: give --format jsonl, csv or lines")


def read_rows(
    stream: TextIO, fmt: Format, column: str = "phrase", id_column: str = "id"
) -> Iterator[Row]:
    """The rows of `stream`, read lazily."""
    if fmt == "jsonl":
        return _jsonl(stream)
    if fmt == "csv":
        return _csv(stream, column, id_column)
    return (
        Row(number, None, text.rstrip("\r\n"))
        for number, text in enumerate(stream, start=1)
        if text.strip()
    )


def _jsonl(stream: TextIO) -> Iterator[Row]:
    for number, text in enumerate(stream, start=1):
        if not text.strip():
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            yield Row(number, None, None, f"not JSON: {exc.msg}")
            continue
        if not isinstance(data, dict):
            yield Row(number, None, None, "not a JSON object")
            continue
        found = data.get("id")
        row_id = found if isinstance(found, str | int) and not isinstance(found, bool) else None
        phrase = data.get("phrase")
        if not isinstance(phrase, str):
            yield Row(number, row_id, None, 'no "phrase" string')
            continue
        yield Row(number, row_id, phrase)


def _csv(stream: TextIO, column: str, id_column: str) -> Iterator[Row]:
    reader = csv.DictReader(stream)
    if reader.fieldnames is None or column not in reader.fieldnames:
        columns = ", ".join(reader.fieldnames or ()) or "none"
        raise BatchError(f"no {column!r} column in the CSV header (columns: {columns})")
    has_id = id_column in reader.fieldnames
    for record in reader:
        row_id = (record.get(id_column) or None) if has_id else None
        yield Row(reader.line_num, row_id, record.get(column) or "")


def run(
    resolver: Resolver,
    rows: Iterable[Row],
    out: TextIO,
    *,
    setting: Setting = "inpatient",
    episode: Episode | None = None,
    explain: bool = False,
    batch_size: int = BATCH,
) -> Summary:
    """Code the rows and write one JSON line each to `out`, in order, a batch at a time."""
    summary = Summary()
    began = time.perf_counter()
    batch: list[Row] = []
    for row in rows:
        batch.append(_checked(row))
        if len(batch) == batch_size:
            _write(resolver, batch, out, summary, setting, episode, explain)
            batch = []
    if batch:
        _write(resolver, batch, out, summary, setting, episode, explain)
    summary.seconds = time.perf_counter() - began
    return summary


def _checked(row: Row) -> Row:
    if row.error is not None or row.phrase is None:
        return row
    if not row.phrase.strip():
        return Row(row.line, row.id, None, "empty phrase")
    if len(row.phrase) > MAX_PHRASE:
        return Row(row.line, row.id, None, f"over {MAX_PHRASE:,} characters")
    return row


def _write(
    resolver: Resolver,
    batch: list[Row],
    out: TextIO,
    summary: Summary,
    setting: Setting,
    episode: Episode | None,
    explain: bool,
) -> None:
    phrases = [row.phrase for row in batch if row.error is None and row.phrase is not None]
    results = iter(
        resolver.code_many(
            phrases, setting=setting, episode=episode, explain=explain, batch_size=len(batch)
        )
    )
    for row in batch:
        head = f'{{"line": {row.line}, "id": {json.dumps(row.id)}, '
        summary.rows += 1
        if row.error is not None or row.phrase is None:
            summary.errors += 1
            out.write(f'{head}"error": {json.dumps(row.error or "no phrase")}}}\n')
            continue
        result = next(results)
        summary.coded += 1
        summary.repeats += result.cached
        out.write(f'{head}"result": {result.model_dump_json()}}}\n')
    out.flush()
