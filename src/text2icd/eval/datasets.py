"""Evaluation datasets: one JSON object per line (JSONL), validated on load.

```json
{"id": "cms-synthetic:000042", "phrase": "aspiration pneumonia due to food",
 "gold": [{"code": "J69.0", "role": "primary"}], "release": "fy2027",
 "source": "cms-synthetic", "split": "dev", "meta": {"generator": "index-path"}}
```

- `gold` lists the full code set: exactly one `primary` code, first, then any `companion` codes
  (manifestation, "use additional code", "code also").
- `status` says how a gold code relates to the dataset's release (task 3.9): `valid` (billable in
  it, the default), `mapped` (deleted or expanded since; `mapped_to` lists the codes that replaced
  it) or `unmappable`.
- `meta` holds anything else: how the example was made, or local identifiers for MIMIC-IV sets.

MIMIC-IV datasets use the same format but are written only under the git-ignored `data/` tree.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from text2icd.config import RELEASE_ID
from text2icd.ontology.codes import dot

Role = Literal["primary", "companion"]
CodeStatus = Literal["valid", "mapped", "unmappable"]
Split = Literal["dev", "test"]
SPLITS: tuple[Split, ...] = ("dev", "test")


class DatasetError(ValueError):
    """A dataset file that cannot be read or does not follow the format."""


class GoldCode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    role: Role
    status: CodeStatus = "valid"
    mapped_to: tuple[str, ...] = ()

    @field_validator("code")
    @classmethod
    def _dotted(cls, value: str) -> str:
        return dot(value)

    @field_validator("mapped_to")
    @classmethod
    def _dotted_all(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dot(code) for code in value)

    @model_validator(mode="after")
    def _mapping_matches_status(self) -> "GoldCode":
        if (self.status == "mapped") != bool(self.mapped_to):
            raise ValueError("mapped_to is required for status 'mapped' and only allowed there")
        return self


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    phrase: str = Field(min_length=1)
    gold: tuple[GoldCode, ...] = Field(min_length=1)
    release: str
    source: str = Field(min_length=1)
    split: Split
    meta: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("phrase")
    @classmethod
    def _one_line(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("phrase is empty")
        return value

    @field_validator("release")
    @classmethod
    def _release_id(cls, value: str) -> str:
        if not RELEASE_ID.fullmatch(value):
            raise ValueError(f"release must look like 'fy2027' or 'fy2026-apr', got {value!r}")
        return value

    @model_validator(mode="after")
    def _one_primary_first(self) -> "Example":
        roles = [gold.role for gold in self.gold]
        if roles[0] != "primary" or roles.count("primary") != 1:
            raise ValueError("gold needs exactly one primary code, listed first")
        codes = [gold.code for gold in self.gold]
        if len(set(codes)) != len(codes):
            raise ValueError("gold lists a code twice")
        return self

    @property
    def primary(self) -> GoldCode:
        return self.gold[0]

    @property
    def companions(self) -> tuple[GoldCode, ...]:
        return self.gold[1:]


def read_dataset(path: Path) -> list[Example]:
    """Read and validate every line; the error names the first bad line."""
    examples = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                examples.append(Example.model_validate_json(line))
            except ValidationError as exc:
                first = exc.errors()[0]
                where = ".".join(str(part) for part in first["loc"]) or "line"
                raise DatasetError(f"{path}:{number}: {where}: {first['msg']}") from None
    return examples


def write_dataset(path: Path, examples: Iterable[Example]) -> int:
    """Write examples as JSONL, leaving out default values; returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for example in examples:
            fh.write(to_line(example) + "\n")
            count += 1
    return count


def to_line(example: Example) -> str:
    data = example.model_dump(mode="json", exclude_defaults=True)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def manifest_path(path: Path) -> Path:
    """`set.jsonl` -> `set.manifest.json`: how the dataset was made."""
    return path.with_name(path.name.removesuffix(".jsonl") + ".manifest.json")


def write_manifest(path: Path, examples: Sequence[Example], **made_with: JsonValue) -> Path:
    """Write the dataset's manifest: its hash and counts, and how it was made."""
    summary = summarize(examples)
    manifest: dict[str, JsonValue] = {
        "examples": summary.examples,
        "sha256": summary.sha256,
        "release": summary.release,
        "splits": dict(summary.splits),
        **made_with,
    }
    target = manifest_path(path)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def read_manifest(path: Path) -> dict[str, JsonValue]:
    """The manifest next to a dataset, or {} if there is none."""
    target = manifest_path(path)
    if not target.is_file():
        return {}
    data = json.loads(target.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def dataset_hash(examples: Iterable[Example]) -> str:
    """SHA-256 over the examples in id order, with every field: independent of line order and of
    how the file was formatted, so it identifies the content (recorded with each result)."""
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda e: e.id):
        data = example.model_dump(mode="json")
        digest.update(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def check_dataset(
    examples: Sequence[Example], billable: Callable[[str], bool] | None = None
) -> list[str]:
    """Problems that a single line cannot show: repeated ids, mixed releases, and (when the
    release's billable codes are given) gold codes whose status does not hold."""
    problems = [f"id {key!r} appears {n} times" for key, n in _repeated(e.id for e in examples)]
    releases = sorted({e.release for e in examples})
    if len(releases) > 1:
        problems.append(f"more than one release: {', '.join(releases)}")
    if billable is None:
        return problems
    for example in examples:
        for gold in example.gold:
            problem = _status_problem(gold, billable)
            if problem:
                problems.append(f"{example.id}: {gold.code} {problem}")
    return problems


def _status_problem(gold: GoldCode, billable: Callable[[str], bool]) -> str | None:
    if gold.status == "valid":
        return None if billable(gold.code) else "is marked valid but is not billable"
    if billable(gold.code):
        return f"is marked {gold.status} but is billable"
    missing = [code for code in gold.mapped_to if not billable(code)]
    return f"maps to codes that are not billable: {', '.join(missing)}" if missing else None


def _repeated(values: Iterable[str]) -> list[tuple[str, int]]:
    return [(value, n) for value, n in Counter(values).items() if n > 1]


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    examples: int
    release: str | None
    splits: dict[str, int]
    sources: dict[str, int]
    statuses: dict[str, int]
    companions: int
    sha256: str

    def lines(self) -> list[str]:
        def counts(values: dict[str, int]) -> str:
            return ", ".join(f"{key} {n:,}" for key, n in values.items()) or "none"

        return [
            f"  {self.examples:,} examples, release {self.release or 'none'}",
            f"  splits: {counts(self.splits)}",
            f"  sources: {counts(self.sources)}",
            f"  gold codes: {counts(self.statuses)}; {self.companions:,} companion codes",
            f"  sha256 {self.sha256}",
        ]


def summarize(examples: Sequence[Example]) -> DatasetSummary:
    releases = {e.release for e in examples}
    statuses = Counter(gold.status for e in examples for gold in e.gold)
    return DatasetSummary(
        examples=len(examples),
        release=releases.pop() if len(releases) == 1 else None,
        splits={split: n for split in SPLITS if (n := sum(e.split == split for e in examples))},
        sources=dict(sorted(Counter(e.source for e in examples).items())),
        statuses={status: statuses[status] for status in ("valid", "mapped", "unmappable")},
        companions=sum(len(e.companions) for e in examples),
        sha256=dataset_hash(examples),
    )
