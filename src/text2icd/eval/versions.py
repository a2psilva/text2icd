"""Version reconciliation: how a code assigned under an older release relates to the evaluated one.

MIMIC-IV codes were assigned under the releases in force from FY2016 to about FY2022; since then
some were deleted or expanded (`N18.3` became `N18.30`, `N18.31` and `N18.32` in FY2021). Each gold
code gets one status against the target release:

- `valid`: billable in the target release, scored by exact match;
- `mapped`: billable in an earlier release and replaced since, scored by mapped match: a prediction
  counts if it is one of `mapped_to`. The replacements come from the CMS conversion tables (new
  code -> the codes it replaced, read backwards and followed across releases), or failing that,
  for a code that became a header, its billable subdivisions in the target release;
- `unmappable`: neither, for example a code that was never billable in any release.

The conversion tables list each new code with its "previous code(s) assignment". Only references
that name codes exactly (`N18.3`, `H35.20-H35.23`) are read backwards; broad ones (`I63.-`,
`Codes in Categories T15-T19`) name codes that still exist and say nothing about a deletion.
"""

import csv
import io
import re
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from text2icd.config import Settings
from text2icd.eval.datasets import CodeStatus, Example, GoldCode
from text2icd.ontology.codes import CodeReference, dot, is_code, parse_reference, undot
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.order import billable_codes, parse_order_file
from text2icd.releases import Registry, Release, load_registry

_HEADER = "current code assignment"
# Separators between the codes of a "previous code(s)" cell: commas, semicolons, "and", "&".
_SEPARATORS = re.compile(r"\s*(?:[,;&]|\band\b)\s*", re.IGNORECASE)
# One code (`N18.3`) or an explicit range of codes (`H35.20-H35.23`); not `I63.-` or `R11-`.
_EXACT = re.compile(
    r"^[A-Z][0-9A-Z]{2}(?:\.[0-9A-Z]{1,4})?(?:-[A-Z][0-9A-Z]{2}(?:\.[0-9A-Z]{1,4})?)?$"
)


@dataclass(frozen=True, slots=True)
class Conversion:
    """One row of a conversion table: a new code and the codes it replaced."""

    code: str
    effective: str
    previous: tuple[CodeReference, ...]
    written: str


def parse_conversion_table(path: Path) -> list[Conversion]:
    """Read a CMS conversion table (CSV or tab-separated text, with a free-text preamble)."""
    text = _read_text(path)
    start = text.casefold().find(_HEADER)
    if start < 0:
        raise ValueError(f"{path}: no '{_HEADER}' header row")
    body = text[start:]
    delimiter = "\t" if "\t" in body.split("\n", 1)[0] else ","
    rows = csv.reader(io.StringIO(body), delimiter=delimiter)
    next(rows)
    conversions = []
    for row in rows:
        if len(row) < 3 or not is_code(row[0]):
            continue
        written = row[2].strip()
        conversions.append(Conversion(dot(row[0]), row[1].strip(), _previous(written), written))
    if not conversions:
        raise ValueError(f"{path}: no conversion rows")
    return conversions


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _previous(written: str) -> tuple[CodeReference, ...]:
    references = []
    for part in _SEPARATORS.split(written.upper()):
        part = part.strip().rstrip(".")
        if _EXACT.fullmatch(part):
            references.append(parse_reference(part))
    return tuple(references)


def conversion_files(raw_dir: Path) -> list[Path]:
    """The machine-readable conversion table of a fetched release (not the spreadsheet)."""
    folder = raw_dir / "conversion-table"
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in (".csv", ".txt"))


@dataclass(frozen=True, slots=True)
class Reconciled:
    code: str
    status: CodeStatus
    mapped_to: tuple[str, ...] = ()
    how: str = ""  # "conversion table", "subdivisions", or why the code is unmappable
    billable_in: tuple[str, ...] = ()  # the releases in which the code itself was billable


@dataclass
class Reconciler:
    """Reconcile codes against one target release, given every release's billable codes (oldest
    first, ending with the target) and the conversion table rows."""

    target: str
    billable: dict[str, frozenset[str]]
    conversions: Sequence[Conversion]
    _successors: dict[str, set[str]] = field(init=False, repr=False)
    _ranges: list[tuple[CodeReference, str]] = field(init=False, repr=False)
    _sorted: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.target not in self.billable:
            raise ValueError(f"no billable codes for the target release {self.target}")
        self._successors = {}
        self._ranges = []
        for row in self.conversions:
            for reference in row.previous:
                if reference.start == reference.end:
                    self._successors.setdefault(reference.start, set()).add(row.code)
                else:
                    self._ranges.append((reference, row.code))
        self._sorted = sorted(undot(code) for code in self.billable[self.target])

    @property
    def releases(self) -> tuple[str, ...]:
        return tuple(self.billable)

    def reconcile(self, code: str) -> Reconciled:
        code = dot(code)
        billable_in = tuple(r for r, codes in self.billable.items() if code in codes)
        if self.target in billable_in:
            return Reconciled(code, "valid", billable_in=billable_in)
        replaced = self._replacements(code)
        if replaced:
            return Reconciled(code, "mapped", replaced, "conversion table", billable_in)
        if not billable_in:
            return Reconciled(code, "unmappable", how="never billable", billable_in=billable_in)
        below = self._subdivisions(code)
        if below:
            return Reconciled(code, "mapped", below, "subdivisions", billable_in)
        return Reconciled(
            code, "unmappable", how="deleted without a replacement", billable_in=billable_in
        )

    def _replacements(self, code: str) -> tuple[str, ...]:
        """Follow the conversion tables forward until codes billable in the target remain."""
        target = self.billable[self.target]
        found: set[str] = set()
        seen = {code}
        pending = [code]
        while pending:
            current = pending.pop()
            for successor in self._next(current):
                if successor in seen:
                    continue
                seen.add(successor)
                if successor in target:
                    found.add(successor)
                else:
                    pending.append(successor)
        return tuple(sorted(found))

    def _next(self, code: str) -> set[str]:
        bare = undot(code)
        successors = set(self._successors.get(bare, ()))
        successors.update(new for reference, new in self._ranges if code in reference)
        return successors

    def _subdivisions(self, code: str) -> tuple[str, ...]:
        bare = undot(code)
        start = bisect_left(self._sorted, bare)
        below = []
        for candidate in self._sorted[start:]:
            if not candidate.startswith(bare):
                break
            if candidate != bare:
                below.append(dot(candidate))
        return tuple(below)

    def spans(self, billable_in: Sequence[str]) -> str:
        """Releases as runs of consecutive ones: `fy2016-fy2020, fy2023`."""
        runs: list[list[str]] = []
        previous = -2
        for position, release in enumerate(self.releases):
            if release not in billable_in:
                continue
            if position == previous + 1:
                runs[-1].append(release)
            else:
                runs.append([release])
            previous = position
        return ", ".join(run[0] if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs)

    def reconcile_all(self, codes: Iterable[str]) -> dict[str, Reconciled]:
        return {result.code: result for result in map(self.reconcile, sorted(set(codes)))}


def reconcile_example(example: Example, reconciler: Reconciler) -> Example:
    """The example against the reconciler's target release: each gold code keeps the code as
    assigned and gets its status (and replacements) in the target."""
    gold = []
    for code in example.gold:
        result = reconciler.reconcile(code.code)
        gold.append(
            GoldCode(
                code=code.code, role=code.role, status=result.status, mapped_to=result.mapped_to
            )
        )
    return example.model_copy(update={"gold": tuple(gold), "release": reconciler.target})


def load_reconciler(
    target: str | None = None, settings: Settings | None = None, registry: Registry | None = None
) -> Reconciler:
    """A reconciler for `target` from the fetched releases up to it: every order file found under
    the data directory and every conversion table. The target's own order file is required."""
    settings = settings or Settings()
    registry = registry or load_registry()
    release = registry.get(target or settings.release)
    history = sorted(
        (r for r in registry.releases.values() if r.effective <= release.effective),
        key=lambda r: r.effective,
    )
    billable: dict[str, frozenset[str]] = {}
    conversions: list[Conversion] = []
    for entry in history:
        codes = _billable(entry, settings, required=entry.id == release.id)
        if codes is not None:
            billable[entry.id] = codes
        for path in conversion_files(settings.cms_raw_dir(entry.id)):
            conversions += parse_conversion_table(path)
    # The target last, so `billable` reads oldest first.
    billable[release.id] = billable.pop(release.id)
    return Reconciler(release.id, billable, conversions)


def _billable(release: Release, settings: Settings, *, required: bool) -> frozenset[str] | None:
    try:
        order = ReleaseFiles(settings.cms_raw_dir(release.id)).order
    except FileNotFoundError:
        if required:
            raise
        return None
    return billable_codes(parse_order_file(order))
