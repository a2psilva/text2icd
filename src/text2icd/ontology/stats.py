"""Parse every file of a release, count what it contains and run the consistency checks."""

import time
from dataclasses import dataclass
from pathlib import Path

from text2icd.ontology.billable import generate_billable
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.index import IndexBook, IndexSource, parse_index
from text2icd.ontology.order import billable_codes, parse_order_file
from text2icd.ontology.tabular import parse_tabular
from text2icd.ontology.validate import IndexReference, load_exceptions, unresolved_references


@dataclass(frozen=True, slots=True)
class ReleaseReport:
    release: str
    counts: dict[str, int]
    billable_missing: tuple[str, ...]
    billable_extra: tuple[str, ...]
    unresolved: tuple[IndexReference, ...]
    allow_listed: int
    stale_exceptions: tuple[IndexReference, ...]
    parse_seconds: float

    @property
    def ok(self) -> bool:
        return not (
            self.billable_missing or self.billable_extra or self.unresolved or self.stale_exceptions
        )

    def lines(self) -> list[str]:
        width = max(len(name) for name in self.counts)
        out = [f"{self.release}  (parsed in {self.parse_seconds:.1f} s)"]
        out += [f"  {name:<{width}}  {value:>8,}" for name, value in self.counts.items()]
        match = "yes" if not (self.billable_missing or self.billable_extra) else "NO"
        out.append(f"  generated billable set equals the order file: {match}")
        out += [f"    missing from generation: {code}" for code in self.billable_missing[:20]]
        out += [f"    generated but not in order file: {code}" for code in self.billable_extra[:20]]
        out.append(
            f"  unresolved Index references: {len(self.unresolved)} "
            f"({self.allow_listed} known exceptions)"
        )
        out += [
            f"    {ref.book}: {ref.path} [{ref.field}] {ref.value!r}" for ref in self.unresolved
        ]
        out += [
            f"    allow-listed but no longer found: {ref.path} [{ref.field}]"
            for ref in self.stale_exceptions
        ]
        return out


def release_report(
    files: ReleaseFiles, release_id: str, exceptions: Path | None = None
) -> ReleaseReport:
    start = time.perf_counter()
    tabular = parse_tabular(files.tabular)
    paths: dict[IndexSource, Path] = {
        "index": files.index,
        "eindex": files.eindex,
        "neoplasm": files.neoplasm,
        "drug": files.drug,
    }
    books: dict[IndexSource, IndexBook] = {s: parse_index(p, s) for s, p in paths.items()}
    order = parse_order_file(files.order)
    seconds = time.perf_counter() - start

    official = billable_codes(order)
    generated = generate_billable(tabular)
    diags = list(tabular.diags())
    index = books["index"]
    counts = {
        "tabular chapters": len(tabular.chapters),
        "tabular sections": sum(1 for n in tabular.nodes.values() if n.kind == "section"),
        "tabular codes (diag)": len(diags),
        "tabular placeholders": sum(1 for n in diags if n.placeholder),
        "7th-character definitions": sum(1 for n in tabular.nodes.values() if n.seventh_characters),
        "index main terms": len(index.main_terms),
        "index terms": len(index),
        "index codes": sum(1 for t in index.terms if t.code),
        "index see": sum(1 for t in index.terms if t.see),
        "index see also": sum(1 for t in index.terms if t.see_also),
        "index manifestations": sum(1 for t in index.terms if t.manifestation),
        "index subcat": sum(1 for t in index.terms if t.subcategory),
        "index seecat": sum(1 for t in index.terms if t.see_category),
        "external cause terms": len(books["eindex"]),
        "neoplasm table rows": len(books["neoplasm"]),
        "drug table rows": len(books["drug"]),
        "order file entries": len(order),
        "billable codes": len(official),
    }

    known = [e for e in load_exceptions(exceptions) if release_id in e.releases]
    known_refs = {e.reference for e in known}
    unresolved = unresolved_references(books, tabular, official)
    found = set(unresolved)
    return ReleaseReport(
        release=release_id,
        counts=counts,
        billable_missing=tuple(sorted(official - generated)),
        billable_extra=tuple(sorted(generated - official)),
        unresolved=tuple(ref for ref in unresolved if ref not in known_refs),
        allow_listed=len(known_refs & found),
        stale_exceptions=tuple(sorted(known_refs - found, key=lambda r: (r.book, r.path))),
        parse_seconds=seconds,
    )
