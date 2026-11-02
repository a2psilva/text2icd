"""The ICD-10-CM codes billed for each MIMIC-IV admission (`hosp/diagnoses_icd`), in `seq_num`
order. MIMIC-IV stores codes without the dot; ICD-9 rows are left out."""

import csv
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from text2icd.ontology.codes import dot, is_code


@dataclass(frozen=True, slots=True)
class BilledCode:
    seq_num: int
    code: str


def admission_codes(
    path: Path, admissions: Collection[str] | None = None
) -> dict[str, list[BilledCode]]:
    """ICD-10 codes per `hadm_id` (all admissions, or only those given), by `seq_num`. A code that
    is not well formed is skipped; a code listed twice keeps its first position."""
    found: dict[str, list[BilledCode]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["icd_version"].strip() != "10":
                continue
            hadm = row["hadm_id"].strip()
            if admissions is not None and hadm not in admissions:
                continue
            code = row["icd_code"].strip()
            if not is_code(code):
                continue
            found.setdefault(hadm, []).append(BilledCode(int(row["seq_num"]), dot(code)))
    for hadm, codes in found.items():
        first: dict[str, BilledCode] = {}
        for billed in sorted(codes, key=lambda c: c.seq_num):
            first.setdefault(billed.code, billed)
        found[hadm] = list(first.values())
    return found
