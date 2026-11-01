"""Parser for the CMS code order file (`icd10cm_order_<year>.txt`, "Code Descriptions in Tabular
Order").

Fixed-width ASCII, one entry per line, in Tabular List order:

| Columns | Content |
| ------- | ------- |
| 1-5     | order number, zero-padded |
| 7-13    | code without the dot, left-justified |
| 15      | `1` if the code is valid for reporting (billable), `0` for a header |
| 17-76   | short description |
| 78-     | long description |

Columns 6, 14, 16 and 77 are blank.
"""

from dataclasses import dataclass
from pathlib import Path

from text2icd.ontology.codes import dot, is_code

_BLANK_COLUMNS = (5, 13, 15)


@dataclass(frozen=True, slots=True)
class OrderEntry:
    order: int
    code: str
    billable: bool
    short_description: str
    long_description: str


def parse_order_file(path: Path) -> tuple[OrderEntry, ...]:
    entries = []
    with path.open(encoding="ascii", newline="") as fh:
        for number, raw in enumerate(fh, start=1):
            line = raw.rstrip("\r\n")
            if line:
                entries.append(_parse_line(line, number, path))
    for expected, entry in enumerate(entries, start=1):
        if entry.order != expected:
            raise ValueError(f"{path}: order number {entry.order} where {expected} was expected")
    codes = [entry.code for entry in entries]
    if len(set(codes)) != len(codes):
        raise ValueError(f"{path}: duplicate codes")
    return tuple(entries)


def billable_codes(entries: tuple[OrderEntry, ...]) -> frozenset[str]:
    return frozenset(entry.code for entry in entries if entry.billable)


def _parse_line(line: str, number: int, path: Path) -> OrderEntry:
    where = f"{path.name} line {number}"
    if len(line) < 17 or any(line[i] != " " for i in _BLANK_COLUMNS):
        raise ValueError(f"{where}: not in the fixed-width order file layout")
    order, code, flag = line[0:5], line[6:13].strip(), line[14]
    if not order.isdigit():
        raise ValueError(f"{where}: order number {order!r} is not numeric")
    if flag not in "01":
        raise ValueError(f"{where}: billable flag must be 0 or 1, got {flag!r}")
    if not is_code(code):
        raise ValueError(f"{where}: {code!r} is not an ICD-10-CM code")
    short = line[16:76].strip()
    long = line[77:].strip() if len(line) > 77 else ""
    return OrderEntry(
        order=int(order),
        code=dot(code),
        billable=flag == "1",
        short_description=short,
        long_description=long or short,
    )
