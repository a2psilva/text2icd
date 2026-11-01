from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.ontology.codes import is_code
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.order import billable_codes, parse_order_file

SAMPLE = Path(__file__).parents[1] / "fixtures" / "cms" / "order_sample.txt"
LINE = "00001 A00     0 Cholera" + " " * 54 + "Cholera"


def test_sample_entries() -> None:
    entries = parse_order_file(SAMPLE)
    assert len(entries) == 10
    first = entries[0]
    assert (first.order, first.code, first.billable) == (1, "A00", False)
    assert first.short_description == "Cholera"
    e1122 = next(e for e in entries if e.code == "E11.22")
    assert e1122.billable
    assert e1122.short_description == "Type 2 diabetes mellitus w diabetic chronic kidney disease"
    assert e1122.long_description == "Type 2 diabetes mellitus with diabetic chronic kidney disease"


def test_billable_codes_include_seventh_characters_only() -> None:
    billable = billable_codes(parse_order_file(SAMPLE))
    assert billable == {"A00.0", "A00.1", "A00.9", "E11.22", "T51.3X1A"}
    assert "T51.3X1" not in billable


def write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "icd10cm_order_2099.txt"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
    return path


def test_short_lines_without_long_description_reuse_the_short_one(tmp_path: Path) -> None:
    entries = parse_order_file(write(tmp_path, "00001 A00     0 Cholera"))
    assert entries[0].long_description == "Cholera"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("00001A00      0 Cholera", "fixed-width"),
        ("0000X A00     0 Cholera", "not numeric"),
        ("00001 A00     2 Cholera", "billable flag"),
        ("00001 1234    1 Cholera", "not an ICD-10-CM code"),
        ("00001 A00", "fixed-width"),
    ],
)
def test_malformed_lines_are_rejected(tmp_path: Path, line: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_order_file(write(tmp_path, line))


def test_order_numbers_must_be_consecutive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="order number 3 where 2 was expected"):
        parse_order_file(write(tmp_path, LINE, LINE.replace("00001 A00 ", "00003 A01 ")))


def test_duplicate_codes_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate codes"):
        parse_order_file(write(tmp_path, LINE, LINE.replace("00001", "00002")))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("release_id", "entries", "billable"),
    [
        ("fy2027", 98_403, 74_879),
        ("fy2026-apr", 98_186, 74_719),
        ("fy2026", 98_186, 74_719),
        ("fy2022", 95_528, 72_750),
        ("fy2016", 91_737, 69_823),
    ],
)
def test_release_order_files(
    cms_release: Callable[[str], ReleaseFiles], release_id: str, entries: int, billable: int
) -> None:
    parsed = parse_order_file(cms_release(release_id).order)
    assert len(parsed) == entries
    assert len(billable_codes(parsed)) == billable
    assert all(is_code(e.code) and e.short_description for e in parsed)
