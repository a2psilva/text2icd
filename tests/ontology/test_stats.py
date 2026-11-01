from collections.abc import Callable

import pytest

from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.stats import ReleaseReport, release_report
from text2icd.ontology.validate import IndexReference


def report(**changes: object) -> ReleaseReport:
    base: dict[str, object] = {
        "release": "fy2099",
        "counts": {"tabular codes (diag)": 3, "billable codes": 2},
        "billable_missing": (),
        "billable_extra": (),
        "unresolved": (),
        "allow_listed": 0,
        "stale_exceptions": (),
        "parse_seconds": 0.5,
    }
    base.update(changes)
    return ReleaseReport(**base)  # type: ignore[arg-type]


def test_clean_report() -> None:
    clean = report()
    assert clean.ok
    text = "\n".join(clean.lines())
    assert "tabular codes (diag)         3" in text
    assert "generated billable set equals the order file: yes" in text
    assert "unresolved Index references: 0 (0 known exceptions)" in text


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"billable_missing": ("A00.0",)}, "missing from generation: A00.0"),
        ({"billable_extra": ("S06.1X7D",)}, "generated but not in order file: S06.1X7D"),
        (
            {"unresolved": (IndexReference("index", "Test > x", "code", "Z99.99"),)},
            "index: Test > x [code] 'Z99.99'",
        ),
        (
            {"stale_exceptions": (IndexReference("index", "Old > y", "code", "Z99.98"),)},
            "allow-listed but no longer found: Old > y [code]",
        ),
    ],
)
def test_problems_fail_the_report(changes: dict[str, object], expected: str) -> None:
    failing = report(**changes)
    assert not failing.ok
    assert expected in "\n".join(failing.lines())


def test_fixture_release_report(fixture_release: ReleaseFiles) -> None:
    result = release_report(fixture_release, "fy2099")
    assert result.counts["tabular codes (diag)"] == 15
    assert result.counts["index main terms"] == 6
    assert result.counts["order file entries"] == 10
    assert result.counts["billable codes"] == 5
    # The order sample holds A00 codes that the trimmed Tabular sample does not.
    assert result.billable_missing == ("A00.0", "A00.1", "A00.9")
    assert "E11.22" not in result.billable_extra
    unresolved = {(ref.book, ref.path) for ref in result.unresolved}
    assert ("drug", "1-propanol") not in unresolved
    assert ("index", "Pneumonia > aspiration") in unresolved
    assert not result.ok


FY2026_APRIL = {
    "tabular chapters": 22,
    "tabular sections": 297,
    "tabular codes (diag)": 46_881,
    "tabular placeholders": 246,
    "7th-character definitions": 508,
    "index main terms": 8_122,
    "index terms": 81_515,
    "index codes": 63_142,
    "index see": 12_141,
    "index see also": 2_063,
    "index manifestations": 769,
    "index subcat": 130,
    "index seecat": 15,
    "external cause terms": 8_090,
    "neoplasm table rows": 1_940,
    "drug table rows": 7_430,
    "order file entries": 98_186,
    "billable codes": 74_719,
}


@pytest.mark.slow
def test_fy2026_april_report(cms_release: Callable[[str], ReleaseFiles]) -> None:
    result = release_report(cms_release("fy2026-apr"), "fy2026-apr")
    assert result.counts == FY2026_APRIL
    assert result.ok, "\n".join(result.lines())
    assert result.allow_listed == 9
    assert result.parse_seconds < 30


@pytest.mark.slow
@pytest.mark.parametrize("release_id", ["fy2026", "fy2027"])
def test_full_releases_pass_the_checks(
    cms_release: Callable[[str], ReleaseFiles], release_id: str
) -> None:
    result = release_report(cms_release(release_id), release_id)
    assert result.ok, "\n".join(result.lines())
    assert result.parse_seconds < 30
