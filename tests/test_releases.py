from datetime import date
from pathlib import Path

import pytest

from text2icd.releases import FILE_KINDS, load_registry

SHA = "a" * 64


def test_shipped_registry_is_valid() -> None:
    registry = load_registry()
    assert registry.default == "fy2027"
    assert registry.get().id == "fy2027"
    assert {r.id for r in registry.by_scope("full")} == {"fy2027", "fy2026-apr", "fy2026"}
    assert [r.id for r in registry.by_scope("history")] == [f"fy{y}" for y in range(2025, 2015, -1)]


def test_shipped_registry_dates_follow_the_fiscal_year() -> None:
    registry = load_registry()
    assert registry.get("fy2027").effective == date(2026, 10, 1)
    assert registry.get("fy2026-apr").effective == date(2026, 4, 1)
    assert registry.get("fy2016").effective == date(2015, 10, 1)


def test_full_releases_have_the_files_needed_to_build() -> None:
    for release in load_registry().by_scope("full"):
        assert release.file("tables-index").archive_name == "tables-index.zip"
        assert release.file("order").archive_name == "order.zip"


def test_history_releases_have_code_lists() -> None:
    for release in load_registry().by_scope("history"):
        assert release.file("order").kind == "order"
        assert {f.kind for f in release.files} <= {"order", "conversion-table"}


def test_guidelines_are_pdfs() -> None:
    assert load_registry().get("fy2026").file("guidelines").archive_name == "guidelines.pdf"


def test_unknown_release_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="unknown release 'fy1999'; known releases: fy2027"):
        load_registry().get("fy1999")


def test_missing_file_kind_raises() -> None:
    with pytest.raises(KeyError, match="fy2016 has no 'tables-index' file"):
        load_registry().get("fy2016").file("tables-index")


def _registry(tmp_path: Path, body: str, default: str = "fy2026") -> Path:
    path = tmp_path / "releases.toml"
    path.write_text(f'schema = 1\ndefault = "{default}"\n{body}')
    return path


def _release(rid: str = "fy2026", scope: str = "full", files: str | None = None) -> str:
    year = int(rid[2:6])
    if files is None:
        files = "".join(_file(kind) for kind in ("tables-index", "order"))
    return (
        f'\n[[release]]\nid = "{rid}"\nfiscal_year = {year}\n'
        f'effective = {year - 1}-10-01\nscope = "{scope}"\n{files}'
    )


def _file(kind: str, url: str = "https://www.cms.gov/files/zip/x.zip", sha: str = SHA) -> str:
    return f'\n[[release.file]]\nkind = "{kind}"\nurl = "{url}"\nsha256 = "{sha}"\nsize = 10\n'


def test_custom_registry_loads(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path, _release()))
    assert registry.get().files[0].sha256 == SHA


@pytest.mark.parametrize(
    ("body", "default", "message"),
    [
        (_release(), "fy2027", "default release 'fy2027' is not defined"),
        (_release(scope="history"), "fy2026", "must have scope 'full'"),
        (_release() + _release(), "fy2026", "duplicate release id"),
        (_release().replace('id = "fy2026"', 'id = "fy26"'), "fy26", "invalid release id"),
        (_release(files=_file("order")), "fy2026", "a full release needs"),
        (_release(files=_file("order") + _file("order")), "fy2026", "only once"),
        (_release(files=_file("tables-index") + _file("readme")), "fy2026", "unknown file kind"),
        (_release(files=_file("order", sha="ABC")), "fy2026", "sha256 must be 64"),
        (
            _release(files=_file("order", url="http://example.com/x.zip")),
            "fy2026",
            "https://www.cms.gov/",
        ),
    ],
)
def test_invalid_registries_are_rejected(
    tmp_path: Path, body: str, default: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_registry(_registry(tmp_path, body, default))


def test_fiscal_year_must_match_the_id(tmp_path: Path) -> None:
    body = _release().replace("fiscal_year = 2026", "fiscal_year = 2025")
    with pytest.raises(ValueError, match="does not match the id"):
        load_registry(_registry(tmp_path, body))


def test_every_file_kind_is_known() -> None:
    kinds = {f.kind for r in load_registry().releases.values() for f in r.files}
    assert kinds <= set(FILE_KINDS)
