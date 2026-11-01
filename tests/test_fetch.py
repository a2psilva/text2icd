import hashlib
import io
import stat
import urllib.error
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import IO

import pytest

from text2icd.fetch import FetchError, fetch_release, is_complete
from text2icd.releases import FileKind, Release, ReleaseFile

ZIP_URL = "https://www.cms.gov/files/zip/tables.zip"
PDF_URL = "https://www.cms.gov/files/document/guidelines-2026.pdf"


def make_zip(members: dict[str, bytes], symlink: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
        if symlink is not None:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "target")
    return buffer.getvalue()


def release_file(kind: FileKind, url: str, content: bytes) -> ReleaseFile:
    return ReleaseFile(kind, url, hashlib.sha256(content).hexdigest(), len(content))


def release(*files: ReleaseFile) -> Release:
    return Release("fy2026", 2026, date(2025, 10, 1), "full", files)


class FakeOpener:
    def __init__(self, content: dict[str, bytes], failures: list[Exception] | None = None) -> None:
        self.content = content
        self.failures = failures or []
        self.calls: list[str] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> IO[bytes]:
        self.calls.append(request.full_url)
        assert request.get_header("User-agent", "").startswith("text2icd/")
        if self.failures:
            raise self.failures.pop(0)
        return io.BytesIO(self.content[request.full_url])


def no_sleep(_: float) -> None:
    pass


TABLES = make_zip(
    {
        "Table and Index/icd10cm_tabular_2026.xml": b"<ICD10CM.tabular/>",
        "Table and Index/icd10cm_index_2026.xml": b"<ICD10CM.index/>",
    }
)


def test_fetch_downloads_verifies_and_unpacks(tmp_path: Path) -> None:
    pdf = b"%PDF-1.7 guidelines"
    files = (
        release_file("tables-index", ZIP_URL, TABLES),
        release_file("guidelines", PDF_URL, pdf),
    )
    opener = FakeOpener({ZIP_URL: TABLES, PDF_URL: pdf})

    results = fetch_release(release(*files), tmp_path, opener=opener, sleep=no_sleep)

    assert [(r.kind, r.downloaded, r.extracted) for r in results] == [
        ("tables-index", True, 2),
        ("guidelines", True, 1),
    ]
    assert (tmp_path / "archives" / "tables-index.zip").read_bytes() == TABLES
    assert (tmp_path / "tables-index" / "icd10cm_tabular_2026.xml").read_bytes() == (
        b"<ICD10CM.tabular/>"
    )
    assert (tmp_path / "guidelines" / "guidelines-2026.pdf").read_bytes() == pdf
    assert is_complete(release(*files), tmp_path)
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_is_idempotent(tmp_path: Path) -> None:
    files = (release_file("tables-index", ZIP_URL, TABLES),)
    fetch_release(release(*files), tmp_path, opener=FakeOpener({ZIP_URL: TABLES}), sleep=no_sleep)
    opener = FakeOpener({ZIP_URL: TABLES})

    results = fetch_release(release(*files), tmp_path, opener=opener, sleep=no_sleep)

    assert opener.calls == []
    assert [(r.downloaded, r.extracted) for r in results] == [(False, 0)]


def test_existing_archive_is_used_without_downloading(tmp_path: Path) -> None:
    (tmp_path / "archives").mkdir()
    (tmp_path / "archives" / "tables-index.zip").write_bytes(TABLES)
    opener = FakeOpener({})

    results = fetch_release(
        release(release_file("tables-index", ZIP_URL, TABLES)), tmp_path, opener=opener
    )

    assert opener.calls == []
    assert results[0].downloaded is False
    assert results[0].extracted == 2


def test_hash_mismatch_is_rejected_and_nothing_is_kept(tmp_path: Path) -> None:
    files = (release_file("tables-index", ZIP_URL, TABLES),)
    opener = FakeOpener({ZIP_URL: TABLES + b"tampered"})

    with pytest.raises(FetchError, match="does not match the registry"):
        fetch_release(release(*files), tmp_path, opener=opener, sleep=no_sleep)

    assert not (tmp_path / "archives" / "tables-index.zip").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_existing_archive_with_wrong_hash_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "archives").mkdir()
    archive = tmp_path / "archives" / "tables-index.zip"
    archive.write_bytes(b"something else")

    with pytest.raises(FetchError, match="refusing to overwrite"):
        fetch_release(
            release(release_file("tables-index", ZIP_URL, TABLES)),
            tmp_path,
            opener=FakeOpener({ZIP_URL: TABLES}),
        )

    assert archive.read_bytes() == b"something else"


def test_extracted_file_with_different_content_is_not_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "tables-index" / "icd10cm_tabular_2026.xml"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"edited by hand")

    with pytest.raises(FetchError, match="exists with different content"):
        fetch_release(
            release(release_file("tables-index", ZIP_URL, TABLES)),
            tmp_path,
            opener=FakeOpener({ZIP_URL: TABLES}),
        )

    assert target.read_bytes() == b"edited by hand"


@pytest.mark.parametrize(
    "member", ["../escape.txt", "/etc/escape.txt", "C:/escape.txt", "docs/../../escape.txt"]
)
def test_unsafe_zip_paths_are_rejected(tmp_path: Path, member: str) -> None:
    evil = make_zip({member: b"x"})
    with pytest.raises(FetchError, match="unsafe path"):
        fetch_release(
            release(release_file("tables-index", ZIP_URL, evil)),
            tmp_path / "raw",
            opener=FakeOpener({ZIP_URL: evil}),
        )
    assert not (tmp_path / "escape.txt").exists()


def test_symlinks_in_zip_are_rejected(tmp_path: Path) -> None:
    evil = make_zip({"a.txt": b"x"}, symlink="link")
    with pytest.raises(FetchError, match="symbolic link"):
        fetch_release(
            release(release_file("tables-index", ZIP_URL, evil)),
            tmp_path,
            opener=FakeOpener({ZIP_URL: evil}),
        )


def test_zip_without_single_top_folder_is_kept_as_is(tmp_path: Path) -> None:
    flat = make_zip({"a.txt": b"a", "sub/b.txt": b"b"})
    fetch_release(
        release(release_file("order", ZIP_URL, flat)),
        tmp_path,
        opener=FakeOpener({ZIP_URL: flat}),
    )
    assert (tmp_path / "order" / "a.txt").read_bytes() == b"a"
    assert (tmp_path / "order" / "sub" / "b.txt").read_bytes() == b"b"


def test_transient_errors_are_retried(tmp_path: Path) -> None:
    sleeps: list[float] = []
    opener = FakeOpener(
        {ZIP_URL: TABLES},
        failures=[
            urllib.error.URLError("connection reset"),
            urllib.error.HTTPError(ZIP_URL, 503, "unavailable", {}, None),  # type: ignore[arg-type]
        ],
    )

    fetch_release(
        release(release_file("tables-index", ZIP_URL, TABLES)),
        tmp_path,
        opener=opener,
        sleep=sleeps.append,
    )

    assert len(opener.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    opener = FakeOpener(
        {},
        failures=[urllib.error.HTTPError(ZIP_URL, 404, "not found", {}, None)],  # type: ignore[arg-type]
    )
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch_release(
            release(release_file("tables-index", ZIP_URL, TABLES)),
            tmp_path,
            opener=opener,
            sleep=no_sleep,
        )
    assert len(opener.calls) == 1


def test_gives_up_after_the_retry_budget(tmp_path: Path) -> None:
    opener = FakeOpener({}, failures=[TimeoutError("slow")] * 3)
    with pytest.raises(FetchError, match="could not download"):
        fetch_release(
            release(release_file("tables-index", ZIP_URL, TABLES)),
            tmp_path,
            opener=opener,
            sleep=no_sleep,
            retries=2,
        )
    assert len(opener.calls) == 3


def test_is_complete_is_false_before_fetching(tmp_path: Path) -> None:
    assert not is_complete(release(release_file("tables-index", ZIP_URL, TABLES)), tmp_path)
