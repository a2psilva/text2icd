from pathlib import Path

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.fetch import FetchError, FileResult
from text2icd.releases import Release

runner = CliRunner()


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path]]:
    calls: list[tuple[str, Path]] = []

    def fake_fetch(release: Release, raw_dir: Path) -> list[FileResult]:
        calls.append((release.id, raw_dir))
        return [FileResult(f.kind, False, 0) for f in release.files]

    monkeypatch.setattr(cli, "fetch_release", fake_fetch)
    monkeypatch.delenv("TEXT2ICD_RELEASE", raising=False)
    return calls


def test_fetch_defaults_to_the_newest_release(
    fetched: list[tuple[str, Path]], tmp_path: Path
) -> None:
    result = runner.invoke(cli.app, ["data", "fetch", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert fetched == [("fy2027", tmp_path / "raw" / "cms" / "fy2027")]
    assert "tables-index      verified, 0 new file(s) unpacked" in result.output


def test_fetch_respects_the_release_setting(
    fetched: list[tuple[str, Path]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TEXT2ICD_RELEASE", "fy2026-apr")
    result = runner.invoke(cli.app, ["data", "fetch", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert [rid for rid, _ in fetched] == ["fy2026-apr"]


def test_fetch_several_releases_and_history(
    fetched: list[tuple[str, Path]], tmp_path: Path
) -> None:
    result = runner.invoke(
        cli.app,
        ["data", "fetch", "-r", "fy2026", "-r", "fy2016", "--history", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    ids = [rid for rid, _ in fetched]
    assert ids[:2] == ["fy2026", "fy2016"]
    assert sorted(ids[2:]) == sorted(f"fy{y}" for y in range(2017, 2026))


def test_fetch_rejects_unknown_release(fetched: list[tuple[str, Path]]) -> None:
    result = runner.invoke(cli.app, ["data", "fetch", "-r", "fy1999"])
    assert result.exit_code != 0
    assert "unknown release 'fy1999'" in result.output
    assert fetched == []


def test_fetch_error_exits_non_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def failing(release: Release, raw_dir: Path) -> list[FileResult]:
        raise FetchError("HTTP 404 for https://www.cms.gov/files/zip/x.zip")

    monkeypatch.setattr(cli, "fetch_release", failing)
    result = runner.invoke(cli.app, ["data", "fetch", "-r", "fy2026", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "error: HTTP 404" in result.output


def test_releases_lists_status(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["data", "releases", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("fy2027      full     effective 2026-10-01  missing  (default)")
    assert len(lines) == 13
