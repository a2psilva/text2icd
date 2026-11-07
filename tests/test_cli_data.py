from pathlib import Path

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.cli import app
from text2icd.config import Settings
from text2icd.fetch import FetchError, FileResult
from text2icd.graph.model import GraphError, graph_file
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


def test_stats_requires_a_fetched_release(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["data", "stats", "-r", "fy2027", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "fy2027 is not fetched" in result.output


def test_stats_rejects_history_releases(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["data", "stats", "-r", "fy2016", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "fy2016 is a history release" in result.output


@pytest.mark.parametrize(("ok", "exit_code"), [(True, 0), (False, 1)])
def test_stats_prints_the_report_and_fails_on_problems(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ok: bool, exit_code: int
) -> None:
    class FakeReport:
        def __init__(self) -> None:
            self.ok = ok

        def lines(self) -> list[str]:
            return ["fy2027  (parsed in 1.0 s)", "  billable codes  74,879"]

    monkeypatch.setattr(cli, "is_complete", lambda release, raw_dir, kinds=None: True)
    monkeypatch.setattr(cli, "release_report", lambda files, release_id: FakeReport())
    result = runner.invoke(cli.app, ["data", "stats", "-r", "fy2027", "--data-dir", str(tmp_path)])
    assert result.exit_code == exit_code
    assert "billable codes  74,879" in result.output


def test_data_where(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Task 10.8: the paths in use and where each came from."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["data", "where", "--data-dir", str(tmp_path / "d")])
    assert result.exit_code == 0, result.output
    rows = dict(line.split(None, 1) for line in result.output.splitlines() if line.strip())
    assert rows["data"] == f"{(tmp_path / 'd').resolve()}  (--data-dir)"
    assert rows["cache"].endswith("(the per-user default)")
    assert "missing: `text2icd build-graph -r" in rows["graph"]
    monkeypatch.setenv("TEXT2ICD_DATA_DIR", str(tmp_path / "e"))
    result = runner.invoke(app, ["data", "where"])
    assert "(TEXT2ICD_DATA_DIR)" in result.output


def test_missing_graph_points_to_the_checkout_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkout's graph in ./data, with the data directory set elsewhere (no .env)."""
    monkeypatch.chdir(tmp_path)
    name = "text2icd-graph-fy2027.sqlite"
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / name).write_bytes(b"")
    with pytest.raises(GraphError, match="set TEXT2ICD_DATA_DIR=data"):
        graph_file("fy2027", settings=Settings(data_dir=tmp_path / "elsewhere"))
    with pytest.raises(GraphError) as missing:
        graph_file("fy2027", settings=Settings(data_dir=tmp_path / "data" / "x"))
    assert "does not exist" in str(missing.value)
