import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.ontology.files import ReleaseFiles

runner = CliRunner()


@pytest.fixture
def data_dir(
    fixture_release: ReleaseFiles, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A data directory whose `fy2027` holds the fixture files, treated as fully fetched."""
    root = tmp_path / "data"
    shutil.copytree(fixture_release.root, root / "raw" / "cms" / "fy2027")
    monkeypatch.setattr(cli, "is_complete", lambda release, raw_dir: True)
    monkeypatch.delenv("TEXT2ICD_RELEASE", raising=False)
    return root


def test_build_graph_writes_the_default_path(data_dir: Path) -> None:
    result = runner.invoke(cli.app, ["build-graph", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    output = data_dir / "processed" / "text2icd-graph-fy2027.sqlite"
    assert output.is_file()
    assert f"fy2027: {output}" in result.output
    assert "tabular_node" in result.output


def test_build_graph_to_an_explicit_output(data_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "elsewhere" / "graph.sqlite"
    result = runner.invoke(
        cli.app, ["build-graph", "-r", "fy2027", "-o", str(output), "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()


def test_build_graph_needs_the_release_files(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["build-graph", "-r", "fy2026", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "fy2026 is not fetched; run `text2icd data fetch -r fy2026`" in result.output


@pytest.mark.parametrize(
    ("release", "message"),
    [("fy2020", "fy2020 is a history release"), ("fy1999", "unknown release 'fy1999'")],
)
def test_build_graph_rejects_releases_it_cannot_build(release: str, message: str) -> None:
    result = runner.invoke(cli.app, ["build-graph", "-r", release])
    assert result.exit_code != 0
    assert message in result.output


@pytest.fixture
def built(data_dir: Path) -> Path:
    """`data_dir` with the fixture graph built as fy2027."""
    result = runner.invoke(cli.app, ["build-graph", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    return data_dir


def test_graph_show(built: Path) -> None:
    result = runner.invoke(
        cli.app, ["graph", "show", "Pneumonia > aspiration", "--data-dir", str(built)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("index: Pneumonia > aspiration\n")
    assert "  subterms (3):\n    due to\n" in result.output


def test_graph_show_a_table_row(built: Path) -> None:
    result = runner.invoke(
        cli.app, ["graph", "show", "1-propanol", "-b", "drug", "--data-dir", str(built)]
    )
    assert result.exit_code == 0, result.output
    assert "cell 2 (Poisoning Accidental (unintentional)): T51.3X1 -> T51.3X1  " in result.output


def test_graph_show_suggests_close_terms(built: Path) -> None:
    result = runner.invoke(
        cli.app, ["graph", "show", "Pneumonia > aspirated", "--data-dir", str(built)]
    )
    assert result.exit_code == 1
    assert "error: no index term at 'Pneumonia > aspirated'" in result.output
    assert "closest: Pneumonia" in result.output


def test_graph_code(built: Path) -> None:
    graph = built / "processed" / "text2icd-graph-fy2027.sqlite"
    result = runner.invoke(cli.app, ["graph", "code", "E1122", "--graph", str(graph)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "E11.22  Type 2 diabetes mellitus with diabetic chronic kidney disease"
    assert "  in E11  Type 2 diabetes mellitus" in lines
    assert "      -> N18.1-N18.6 [unresolved]" in lines
    assert "  notes inherited from E11:" in lines


def test_graph_code_rejects_unknown_codes(built: Path) -> None:
    result = runner.invoke(cli.app, ["graph", "code", "Z99.99", "--data-dir", str(built)])
    assert result.exit_code == 1
    assert "error: Z99.99 is not a code of fy2027" in result.output


def test_graph_commands_need_a_built_graph(data_dir: Path) -> None:
    result = runner.invoke(cli.app, ["graph", "code", "E11.22", "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "run `text2icd build-graph -r fy2027`" in result.output
