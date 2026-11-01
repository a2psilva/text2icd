import subprocess
import sys
from importlib.metadata import version

from typer.testing import CliRunner

import text2icd
from text2icd.cli import app

runner = CliRunner()


def test_package_version_matches_distribution_metadata() -> None:
    assert text2icd.__version__ == version("text2icd")


def test_cli_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == text2icd.__version__


def test_cli_without_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_module_entry_point_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "text2icd", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == text2icd.__version__
