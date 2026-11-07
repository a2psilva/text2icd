"""Task 10.7: scripts/release.py, which the release workflow runs before building."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release.py"

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- Something new.

## [0.2.0rc1] - 2026-11-08

The web UI.

### Added

- `text2icd serve`.

## [0.1.0] - 2026-11-03

### Added

- The coder.
"""


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def project(tmp_path: Path, version: str, changelog: str = CHANGELOG) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "text2icd"\nversion = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return tmp_path


def test_a_release(tmp_path: Path) -> None:
    root = project(tmp_path, "0.2.0rc1")
    done = run(root, "v0.2.0rc1", "--notes", str(tmp_path / "notes.md"))
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["version=0.2.0rc1", "prerelease=true"]
    notes = (tmp_path / "notes.md").read_text("utf-8")
    assert notes.startswith("The web UI.\n\n### Added\n\n- `text2icd serve`.\n\n## Install")
    assert "The coder" not in notes  # the next section is left out
    assert "releases/download/v0.2.0rc1/text2icd-0.2.0rc1-py3-none-any.whl" in notes
    assert "gh attestation verify" in notes


def test_a_dry_run(tmp_path: Path) -> None:
    done = run(project(tmp_path, "0.1.0"))
    assert done.stdout.splitlines() == ["version=0.1.0", "prerelease=false"]


@pytest.mark.parametrize(
    ("version", "tag", "error"),
    [
        ("0.2.0rc1", "v0.2.0", "is not v0.2.0rc1"),
        ("0.2.0rc1", "0.2.0rc1", "is not v0.2.0rc1"),
        ("0.3.0", "v0.3.0", "no '## [0.3.0] - YYYY-MM-DD' section"),
        ("0.2.0-rc1", "v0.2.0-rc1", "not MAJOR.MINOR.PATCH"),
        ("1.0", "v1.0", "not MAJOR.MINOR.PATCH"),
    ],
)
def test_disagreements(tmp_path: Path, version: str, tag: str, error: str) -> None:
    done = run(project(tmp_path, version), tag)
    assert done.returncode == 1
    assert error in done.stderr
    assert done.stdout == ""


def test_an_empty_section(tmp_path: Path) -> None:
    root = project(tmp_path, "0.2.0", "## [0.2.0] - 2026-11-08\n\n## [0.1.0] - 2026-11-03\n- x\n")
    done = run(root, "v0.2.0")
    assert "is empty" in done.stderr


def test_the_repository_agrees_with_itself() -> None:
    """pyproject.toml's version always has its CHANGELOG.md section."""
    done = run(SCRIPT.parent.parent)
    assert done.returncode == 0, done.stderr
