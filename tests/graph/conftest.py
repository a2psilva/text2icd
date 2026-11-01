import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from text2icd.graph.build import BuildReport, build_graph
from text2icd.ontology.files import ReleaseFiles


@pytest.fixture
def fixture_build(fixture_release: ReleaseFiles, tmp_path: Path) -> BuildReport:
    """The graph of the fixture release `fy2099`."""
    return build_graph(fixture_release, "fy2099", tmp_path / "graph" / "fy2099.sqlite")


@pytest.fixture
def fixture_db(fixture_build: BuildReport) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(fixture_build.path)
    yield conn
    conn.close()
