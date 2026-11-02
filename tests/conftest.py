import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from text2icd.fetch import COMPLETE_MARKER
from text2icd.graph.build import BuildReport, build_graph
from text2icd.ontology.files import ReleaseFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
MIMIC_DIR = REPO_ROOT / "data" / "raw" / "mimic-iv"
CMS_DIR = REPO_ROOT / "data" / "raw" / "cms"


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def _locate(release_id: str) -> ReleaseFiles:
    files = ReleaseFiles(CMS_DIR / release_id)
    if not any(files.root.glob(f"*/{COMPLETE_MARKER}")):
        pytest.skip(f"CMS release {release_id} not fetched (run `text2icd data fetch`)")
    return files


@pytest.fixture
def cms_release() -> Callable[[str], ReleaseFiles]:
    """Files of a release fetched into data/raw/cms; skips the test when they are absent."""
    return _locate


@pytest.fixture(scope="session")
def release_graph(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], BuildReport]:
    """The graph of a fetched release, built once per session; skips when it is absent."""
    built: dict[str, BuildReport] = {}

    def build(release_id: str) -> BuildReport:
        if release_id not in built:
            output = tmp_path_factory.mktemp(release_id) / "graph.sqlite"
            built[release_id] = build_graph(_locate(release_id), release_id, output)
        return built[release_id]

    return build


@pytest.fixture
def fixture_release(tmp_path: Path, fixtures_dir: Path) -> ReleaseFiles:
    """A tiny fake release, `fy2099`, assembled from the committed CMS fixtures."""
    root = tmp_path / "raw" / "cms" / "fy2099"
    (root / "tables-index").mkdir(parents=True)
    (root / "order").mkdir()
    cms = fixtures_dir / "cms"
    for book in ("tabular", "index", "eindex", "neoplasm", "drug"):
        target = root / "tables-index" / f"icd10cm_{book}_2099.xml"
        target.write_bytes((cms / f"{book}_sample.xml").read_bytes())
    (root / "order" / "icd10cm_order_2099.txt").write_bytes((cms / "order_sample.txt").read_bytes())
    return ReleaseFiles(root)


@pytest.fixture
def fixture_build(fixture_release: ReleaseFiles, tmp_path: Path) -> BuildReport:
    """The graph of the fixture release `fy2099`."""
    return build_graph(fixture_release, "fy2099", tmp_path / "graph" / "fy2099.sqlite")


@pytest.fixture
def fixture_db(fixture_build: BuildReport) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(fixture_build.path)
    yield conn
    conn.close()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if any(MIMIC_DIR.glob("*.csv*")):
        return
    skip_mimic = pytest.mark.skip(reason=f"MIMIC-IV files not found in {MIMIC_DIR}")
    for item in items:
        if "mimic" in item.keywords:
            item.add_marker(skip_mimic)
