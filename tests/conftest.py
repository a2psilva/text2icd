import os
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from text2icd.config import Settings
from text2icd.fetch import COMPLETE_MARKER
from text2icd.graph.build import BuildReport, build_graph
from text2icd.graph.model import Graph, load
from text2icd.ontology.files import ReleaseFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
MIMIC_DIR = REPO_ROOT / "data" / "raw" / "mimic-iv"
CMS_DIR = REPO_ROOT / "data" / "raw" / "cms"
# Settings that tests must not inherit from the developer's shell: an API key could make a test
# call a paid endpoint, and a provider or data dir would change what the code under test does.
_ISOLATED_PREFIXES = ("TEXT2ICD_", "ANTHROPIC_")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test sees the default settings: no `.env` file and no TEXT2ICD_* or ANTHROPIC_*
    variables. A test that needs one sets it."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in list(os.environ):
        if name.startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(name)


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


# Codes the fixture's Index names but its Tabular List lacks; added so the Index terms resolve.
ADDED_CODES = {
    "J69.0": "Pneumonitis due to inhalation of food and vomit",
    "J12.9": "Viral pneumonia, unspecified",
    "J15.9": "Unspecified bacterial pneumonia",
    "J85.1": "Abscess of lung with pneumonia",
    "E85.4": "Organ-limited amyloidosis",
    "I43": "Cardiomyopathy in diseases classified elsewhere",
    "N11.9": "Chronic tubulo-interstitial nephritis, unspecified",
    "N13.6": "Pyonephrosis",
    "A18.11": "Tuberculosis of kidney and ureter",
    "C76.2": "Malignant neoplasm of abdomen",
    "D36.7": "Benign neoplasm of other specified sites",
}


@pytest.fixture
def graph_file(fixture_build: BuildReport, tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.sqlite"
    shutil.copy(fixture_build.path, path)
    with sqlite3.connect(path) as conn:
        for ordinal, (code, description) in enumerate(ADDED_CODES.items(), start=1000):
            conn.execute(
                "INSERT INTO tabular_node VALUES (?, 'diag', '4', ?, 1, ?, 1, 0, NULL, NULL)",
                (code, ordinal, description),
            )
        conn.execute(
            f"UPDATE index_pointer SET target = code, status = 'ok' "
            f"WHERE code IN ({', '.join('?' * len(ADDED_CODES))})",
            list(ADDED_CODES),
        )
        conn.execute(
            "INSERT INTO tabular_term VALUES (1000, 'J69.0', 'inclusion', "
            "'Pneumonitis due to inhalation of food (regurgitated)')"
        )
    conn.close()
    return path


@pytest.fixture
def graph(graph_file: Path) -> Iterator[Graph]:
    with load(path=graph_file) as loaded:
        yield loaded


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if any(MIMIC_DIR.glob("*.csv*")):
        return
    skip_mimic = pytest.mark.skip(reason=f"MIMIC-IV files not found in {MIMIC_DIR}")
    for item in items:
        if "mimic" in item.keywords:
            item.add_marker(skip_mimic)
