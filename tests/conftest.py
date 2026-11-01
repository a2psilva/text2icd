from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.fetch import COMPLETE_MARKER
from text2icd.ontology.files import ReleaseFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
MIMIC_DIR = REPO_ROOT / "data" / "raw" / "mimic-iv"
CMS_DIR = REPO_ROOT / "data" / "raw" / "cms"


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def cms_release() -> Callable[[str], ReleaseFiles]:
    """Files of a release fetched into data/raw/cms; skips the test when they are absent."""

    def locate(release_id: str) -> ReleaseFiles:
        files = ReleaseFiles(CMS_DIR / release_id)
        if not any(files.root.glob(f"*/{COMPLETE_MARKER}")):
            pytest.skip(f"CMS release {release_id} not fetched (run `text2icd data fetch`)")
        return files

    return locate


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if any(MIMIC_DIR.glob("*.csv*")):
        return
    skip_mimic = pytest.mark.skip(reason=f"MIMIC-IV files not found in {MIMIC_DIR}")
    for item in items:
        if "mimic" in item.keywords:
            item.add_marker(skip_mimic)
