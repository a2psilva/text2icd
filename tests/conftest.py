from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MIMIC_DIR = REPO_ROOT / "data" / "raw" / "mimic-iv"


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if any(MIMIC_DIR.glob("*.csv*")):
        return
    skip_mimic = pytest.mark.skip(reason=f"MIMIC-IV files not found in {MIMIC_DIR}")
    for item in items:
        if "mimic" in item.keywords:
            item.add_marker(skip_mimic)
