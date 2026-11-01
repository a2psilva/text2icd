import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load

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
