"""The local MIMIC-IV files: discharge summaries (MIMIC-IV-Note) and billed diagnoses (MIMIC-IV
hosp), read as streams of rows. The files never leave `data/raw/mimic-iv/`."""

import csv
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from text2icd.config import Settings

DISCHARGE = "discharge.csv"
DIAGNOSES = "diagnoses_icd.csv"


class MimicFilesMissingError(FileNotFoundError):
    """A MIMIC-IV file is not in the data directory."""


@dataclass(frozen=True, slots=True)
class Note:
    note_id: str
    subject_id: str
    hadm_id: str
    text: str


def mimic_dir(settings: Settings) -> Path:
    return settings.data_dir / "raw" / "mimic-iv"


def mimic_file(settings: Settings, name: str) -> Path:
    path = mimic_dir(settings) / name
    if not path.is_file():
        raise MimicFilesMissingError(
            f"{path} not found; MIMIC-IV files are placed there by hand (see data/README.md)"
        )
    return path


def read_notes(path: Path) -> Iterator[Note]:
    """Discharge summaries, in file order."""
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            yield Note(row["note_id"], row["subject_id"], row["hadm_id"], row["text"])
