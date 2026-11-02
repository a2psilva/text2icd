"""The discharge diagnosis extractor, on hand-written synthetic notes (no MIMIC-IV text)."""

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.eval.mimic.files import Note, read_notes
from text2icd.eval.mimic.sections import (
    MAX_ITEMS,
    DiagnosisItem,
    SectionStats,
    diagnosis_items,
    section,
)

NOTE = """Name: ___ Unit No: ___

Discharge Disposition:
Home With Service

Facility:
___

Discharge Diagnosis:
PRIMARY DIAGNOSIS
=================
1. Community acquired pneumonia
2) Sepsis

SECONDARY DIAGNOSES:
- Hypertension; Hyperlipidemia
* Acute renal failure
Chronic: atrial fibrillation

Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
"""


def texts(note: str) -> list[tuple[str, str | None]]:
    return [(item.text, item.group) for item in diagnosis_items(note)]


def test_a_full_section() -> None:
    assert texts(NOTE) == [
        ("Community acquired pneumonia", "primary"),
        ("Sepsis", "primary"),
        ("Hypertension", "secondary"),
        ("Hyperlipidemia", "secondary"),
        ("Acute renal failure", "secondary"),
        ("atrial fibrillation", "secondary"),
    ]
    assert diagnosis_items(NOTE)[3] == DiagnosisItem("Hyperlipidemia", 3, "secondary")
    body = section(NOTE)
    assert body is not None
    assert body.startswith("PRIMARY DIAGNOSIS")
    assert "Mental Status" not in body


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("Discharge Diagnosis: UTI\nDischarge Condition:\nstable", [("UTI", None)]),
        (
            "Discharge Diagnoses:\nPrimary: sepsis\nSecondary diagnoses - anemia\n\n"
            "Discharge Instructions:\nCall your doctor.",
            [("sepsis", "primary"), ("anemia", "secondary")],
        ),
        (
            "DISCHARGE DIAGNOSIS:\nPrimary.\n# CHF exacerbation\nSecondary\n• COPD\n"
            "Discharge Condition:\n",
            [("CHF exacerbation", "primary"), ("COPD", "secondary")],
        ),
        (
            "Discharge Diagnosis:\nDx: small bowel obstruction\nPMH: asthma\n"
            "Discharge Condition:\n",
            [("small bowel obstruction", None), ("asthma", "secondary")],
        ),
        (
            "Discharge Diagnosis:\nAcute on chronic systolic heart failure\n"
            "Active issues:\nhypokalemia\nFollowup Instructions:\n",
            [("Acute on chronic systolic heart failure", None), ("hypokalemia", "primary")],
        ),
        (
            "Discharge Diagnosis:\nfall at ___\n--\nx\n123\nDischarge Condition:\n",
            [("fall at ___", None)],
        ),
    ],
)
def test_variants(note: str, expected: list[tuple[str, str | None]]) -> None:
    assert texts(note) == expected


def test_no_section() -> None:
    note = "Chief Complaint:\nfever\n\nBrief Hospital Course:\nThe patient did well."
    assert section(note) is None
    assert diagnosis_items(note) == []


def test_section_running_to_the_end() -> None:
    assert texts("Discharge Diagnosis:\nmigraine") == [("migraine", None)]


def test_too_many_items_is_not_a_list() -> None:
    lines = "\n".join(f"- problem {i}" for i in range(MAX_ITEMS + 1))
    assert diagnosis_items(f"Discharge Diagnosis:\n{lines}\nDischarge Condition:\n") == []
    lines = "\n".join(f"- problem {i}" for i in range(MAX_ITEMS))
    assert len(diagnosis_items(f"Discharge Diagnosis:\n{lines}\n")) == MAX_ITEMS


def test_stats() -> None:
    stats = SectionStats()
    stats.add(NOTE)
    stats.add("Chief Complaint:\nfever")
    stats.add("Discharge Diagnosis:\n\nDischarge Condition:\nstable")
    assert (stats.notes, stats.with_section, stats.with_items, stats.items) == (3, 2, 1, 6)
    assert stats.groups == {"primary": 2, "secondary": 4}
    assert stats.per_note == {"6-10": 1, "0": 1}
    lines = stats.lines()
    assert lines[0].startswith("notes 3; with a discharge diagnosis section 2 (66.7%)")
    assert lines[2] == "groups: primary 33.3%, secondary 66.7%, none 0.0%"


def write_notes(path: Path, notes: list[tuple[str, str, str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "note_id",
                "subject_id",
                "hadm_id",
                "note_type",
                "note_seq",
                "charttime",
                "storetime",
                "text",
            ]
        )
        for note_id, subject, hadm, text in notes:
            writer.writerow([note_id, subject, hadm, "DS", "1", "", "", text])
    return path


def test_read_notes(tmp_path: Path) -> None:
    path = write_notes(
        tmp_path / "discharge.csv", [("n1", "1", "11", NOTE), ("n2", "2", "22", "x")]
    )
    notes = list(read_notes(path))
    assert notes[0] == Note("n1", "1", "11", NOTE)
    assert notes[1].hadm_id == "22"


def test_sections_command(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_notes(data / "raw" / "mimic-iv" / "discharge.csv", [("n1", "1", "11", NOTE)] * 3)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["eval", "mimic", "sections", "--data-dir", str(data)])
    assert result.exit_code == 0, result.output
    assert "notes 3; with a discharge diagnosis section 3 (100.0%)" in result.output
    assert "Sepsis" not in result.output
    limited = runner.invoke(
        cli.app, ["eval", "mimic", "sections", "--limit", "1", "--data-dir", str(data)]
    )
    assert "notes 1;" in limited.output
    missing = runner.invoke(cli.app, ["eval", "mimic", "sections", "--data-dir", str(tmp_path)])
    assert missing.exit_code == 1
    assert "placed there by hand" in missing.output
