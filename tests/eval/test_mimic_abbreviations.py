"""Abbreviation mining on hand-written synthetic notes (no MIMIC-IV data)."""

import csv
from pathlib import Path

from typer.testing import CliRunner

from text2icd import cli
from text2icd.eval.mimic.abbreviations import _known, expansion_words, mine, write_candidates
from text2icd.eval.mimic.files import Note
from text2icd.eval.mimic.splits import split_of_subject
from text2icd.text.abbreviations import default_lexicon

SUBJECTS = [str(n) for n in range(1, 200)]
DEV = [s for s in SUBJECTS if split_of_subject(s) == "dev"]
TEST = [s for s in SUBJECTS if split_of_subject(s) == "test"]
KNOWN = frozenset(
    {"hypertension", "atrial", "fibrillation", "with", "rapid", "ventricular", "response"}
    | {"syndrome", "sepsis"}
)


def note(i: int, subject: str, *items: str) -> Note:
    section = "\n".join(items)
    text = f"Chief Complaint:\nx\n\nDischarge Diagnosis:\n{section}\n\nDischarge Condition:\nok\n"
    return Note(f"n{i}", subject, f"h{i}", text)


NOTES = [
    note(1, DEV[0], "HTN", "Afib w RVR", "Zorbex syndrome"),
    note(2, DEV[1], "HTN", "zorbex syndrome", "Qlt"),
    note(3, DEV[2], "Sepsis"),
    note(4, TEST[0], "HTN", "Zorbex syndrome"),  # test split: never read
    note(5, DEV[3], ""),
]


def test_mine_counts() -> None:
    mined = mine(NOTES, default_lexicon(), KNOWN)
    assert mined.notes == 3  # dev notes with items
    assert mined.items == 7
    assert mined.with_expansion == 3  # HTN twice, Afib w RVR
    assert mined.key_notes["HTN"] == 2
    assert mined.keys["A-fib"] == 1
    assert mined.chosen[("HTN", "only")] == 2
    assert mined.token_notes == {"zorbex": 2, "qlt": 1}
    assert mined.token_capitals["qlt"] == 0  # written "Qlt"
    assert mined.with_unknown == 3
    limited = mine(NOTES, default_lexicon(), KNOWN, limit=1)
    assert limited.notes == 1


def test_lines_show_only_frequent_words() -> None:
    mined = mine(NOTES, default_lexicon(), KNOWN)
    shown = "\n".join(mined.lines(min_notes=2))
    assert "zorbex" in shown
    assert "qlt" not in shown
    assert "most used lexicon keys (notes): HTN 2, A-fib 1, RVR 1" in shown
    assert "dev notes 3; items 7" in shown


def test_write_candidates(tmp_path: Path) -> None:
    mined = mine(NOTES, default_lexicon(), KNOWN)
    path = tmp_path / "interim" / "candidates.tsv"
    assert write_candidates(path, mined) == 1
    rows = list(csv.reader(path.open(), delimiter="\t"))
    assert rows == [["token", "notes", "items", "items_in_capitals"], ["zorbex", "2", "2", "0"]]


def test_known_words() -> None:
    known = frozenset({"effusion", "metastasis", "allergy", "stroke"})
    for word in ("effusion", "effusions", "metastases", "allergies", "strokes"):
        assert _known(word, known)
    assert not _known("zorbex", known)
    assert "percutaneous" in expansion_words(default_lexicon())


def test_mine_command(tmp_path: Path, graph_file: Path) -> None:
    data = tmp_path / "data"
    discharge = data / "raw" / "mimic-iv" / "discharge.csv"
    discharge.parent.mkdir(parents=True)
    with discharge.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["note_id", "subject_id", "hadm_id", "text"])
        for n in NOTES:
            writer.writerow([n.note_id, n.subject_id, n.hadm_id, n.text])
    runner = CliRunner()
    args = ["eval", "mimic", "abbrev", "mine", "--graph", str(graph_file), "--data-dir", str(data)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "dev notes 3; items 7" in result.output
    assert "zorbex" not in result.output  # used in fewer than 50 notes
    assert (data / "interim" / "abbreviation-candidates.tsv").is_file()
    low = runner.invoke(cli.app, [*args, "--min-notes", "5"])
    assert low.exit_code == 2
    assert "at least 50" in low.output
    missing = runner.invoke(cli.app, [*args[:4], "--data-dir", str(tmp_path)])
    assert missing.exit_code == 1
