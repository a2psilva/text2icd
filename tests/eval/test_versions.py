import csv
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.eval.datasets import Example, read_dataset, write_dataset
from text2icd.eval.versions import (
    Conversion,
    Reconciler,
    load_reconciler,
    parse_conversion_table,
    reconcile_example,
)
from text2icd.ontology.codes import parse_reference
from text2icd.ontology.files import ReleaseFiles

REPO_DATA = Path(__file__).resolve().parents[2] / "data"

PREAMBLE = (
    '"The National Center for Health Statistics (NCHS) has published an update.\n'
    'For each new code the table shows its previously assigned code.",,\n'
)
TABLE = [
    ("N18.30", "2020", "N18.3"),
    ("N18.31", "2020", "N18.3"),
    ("H36.811", "2023", "H35.20-H35.23"),
    ("P91.821", "2020", "I63.- and P91.88"),
    ("W44.A0XA", "2023", "Codes in Categories T15-T19"),
    ("M04.8", "2016", "None"),
    ("C56.3", "2021", "C56.1 and C56.2"),
    ("S20.214A", "2020", "S20.211A; S20.212A & S20.219A"),
    ("T40.713D", "2021", "T40.7X3D "),
]


def write_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(PREAMBLE)
        writer = csv.writer(fh)
        writer.writerow(["Current code assignment", "Effective", "Previous Code(s) Assignment"])
        writer.writerows(TABLE)
        writer.writerow(["", "", ""])
    return path


def test_parse_conversion_table_csv(tmp_path: Path) -> None:
    rows = {row.code: row for row in parse_conversion_table(write_csv(tmp_path / "t.csv"))}
    assert len(rows) == len(TABLE)
    assert rows["N18.30"].previous == (parse_reference("N18.3"),)
    assert rows["H36.811"].previous == (parse_reference("H35.20-H35.23"),)
    # Broad references and free text name codes that still exist: not read backwards.
    assert rows["P91.821"].previous == (parse_reference("P91.88"),)
    assert rows["W44.A0XA"].previous == ()
    assert rows["M04.8"].previous == ()
    assert len(rows["S20.214A"].previous) == 3
    assert rows["T40.713D"].previous == (parse_reference("T40.7X3D"),)
    assert rows["C56.3"].written == "C56.1 and C56.2"


def test_parse_conversion_table_tab_separated(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    lines = [PREAMBLE.replace(",,", "\t\t").rstrip("\n")]
    lines.append("Current code assignment\tEffective\tPrevious Code(s) Assignment")
    lines += ["\t".join(row) for row in TABLE]
    path.write_bytes("\r\n".join(lines).encode("cp1252"))
    assert [row.code for row in parse_conversion_table(path)] == [row[0] for row in TABLE]


def test_parse_conversion_table_needs_its_header(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("Code,Previous\nN18.30,N18.3\n")
    with pytest.raises(ValueError, match="no 'current code assignment' header"):
        parse_conversion_table(path)


@pytest.fixture
def reconciler(tmp_path: Path) -> Reconciler:
    billable = {
        "fy2016": frozenset({"N18.3", "H35.21", "K85.8", "V29.88XA", "C56.1"}),
        "fy2020": frozenset({"N18.3", "K85.8", "C56.1", "A01.1"}),
        "fy2027": frozenset(
            {"N18.30", "N18.31", "H36.811", "K85.80", "K85.81", "C56.1", "C56.3", "A01.1"}
        ),
    }
    conversions = parse_conversion_table(write_csv(tmp_path / "t.csv"))
    # A chain: X10.0 was replaced by X10.1, which was itself expanded into X10.11 and X10.12.
    conversions += [
        Conversion("X10.1", "2018", (parse_reference("X10.0"),), "X10.0"),
        Conversion("X10.11", "2022", (parse_reference("X10.1"),), "X10.1"),
        Conversion("X10.12", "2022", (parse_reference("X10.1"),), "X10.1"),
    ]
    billable["fy2027"] |= {"X10.11", "X10.12"}
    return Reconciler("fy2027", billable, conversions)


@pytest.mark.parametrize(
    ("code", "status", "mapped_to", "how"),
    [
        ("C56.1", "valid", (), ""),
        ("N183", "mapped", ("N18.30", "N18.31"), "conversion table"),
        ("H35.21", "mapped", ("H36.811",), "conversion table"),
        ("X10.0", "mapped", ("X10.11", "X10.12"), "conversion table"),
        ("K85.8", "mapped", ("K85.80", "K85.81"), "subdivisions"),
        ("V29.88XA", "unmappable", (), "deleted without a replacement"),
        ("E11.6", "unmappable", (), "never billable"),
    ],
)
def test_reconcile(
    reconciler: Reconciler, code: str, status: str, mapped_to: tuple[str, ...], how: str
) -> None:
    result = reconciler.reconcile(code)
    assert (result.status, result.mapped_to, result.how) == (status, mapped_to, how)


def test_billable_spans(reconciler: Reconciler) -> None:
    assert reconciler.releases == ("fy2016", "fy2020", "fy2027")
    assert reconciler.spans(reconciler.reconcile("C56.1").billable_in) == "fy2016-fy2027"
    assert reconciler.spans(("fy2016", "fy2027")) == "fy2016, fy2027"
    assert reconciler.spans(()) == ""


def test_reconcile_example(reconciler: Reconciler) -> None:
    example = Example.model_validate(
        {
            "id": "x",
            "phrase": "ckd stage 3",
            "gold": [{"code": "N18.3", "role": "primary"}, {"code": "E11.6", "role": "companion"}],
            "release": "fy2020",
            "source": "test",
            "split": "dev",
        }
    )
    result = reconcile_example(example, reconciler)
    assert result.release == "fy2027"
    assert [(g.code, g.status, g.mapped_to) for g in result.gold] == [
        ("N18.3", "mapped", ("N18.30", "N18.31")),
        ("E11.6", "unmappable", ()),
    ]
    assert reconciler.reconcile_all(["N183", "N18.3", "C56.1"]).keys() == {"N18.3", "C56.1"}


def test_target_needs_billable_codes() -> None:
    with pytest.raises(ValueError, match="no billable codes for the target release fy2027"):
        Reconciler("fy2027", {"fy2026": frozenset()}, [])


@pytest.fixture
def data_dir(fixture_release: ReleaseFiles, tmp_path: Path) -> Path:
    """`fy2027` holds the fixture order file; `fy2016` an older one and a conversion table."""
    root = tmp_path / "data"
    shutil.copytree(fixture_release.root, root / "raw" / "cms" / "fy2027")
    old = root / "raw" / "cms" / "fy2016" / "order"
    old.mkdir(parents=True)
    old.joinpath("icd10cm_order_2016.txt").write_text(
        "00001 A00     0 Cholera" + " " * 53 + "Cholera\n"
        "00002 A009    1 Cholera, unspecified" + " " * 40 + "Cholera, unspecified\n"
        "00003 A01     1 Old typhoid code" + " " * 44 + "Old typhoid code\n"
    )
    table = root / "raw" / "cms" / "fy2027" / "conversion-table"
    table.mkdir()
    with (table / "table.csv").open("w", newline="") as fh:
        fh.write(PREAMBLE)
        csv.writer(fh).writerows(
            [["Current code assignment", "Effective", "Previous"], ["A00.0", "2017", "A01"]]
        )
    return root


def test_load_reconciler(data_dir: Path) -> None:
    reconciler = load_reconciler("fy2027", Settings(data_dir=data_dir))
    assert reconciler.releases == ("fy2016", "fy2027")
    assert reconciler.reconcile("A01").mapped_to == ("A00.0",)
    assert reconciler.reconcile("A00.9").status == "valid"
    with pytest.raises(FileNotFoundError):
        load_reconciler("fy2026", Settings(data_dir=data_dir))


runner = CliRunner()


def test_reconcile_command(data_dir: Path, tmp_path: Path) -> None:
    args = ["eval", "reconcile", "A01", "A009", "Z99", "-r", "fy2027", "--data-dir", str(data_dir)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "A01  mapped  (billable fy2016) -> A00.0 [conversion table]",
        "A00.9  valid  (billable fy2016-fy2027)",
        "Z99  unmappable: never billable",
    ]
    bad = runner.invoke(cli.app, ["eval", "reconcile", "HIV", "--data-dir", str(data_dir)])
    assert bad.exit_code == 1
    assert "not an ICD-10-CM code" in bad.output


def test_reconcile_command_rewrites_a_dataset(data_dir: Path, tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    example = {
        "id": "x",
        "phrase": "typhoid",
        "gold": [{"code": "A01", "role": "primary"}],
        "release": "fy2016",
        "source": "test",
        "split": "dev",
    }
    write_dataset(source, [Example.model_validate(example)])
    output = tmp_path / "out.jsonl"
    args = ["eval", "reconcile", "--dataset", str(source), "-o", str(output)]
    result = runner.invoke(cli.app, [*args, "-r", "fy2027", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    assert "1 examples against fy2027" in result.output
    (rewritten,) = read_dataset(output)
    assert (rewritten.release, rewritten.primary.mapped_to) == ("fy2027", ("A00.0",))
    alone = runner.invoke(cli.app, ["eval", "reconcile", "--dataset", str(source)])
    assert alone.exit_code != 0
    assert "--dataset and --output go together" in alone.output
    nothing = runner.invoke(cli.app, ["eval", "reconcile"])
    assert nothing.exit_code != 0


@pytest.mark.slow
def test_release_history(cms_release: Callable[[str], ReleaseFiles]) -> None:
    cms_release("fy2027")
    cms_release("fy2016")
    reconciler = load_reconciler("fy2027", Settings(data_dir=REPO_DATA))
    assert reconciler.releases[0] == "fy2016"
    assert reconciler.releases[-1] == "fy2027"
    assert reconciler.reconcile("N18.3").mapped_to == ("N18.30", "N18.31", "N18.32")
    assert reconciler.reconcile("A04.7").mapped_to == ("A04.71", "A04.72")
    assert reconciler.reconcile("E11.6").how == "never billable"


@pytest.mark.mimic
@pytest.mark.slow
def test_mimic_codes_reconcile(cms_release: Callable[[str], ReleaseFiles]) -> None:
    """Nearly every ICD-10 code assigned in MIMIC-IV is valid in FY2027 or maps to it."""
    cms_release("fy2027")
    reconciler = load_reconciler("fy2027", Settings(data_dir=REPO_DATA))
    counts: Counter[str] = Counter()
    with (REPO_DATA / "raw" / "mimic-iv" / "diagnoses_icd.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["icd_version"] == "10":
                counts[row["icd_code"]] += 1
    statuses: Counter[str] = Counter()
    for code, n in counts.items():
        statuses[reconciler.reconcile(code).status] += n
    assert statuses["unmappable"] / sum(statuses.values()) < 0.001
