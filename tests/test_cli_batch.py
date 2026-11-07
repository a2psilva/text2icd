"""Tasks 10.1 and 10.2: the Python API, and `text2icd batch` over JSON lines, CSV and lines."""

import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import text2icd
from text2icd import batch, cli
from text2icd.graph.model import Graph
from text2icd.resolve.resolver import MAX_PHRASE, Resolver
from text2icd.result import CodingResult

PNEUMONIA = "aspiration pneumonia due to food"
PROPANOL = "poisoning by propanol"


def lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines()]


@pytest.mark.parametrize(
    ("name", "fmt"),
    [
        ("a.jsonl", "jsonl"),
        ("A.NDJSON", "jsonl"),
        ("a.csv", "csv"),
        ("a.txt", "lines"),
        ("-", "lines"),
    ],
)
def test_detect(name: str, fmt: str) -> None:
    assert batch.detect(name) == fmt


def test_detect_refuses_an_unknown_extension() -> None:
    with pytest.raises(batch.BatchError, match="--format"):
        batch.detect("phrases.xlsx")


def test_read_jsonl() -> None:
    text = '{"id": 7, "phrase": "a"}\n\nnot json\n[1]\n{"id": true, "phrase": 3}\n{"phrase": "b"}\n'
    rows = list(batch.read_rows(io.StringIO(text), "jsonl"))
    assert [(r.line, r.id, r.phrase, r.error) for r in rows] == [
        (1, 7, "a", None),
        (3, None, None, "not JSON: Expecting value"),
        (4, None, None, "not a JSON object"),
        (5, None, None, 'no "phrase" string'),  # a boolean is not an id
        (6, None, "b", None),
    ]


def test_read_csv_and_lines() -> None:
    csv = 'id,phrase\nx1,"hip fracture, left"\nx2,\n'
    rows = list(batch.read_rows(io.StringIO(csv), "csv"))
    assert [(r.line, r.id, r.phrase) for r in rows] == [
        (2, "x1", "hip fracture, left"),
        (3, "x2", ""),
    ]
    with pytest.raises(batch.BatchError, match="no 'text' column"):
        list(batch.read_rows(io.StringIO(csv), "csv", column="text"))
    plain = list(batch.read_rows(io.StringIO("a\r\n\n  \nb\n"), "lines"))
    assert [(r.line, r.phrase) for r in plain] == [(1, "a"), (4, "b")]


def test_run_keeps_the_order_and_reports_errors(graph: Graph) -> None:
    rows = [
        batch.Row(1, "a", PNEUMONIA),
        batch.Row(2, "b", "   "),
        batch.Row(3, "c", PROPANOL),
        batch.Row(4, "d", "pneumonia " * (MAX_PHRASE // 10 + 1)),
        batch.Row(5, "e", PNEUMONIA),
        batch.Row(6, None, None, "not a JSON object"),
    ]
    out = io.StringIO()
    summary = batch.run(Resolver(graph=graph), rows, out, batch_size=2)
    written = lines(out.getvalue())
    assert [(r["line"], r["id"]) for r in written] == [(r.line, r.id) for r in rows]
    assert [r.get("error") for r in written] == [
        None,
        "empty phrase",
        None,
        f"over {MAX_PHRASE:,} characters",
        None,
        "not a JSON object",
    ]
    first = CodingResult.model_validate(written[0]["result"])
    assert [c.code for c in first.codes] == ["J69.0"]
    assert CodingResult.model_validate(written[4]["result"]).cached  # the cache spans batches
    assert (summary.rows, summary.coded, summary.errors, summary.repeats) == (6, 3, 3, 1)


def test_batch_command(graph_file: Path, tmp_path: Path) -> None:
    source = tmp_path / "phrases.csv"
    source.write_text(f"id,phrase\n1,{PNEUMONIA}\n2,{PROPANOL}\n3,\n", encoding="utf-8")
    out = tmp_path / "out" / "coded.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["batch", str(source), "-o", str(out), "--graph", str(graph_file), "--explain"]
    )
    assert result.exit_code == 0, result.output
    assert "3 rows: 2 coded" in result.output  # the summary, on stderr
    written = lines(out.read_text("utf-8"))
    assert [r["id"] for r in written] == ["1", "2", "3"]
    coded = CodingResult.model_validate(written[0]["result"])
    assert coded.detail is not None
    later = runner.invoke(
        cli.app,
        ["batch", "-", "--graph", str(graph_file), "--episode", "subsequent"],
        input=f"{PROPANOL}\n",
    )
    assert lines(later.stdout)[0]["result"]["codes"][0]["code"] == "T51.3X1D"  # type: ignore[index]


def test_batch_command_errors(graph_file: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    missing = runner.invoke(
        cli.app, ["batch", str(tmp_path / "none.txt"), "--graph", str(graph_file)]
    )
    assert missing.exit_code == 1
    assert "does not exist" in missing.output
    unknown = runner.invoke(cli.app, ["batch", "a.xlsx", "--graph", str(graph_file)])
    assert unknown.exit_code == 2
    no_column = tmp_path / "x.csv"
    no_column.write_text("text\nacute appendicitis\n", encoding="utf-8")
    refused = runner.invoke(cli.app, ["batch", str(no_column), "--graph", str(graph_file)])
    assert refused.exit_code == 1
    assert "no 'phrase' column" in refused.output


@pytest.fixture
def shared(graph: Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    """The API's shared resolver, on the fixture graph."""
    monkeypatch.setattr(text2icd, "_resolvers", {None: Resolver(graph=graph)})


@pytest.mark.usefixtures("shared")
def test_python_api() -> None:
    result = text2icd.code(PNEUMONIA)
    assert isinstance(result, text2icd.CodingResult)
    assert [c.code for c in result.codes] == ["J69.0"]
    assert text2icd.code(PNEUMONIA).cached  # the same resolver
    many = list(text2icd.code_many([PROPANOL, PNEUMONIA], episode="subsequent"))
    assert [r.codes[0].code for r in many] == ["T51.3X1D", "J69.0"]
    assert text2icd.Resolver is Resolver
    with pytest.raises(AttributeError):
        text2icd.missing  # noqa: B018
