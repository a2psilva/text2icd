"""Task 10.12: the local review mode, on hand-written phrases and the fixture graph."""

import csv
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from text2icd import cli
from text2icd.api.app import create_app
from text2icd.api.review import AuditReview, GoldReview, ReviewError
from text2icd.eval.mimic import annotate, audit
from text2icd.graph.model import Graph
from text2icd.resolve.resolver import Resolver

SILVER = "T51.3X1A"  # the gold candidates' silver code: never among their choices


@pytest.fixture
def audit_file(tmp_path: Path) -> Path:
    path = tmp_path / "audit" / "mimic-silver.csv"
    path.parent.mkdir()
    rows = [
        {
            "id": f"n{i}:0",
            "phrase": f"type 2 diabetes with kidney disease {i}",
            "code": "E11.22",
            "description": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
            "score": "0.87",
            "verdict": "",
            "comment": "",
        }
        for i in range(3)
    ]
    audit.save_review(path, rows)
    return path


@pytest.fixture
def candidates(tmp_path: Path) -> Path:
    path = tmp_path / "annotation" / "candidates.jsonl"
    rows = [
        annotate.Candidate(
            id=f"n{i}:0",
            phrase=f"diabetic CKD {i}",
            subject_id=str(i),
            hadm_id=str(100 + i),
            note_id=f"n{i}",
            silver=SILVER,
            band="tail",
            chapter="19",
            choices=(
                ("E11.22", "Type 2 diabetes mellitus with diabetic CKD"),
                ("J69.0", "Pneumonitis due to inhalation of food and vomit"),
            ),
        )
        for i in range(2)
    ]
    annotate.write_jsonl(path, rows)
    return path


def client_for(graph: Graph, review: Any) -> TestClient:
    return TestClient(create_app(Resolver(graph=graph, retrieval=False), review))


def test_audit(graph: Graph, audit_file: Path) -> None:
    client = client_for(graph, AuditReview(audit_file))
    assert client.get("/healthz").json()["review"] == "audit"
    assert client.get("/v1/review").json() == {
        "mode": "audit",
        "total": 3,
        "answered": 0,
        "next": 0,
    }
    item = client.get("/v1/review/1")
    assert item.headers["cache-control"] == "no-store"
    assert item.json()["code"] == "E11.22"
    assert "score" not in item.json()  # judged blind to the alignment score
    saved = client.put("/v1/review/1", json={"verdict": "n", "comment": " wrong type "})
    assert saved.json()["verdict"] == "n"
    rows = audit.read_review(audit_file)
    assert (rows[1]["verdict"], rows[1]["comment"], rows[1]["score"]) == ("n", "wrong type", "0.87")
    assert audit.precision(rows).wrong == 1  # the terminal's scoring reads it
    assert client.get("/v1/review").json()["answered"] == 1
    assert client.put("/v1/review/0", json={"verdict": "maybe"}).status_code == 422
    assert client.put("/v1/review/9", json={"verdict": "y"}).status_code == 404
    assert client.get("/v1/review/-1").status_code == 422


def test_audit_resumes_with_the_terminal(graph: Graph, audit_file: Path) -> None:
    """A verdict typed in the terminal shows in the browser, and the other way round."""
    review = AuditReview(audit_file)
    rows = audit.read_review(audit_file)
    rows[0].update(verdict="y")
    audit.save_review(audit_file, rows)
    assert review.progress().next == 1
    client = client_for(graph, review)
    client.put("/v1/review/1", json={"verdict": "?"})
    assert [r["verdict"] for r in audit.read_review(audit_file)] == ["y", "?", ""]


def test_audit_keeps_added_columns(graph: Graph, audit_file: Path) -> None:
    with audit_file.open(newline="", encoding="utf-8") as fh:
        rows = [{**row, "note": "mine"} for row in csv.DictReader(fh)]
    audit.save_review(audit_file, rows, [*audit.COLUMNS, "note"])
    client_for(graph, AuditReview(audit_file)).put("/v1/review/0", json={"verdict": "y"})
    assert audit.read_review(audit_file)[0]["note"] == "mine"


def test_gold(graph: Graph, candidates: Path) -> None:
    client = client_for(graph, GoldReview(candidates, graph))
    item = client.get("/v1/review/0")
    assert item.headers["cache-control"] == "no-store"
    assert [c["code"] for c in item.json()["choices"]] == ["E11.22", "J69.0"]
    for hidden in (SILVER, "tail", "hadm", "subject", "note_id", "silver", "band", "chapter"):
        assert hidden not in item.text  # coded blind

    check = client.post("/v1/review/0/check", json={"line": "1, E85.4"}).json()
    assert [c["code"] for c in check["codes"]] == ["E11.22", "E85.4"]
    assert check["codes"][0]["description"].startswith("Type 2 diabetes")
    assert (check["verdict"], check["diagnoses"], check["error"]) == ("coded", 2, None)
    for line, error in [
        ("7", "no candidate 7"),
        ("E11.2", "not a billable code"),
        ("hello", "neither a candidate number nor a code"),
        ("l diabetes", "the page has the other commands"),
        (",", "at least one code"),
    ]:
        assert error in client.post("/v1/review/0/check", json={"line": line}).json()["error"]
        assert client.put("/v1/review/0", json={"line": line}).status_code == 422
    assert not (candidates.parent / "answers.jsonl").exists()  # nothing saved by a check

    saved = client.put("/v1/review/0", json={"line": "1 E85.4", "comment": "both"}).json()
    assert saved["answer"]["codes"][1]["code"] == "E85.4"
    client.put("/v1/review/1", json={"line": "s", "comment": "a plan"})
    answers = annotate.read_answers(candidates.parent / "answers.jsonl")
    assert answers["n0:0"] == annotate.Answer(
        id="n0:0", verdict="coded", codes=("E11.22", "E85.4"), comment="both", diagnoses=1
    )
    assert (answers["n1:0"].verdict, answers["n1:0"].comment) == ("skip", "a plan")
    assert client.get("/v1/review").json() == {
        "mode": "gold",
        "total": 2,
        "answered": 2,
        "next": None,
    }
    examples, verdicts = annotate.build_gold(
        annotate.read_candidates(candidates), answers, "fy2099"
    )
    assert [e.primary.code for e in examples] == ["E11.22"]  # `gold build` reads them
    assert verdicts["skip"] == 1


def test_missing_review_files(graph: Graph, tmp_path: Path) -> None:
    with pytest.raises(ReviewError, match="audit sample"):
        AuditReview(tmp_path / "none.csv")
    with pytest.raises(ReviewError, match="gold sample"):
        GoldReview(tmp_path / "candidates.jsonl", graph)
    bad = tmp_path / "bad.csv"
    bad.write_text("id,phrase\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="no code, comment"):
        AuditReview(bad)


def test_no_review_routes_by_default(graph: Graph) -> None:
    client = client_for(graph, None)
    assert client.get("/v1/review").status_code == 404
    assert client.get("/healthz").json()["review"] is None
    assert "cache-control" not in client.get(f"/v1/{graph.release}/codes/E11.22").headers


def test_serve_review(graph_file: Path, audit_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    started: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: started.update(options, app=app))
    graph = ["--graph", str(graph_file)]
    result = CliRunner().invoke(
        cli.app, ["serve", *graph, "--review", "audit", "-f", str(audit_file)]
    )
    assert result.exit_code == 0, result.output
    assert (started["host"], started["access_log"]) == ("127.0.0.1", False)
    assert "review: audit, 0 of 3 answered" in result.output
    for extra in (["--host", "0.0.0.0"], ["--access-log"], ["--llm"], ["--retrieval"]):
        refused = CliRunner().invoke(cli.app, ["serve", *graph, "--review", "audit", *extra])
        assert refused.exit_code == 2
        assert "127.0.0.1 only" in refused.output
    assert CliRunner().invoke(cli.app, ["serve", *graph, "--review", "x"]).exit_code == 2
    assert CliRunner().invoke(cli.app, ["serve", *graph, "-f", str(audit_file)]).exit_code == 2
    missing = CliRunner().invoke(
        cli.app, ["serve", *graph, "--review", "gold", "-f", str(audit_file.with_name("no"))]
    )
    assert missing.exit_code == 1
    assert "gold sample" in missing.output
