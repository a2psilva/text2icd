"""Task 10.3: the HTTP service, on the fixture graph."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from text2icd import cli
from text2icd.api.app import MAX_BATCH, MAX_BODY, create_app
from text2icd.graph.model import Graph
from text2icd.resolve.resolver import MAX_PHRASE, Resolver
from text2icd.result import CodingResult

PNEUMONIA = "aspiration pneumonia due to food"
PROPANOL = "poisoning by propanol"


@pytest.fixture
def client(graph: Graph) -> TestClient:
    return TestClient(create_app(Resolver(graph=graph)))


def test_code(client: TestClient) -> None:
    response = client.post("/v1/code", json={"phrase": PNEUMONIA})
    assert response.status_code == 200
    result = CodingResult.model_validate(response.json())
    assert [c.code for c in result.codes] == ["J69.0"]
    assert result.detail is None
    explained = client.post("/v1/code", json={"phrase": PNEUMONIA, "explain": True}).json()
    assert explained["detail"]["paths"][0]["code"] == "J69.0"
    later = client.post("/v1/code", json={"phrase": PROPANOL, "episode": "subsequent"}).json()
    assert later["codes"][0]["code"] == "T51.3X1D"


@pytest.mark.parametrize(
    "body",
    [
        {"phrase": ""},
        {"phrase": "x" * (MAX_PHRASE + 1)},
        {"phrase": PNEUMONIA, "setting": "emergency"},
        {"phrase": PNEUMONIA, "extra": 1},
        {},
    ],
)
def test_code_refuses_bad_requests(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/v1/code", json=body).status_code == 422


def test_batch(client: TestClient) -> None:
    response = client.post("/v1/batch", json={"phrases": [PROPANOL, PNEUMONIA, PROPANOL]})
    results = [CodingResult.model_validate(r) for r in response.json()["results"]]
    assert [r.codes[0].code for r in results] == ["T51.3X1A", "J69.0", "T51.3X1A"]
    assert results[2].cached
    too_many = client.post("/v1/batch", json={"phrases": ["a"] * (MAX_BATCH + 1)})
    assert too_many.status_code == 422


def test_body_limit(client: TestClient) -> None:
    big = b'{"phrase": "' + b"a" * MAX_BODY + b'"}'
    response = client.post("/v1/code", content=big, headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_code_page(client: TestClient, graph: Graph) -> None:
    page = client.get(f"/v1/{graph.release}/codes/E11.22").json()
    assert page["code"]["key"] == "E11.22"
    assert page["code"]["billable"] is True
    assert [a["key"] for a in page["ancestors"]] == ["E11.2", "E11", "E08-E13", "4"]
    assert page["notes"]
    assert page["inherited"]  # E11's notes, among others
    pointed = client.get(f"/v1/{graph.release}/codes/J69.0").json()
    assert "Pneumonia > aspiration > due to > food" in [row["path"] for row in pointed["terms"]]
    assert client.get(f"/v1/{graph.release}/codes/J690").json()["code"]["key"] == "J69.0"
    assert client.get(f"/v1/{graph.release}/codes/Z99.999").status_code == 404
    assert client.get("/v1/fy1999/codes/J69.0").status_code == 404


def test_term_page(client: TestClient, graph: Graph) -> None:
    term = graph.term_at("Pneumonia > aspiration")
    assert term is not None
    page = client.get(f"/v1/{graph.release}/terms/{term.id}").json()
    assert page["term"]["path"] == "Pneumonia > aspiration"
    assert [a["title"] for a in page["ancestors"]] == ["Pneumonia"]
    assert any(s["title"] == "due to" for s in page["subterms"])
    assert client.get(f"/v1/{graph.release}/terms/99999999").status_code == 404
    assert client.get(f"/v1/{graph.release}/terms/-1").status_code == 422


def test_search(client: TestClient, graph: Graph) -> None:
    found = client.get(f"/v1/{graph.release}/search", params={"q": "pneumonia"}).json()
    assert found["index"][0]["term"]["path"] == "Pneumonia"
    diabetes = client.get(f"/v1/{graph.release}/search", params={"q": "diabetes"}).json()
    assert "E11.22" in [hit["code"]["key"] for hit in diabetes["tabular"]]
    drug = client.get(f"/v1/{graph.release}/search", params={"q": "propanol", "book": "drug"})
    assert drug.json()["tabular"] == []  # one book: the Index only
    assert client.get(f"/v1/{graph.release}/search", params={"q": ""}).status_code == 422
    assert (
        client.get(f"/v1/{graph.release}/search", params={"q": "a", "book": "x"}).status_code == 422
    )


def test_service_routes(client: TestClient, graph: Graph) -> None:
    health = client.get("/healthz").json()
    assert health == {
        "status": "ok",
        "release": graph.release,
        "graph": graph.meta["content_sha256"],
        "fts5": True,
    }
    assert client.get("/version").json()["release"] == graph.release
    page = client.get(f"/v1/{graph.release}/codes/J69.0", headers={"accept-encoding": "gzip"})
    assert page.status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"] == "text2icd"


def test_serve_command(graph_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    started: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: started.update(options, app=app))
    result = CliRunner().invoke(cli.app, ["serve", "--graph", str(graph_file)])
    assert result.exit_code == 0, result.output
    assert (started["host"], started["port"], started["access_log"]) == ("127.0.0.1", 8000, False)
    assert "at http://127.0.0.1:8000" in result.output
    exposed = CliRunner().invoke(
        cli.app, ["serve", "--graph", str(graph_file), "--host", "0.0.0.0"]
    )
    assert "no authentication" in exposed.output
