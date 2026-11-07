"""Task 10.13: the static demo, exported from the fixture graph."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.demo import STATIC, Demo, DemoError, export, key, load_demo
from text2icd.graph.model import Graph
from text2icd.resolve.resolver import Resolver

PHRASES = """
release = "fy2099"

[[group]]
name = "Combination codes"
examples = [
  { phrase = "aspiration pneumonia due to food", codes = ["J69.0"] },
  { phrase = "amyloid heart disease", codes = ["E85.4", "I43"] },
]

[[group]]
name = "Uncertain diagnoses"
examples = [{ phrase = "possible  Aspiration pneumonia", codes = ["J69.0"], outpatient = [] }]
"""


@pytest.fixture
def phrases(tmp_path: Path) -> Path:
    path = tmp_path / "demo.toml"
    path.write_text(PHRASES, encoding="utf-8")
    return path


def read(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def test_export(graph: Graph, phrases: Path, tmp_path: Path) -> None:
    out = tmp_path / "site"
    with Resolver(graph=graph, retrieval=False) as resolver:
        done = export(resolver, load_demo(phrases), out)
    assert (done.results, done.checked) == (4, True)
    index = read(out / "data" / "index.json")
    assert index["health"]["release"] == "fy2099"
    assert index["examples"][1] == {
        "group": "Uncertain diagnoses",
        "phrases": ["possible  Aspiration pneumonia"],
    }
    assert set(index["results"]) == {
        "inpatient||aspiration pneumonia due to food",
        "inpatient||amyloid heart disease",
        "inpatient||possible aspiration pneumonia",
        "outpatient||possible aspiration pneumonia",
    }
    outpatient = read(out / "data" / index["results"]["outpatient||possible aspiration pneumonia"])
    assert (outpatient["setting"], outpatient["codes"]) == ("outpatient", [])

    # Every term and code a result names has its page, and so does every page's ancestor.
    codes, terms = set(index["pages"]["codes"]), set(index["pages"]["terms"])
    assert {"J69.0", "E85.4", "I43"} <= codes
    for name in index["results"].values():
        result = read(out / "data" / name)
        for path in result["detail"]["paths"]:
            assert {step["term"] for step in path["steps"]} <= terms
            assert path["code"] in codes
    for code in codes:
        page = read(out / "data" / "fy2099" / "codes" / f"{code}.json")
        assert {a["key"] for a in page["ancestors"]} <= codes
    for term in terms:
        page = read(out / "data" / "fy2099" / "terms" / f"{term}.json")
        assert {a["id"] for a in page["ancestors"]} <= terms

    html = (out / "index.html").read_text("utf-8")
    assert '<meta name="text2icd-source" content="static" />' in html
    assert {p.name for p in STATIC.glob("*.js")} <= {p.name for p in out.glob("*.js")}
    assert (out / ".nojekyll").is_file()


def test_the_files_are_the_service_answers(graph: Graph, phrases: Path, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from text2icd.api.app import create_app

    out = tmp_path / "site"
    with Resolver(graph=graph, retrieval=False) as resolver:
        export(resolver, load_demo(phrases), out)
        client = TestClient(create_app(resolver))
        index = read(out / "data" / "index.json")
        live = client.post("/v1/code", json={"phrase": "amyloid heart disease", "explain": True})
        saved = read(out / "data" / index["results"]["inpatient||amyloid heart disease"])
        assert saved == live.json() | {"cached": False}  # the export filled the cache
        for code in index["pages"]["codes"][:20]:
            saved = read(out / "data" / "fy2099" / "codes" / f"{code}.json")
            assert saved == client.get(f"/v1/fy2099/codes/{code}").json()
        term = index["pages"]["terms"][0]
        saved = read(out / "data" / "fy2099" / "terms" / f"{term}.json")
        assert saved == client.get(f"/v1/fy2099/terms/{term}").json()
        assert index["health"] == client.get("/healthz").json() | {"fts5": False}


def test_an_example_that_codes_differently_stops_the_export(
    graph: Graph, phrases: Path, tmp_path: Path
) -> None:
    wrong = Demo.model_validate_json(
        '{"release": "fy2099", "group": [{"name": "x", "examples": '
        '[{"phrase": "amyloid heart disease", "codes": ["I43"]}]}]}'
    )
    with Resolver(graph=graph, retrieval=False) as resolver:
        with pytest.raises(DemoError, match=r"E85\.4 I43, listed I43"):
            export(resolver, wrong, tmp_path / "site")
        assert not (tmp_path / "site").exists()
        other = wrong.model_copy(update={"release": "fy2027"})  # another release: not checked
        assert not export(resolver, other, tmp_path / "site").checked


def test_export_directory(graph: Graph, phrases: Path, tmp_path: Path) -> None:
    demo = load_demo(phrases)
    with Resolver(graph=graph, retrieval=False) as resolver:
        busy = tmp_path / "busy"
        busy.mkdir()
        (busy / "notes.txt").write_text("mine", encoding="utf-8")
        with pytest.raises(DemoError, match="not empty"):
            export(resolver, demo, busy)
        out = tmp_path / "site"
        export(resolver, demo, out)
        stale = out / "data" / "fy2099" / "codes" / "Z99.9.json"
        stale.write_text("{}", encoding="utf-8")
        export(resolver, demo, out)  # an earlier export is replaced
        assert not stale.exists()


def test_key() -> None:
    assert key("inpatient", None, "  Possible   Pneumonia ") == "inpatient||possible pneumonia"
    assert key("outpatient", "initial", "x") == "outpatient|initial|x"


def test_the_packaged_phrases() -> None:
    demo = load_demo()
    phrases = [e.phrase for g in demo.group for e in g.examples]
    assert demo.release == "fy2027"
    assert 35 <= len(phrases) <= 60
    assert len({key("inpatient", None, p) for p in phrases}) == len(phrases)
    assert all(code for g in demo.group for e in g.examples for code in e.codes)


def test_bad_phrases(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('release = "fy2027"\n[[group]]\nname = "x"\nexamples = []\n', encoding="utf-8")
    with pytest.raises(DemoError, match="cannot read"):
        load_demo(bad)
    with pytest.raises(DemoError, match="cannot read"):
        load_demo(tmp_path / "missing.toml")


def test_demo_export_command(graph_file: Path, phrases: Path, tmp_path: Path) -> None:
    out = tmp_path / "site"
    args = ["demo", "export", str(out), "--phrases", str(phrases), "--graph", str(graph_file)]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "4 answers" in result.output
    assert "checked on fy2099" in result.output
    assert "http.server" in result.output
    busy = CliRunner().invoke(cli.app, [*args[:2], str(phrases.parent), *args[3:]])
    assert busy.exit_code == 1
    assert "not empty" in busy.output
