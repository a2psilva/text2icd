"""Tasks 10.10 to 10.13 in a real browser (Playwright's Chromium), on the fixture graph.
Skipped unless installed: `uv sync --group ui && uv run playwright install chromium`."""

import re
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
playwright = pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402

from text2icd.api.app import create_app  # noqa: E402
from text2icd.api.review import AuditReview, GoldReview, Review  # noqa: E402
from text2icd.demo import export, load_demo  # noqa: E402
from text2icd.eval.mimic import annotate, audit  # noqa: E402
from text2icd.graph.model import Graph  # noqa: E402
from text2icd.resolve.resolver import Resolver  # noqa: E402


@contextmanager
def serving(graph: Graph, review: Review | None = None) -> Iterator[str]:
    """The service on a free port, in a thread."""
    with serving_app(create_app(Resolver(graph=graph, retrieval=False), review)) as url:
        yield url


@contextmanager
def serving_app(app: Any) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


@pytest.fixture
def base(graph: Graph) -> Iterator[str]:
    with serving(graph) as url:
        yield url


@pytest.fixture
def page() -> Iterator[object]:
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as exc:
            pytest.skip(f"no browser: {exc}")
        page = browser.new_page()
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.errors = errors  # type: ignore[attr-defined]
        yield page
        browser.close()
        assert errors == []


def test_path_view(page: object, base: str) -> None:
    p: Any = page
    p.goto(f"{base}/#/")
    p.fill("#phrase", "aspiration pneumonia due to food")
    p.click("button[type=submit]")
    p.wait_for_selector(".lane")
    assert "J69.0" in p.inner_text(".summary .primary")
    lane = p.locator(".lane").first
    assert lane.locator(".node.term .title").all_inner_texts() == [
        "Pneumonia",
        "aspiration",
        "due to",
        "food",
    ]
    assert "J69.0" in lane.locator(".node.coded").inner_text()
    p.hover(".chunk")
    assert "lit" in (lane.get_attribute("class") or "")
    lane.locator(".node.final button").click()
    p.wait_for_selector(".panel .node-view")
    assert "J69.0" in p.inner_text(".panel .crumbs")
    assert "q=aspiration" in p.url  # the result can be linked


def test_explorer(page: object, base: str) -> None:
    p: Any = page
    p.goto(f"{base}/#/explore")
    p.fill("input[type=search]", "pneumonia")
    p.press("input[type=search]", "Enter")
    p.wait_for_selector(".search-results li a")
    p.locator(".search-results li a").first.click()
    p.wait_for_selector(".node-view")
    assert "#/term/" in p.url
    p.goto(f"{base}/#/code/E11.22")
    p.wait_for_selector(".node-view .description")
    assert "Type 2 diabetes" in p.inner_text(".node-view .description")


PHRASES = ["diabetic kidney disease", "unclear fragment"]


def test_gold_review(page: object, graph: Graph, tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    annotate.write_jsonl(
        candidates,
        [
            annotate.Candidate(
                id=f"n{i}:0",
                phrase=phrase,
                subject_id=str(i),
                hadm_id=str(i),
                note_id=f"n{i}",
                silver="T51.3X1A",
                band="head",
                chapter="19",
                choices=(("E11.22", "Type 2 diabetes mellitus with diabetic CKD"),),
            )
            for i, phrase in enumerate(PHRASES)
        ],
    )
    p: Any = page
    with serving(graph, GoldReview(candidates, graph)) as url:
        p.goto(f"{url}/")
        p.wait_for_selector(".review-phrase")
        assert p.url.endswith("#/review/1")
        assert "T51.3X1A" not in p.content()  # coded blind
        p.fill("input.codes", "1, E85.4")
        p.wait_for_selector(".preview li >> nth=1")
        assert "2 diagnoses" in p.inner_text(".preview")
        p.fill(".lookup-search input", "pneumonia")
        p.press(".lookup-search input", "Enter")
        p.locator(".lookup .search-results li a").first.click()
        p.wait_for_selector(".lookup .node-view")
        p.press("input.codes", "Enter")
        p.wait_for_url(re.compile("#/review/2$"))
        p.wait_for_selector(f"text={PHRASES[1]}")
        p.click("text=Skip")
        p.wait_for_selector(".answered")
        assert "skipped" in p.inner_text(".answered")
        assert not any(word in p.url for word in ["diabetic", "kidney", "fragment", "pneumonia"])
    answers = annotate.read_answers(tmp_path / "answers.jsonl")
    assert answers["n0:0"].codes == ("E11.22", "E85.4")
    assert answers["n0:0"].diagnoses == 2
    assert answers["n1:0"].verdict == "skip"


def test_audit_review(page: object, graph: Graph, tmp_path: Path) -> None:
    path = tmp_path / "audit.csv"
    rows = [
        {"id": f"n{i}:0", "phrase": phrase, "code": "E11.22", "description": "Type 2 DM w CKD"}
        for i, phrase in enumerate(PHRASES)
    ]
    audit.save_review(path, [{**r, "score": "0.9", "verdict": "", "comment": ""} for r in rows])
    p: Any = page
    with serving(graph, AuditReview(path)) as url:
        p.goto(f"{url}/#/review/1")
        p.wait_for_selector(".review-phrase")
        p.fill(".field input", "checked the Index")
        p.locator(".review-phrase").click()  # out of the field: the keys give verdicts
        p.keyboard.press("n")
        p.wait_for_url(re.compile("#/review/2$"))
        p.wait_for_selector(f"text={PHRASES[1]}")
        p.keyboard.press("ArrowLeft")
        p.wait_for_selector(".verdict.on")
        assert "Wrong" in p.inner_text(".verdict.on")
    saved = audit.read_review(path)
    assert (saved[0]["verdict"], saved[0]["comment"]) == ("n", "checked the Index")


DEMO = """
release = "fy2099"
[[group]]
name = "Combination codes"
examples = [{ phrase = "amyloid heart disease", codes = ["E85.4", "I43"] }]
[[group]]
name = "Uncertain diagnoses"
examples = [{ phrase = "possible aspiration pneumonia", codes = ["J69.0"], outpatient = [] }]
"""


def test_static_demo(page: object, graph: Graph, tmp_path: Path) -> None:
    """The export, served as files under a path, as GitHub Pages serves it."""
    from starlette.applications import Starlette
    from starlette.staticfiles import StaticFiles

    phrases = tmp_path / "demo.toml"
    phrases.write_text(DEMO, encoding="utf-8")
    with Resolver(graph=graph, retrieval=False) as resolver:
        export(resolver, load_demo(phrases), tmp_path / "site")
    app = Starlette()
    app.mount("/text2icd", StaticFiles(directory=tmp_path / "site", html=True))
    p: Any = page
    with serving_app(app) as url:
        p.goto(f"{url}/text2icd/")
        p.wait_for_selector(".example-groups")
        assert "fy2099 · demo" in p.inner_text("#release")
        p.click(".example-groups >> text=amyloid heart disease")
        p.wait_for_selector(".lane")
        assert "E85.4" in p.inner_text(".summary .primary")
        p.locator(".lane .node.final button").first.click()
        p.wait_for_selector(".panel .node-view")
        p.goto(f"{url}/text2icd/#/?q=possible+aspiration+pneumonia&setting=outpatient")
        p.wait_for_selector(".summary")
        assert "Guideline IV.H" in p.inner_text(".summary")
        p.goto(f"{url}/text2icd/#/?q=headache")
        p.wait_for_selector("#result .error")
        assert "example phrases only" in p.inner_text("#result .error")
        p.goto(f"{url}/text2icd/#/explore")
        p.wait_for_selector(".explore-body p")
        assert p.locator("input[type=search]").count() == 0  # no search in the demo
