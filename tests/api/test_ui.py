"""Tasks 10.10 and 10.11: the UI is served at / with a strict CSP, and its modules parse."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from text2icd.api.app import STATIC, create_app
from text2icd.graph.model import Graph
from text2icd.resolve.resolver import Resolver

MODULES = sorted(STATIC.glob("*.js"))


@pytest.fixture
def client(graph: Graph) -> TestClient:
    return TestClient(create_app(Resolver(graph=graph)))


def test_the_ui_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert '<script type="module" src="app.js">' in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"
    script = client.get("/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert client.get("/style.css").headers["content-type"].startswith("text/css")
    assert client.get("/missing.js").status_code == 404


def test_the_api_comes_first(client: TestClient, graph: Graph) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get(f"/v1/{graph.release}/codes/E11.22").json()["code"]["key"] == "E11.22"
    assert "content-security-policy" not in client.get("/docs").headers  # its own CDN assets


def test_pages_use_relative_urls() -> None:
    """The demo is served under a path (GitHub Pages): no URL may start at the root."""
    for path in [STATIC / "index.html", *MODULES]:
        text = path.read_text("utf-8")
        assert 'src="/' not in text
        assert 'href="/' not in text
        assert 'fetch("/' not in text


def test_text_is_never_parsed_as_html() -> None:
    """Phrases and CMS notes go into the page as text nodes only."""
    for path in MODULES:
        text = path.read_text("utf-8")
        for unsafe in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
            assert unsafe not in text, f"{path.name} uses {unsafe}"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_modules_parse(module: Path) -> None:
    done = subprocess.run(
        ["node", "--input-type=module", "--check"],
        input=module.read_text("utf-8"),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
