"""Tasks 10.10 and 10.11 in a real browser (Playwright's Chromium), on the fixture graph.
Skipped unless installed: `uv sync --group ui && uv run playwright install chromium`."""

import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")
playwright = pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402

from text2icd.api.app import create_app  # noqa: E402
from text2icd.graph.model import Graph  # noqa: E402
from text2icd.resolve.resolver import Resolver  # noqa: E402


@pytest.fixture
def base(graph: Graph) -> Iterator[str]:
    """The service on a free port, in a thread."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(Resolver(graph=graph)), port=port, log_level="warning")
    )
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
def page(base: str) -> Iterator[object]:
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
