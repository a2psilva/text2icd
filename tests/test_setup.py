"""Task 10.9: `text2icd setup`, from an install to a working coder; network calls are stubbed."""

import hashlib
import io
import shutil
import urllib.request
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import text2icd.setup as setup_impl
from text2icd import __version__, cli
from text2icd.config import Settings
from text2icd.embeddings import Vectors, normalize
from text2icd.graph.model import load, read_meta
from text2icd.ontology.files import ReleaseFiles
from text2icd.published import BASE, Published, package
from text2icd.releases import Release
from text2icd.resolve.retrieve import build_vectors
from text2icd.setup import SetupError, Step
from text2icd.setup import setup as run_setup


class Words:
    """A stand-in embedder: hashed word counts."""

    model = "test/words"
    dim = 16

    def encode(self, batch: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(batch), 16), dtype=np.float32)
        for row, text in enumerate(batch):
            for word in text.casefold().split():
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 16] += 1
        return normalize(vectors)


RELEASE = Release("fy2099", 2099, date(2098, 10, 1), "full", ())


def statuses(steps: list[Step]) -> dict[str, str]:
    return {step.name: step.status for step in steps}


@pytest.fixture
def published(graph_file: Path, tmp_path: Path) -> tuple[Published, Path]:
    built = tmp_path / "built" / "text2icd-graph-fy2099.sqlite"
    built.parent.mkdir()
    shutil.copy(graph_file, built)
    with load(path=built) as graph:
        build_vectors(graph, Words(), "test:words")
    dist = tmp_path / "dist"
    return package(built, dist, __version__), dist


def opener(dist: Path, calls: list[str]) -> Any:
    def open_url(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        assert request.full_url.startswith(BASE)
        calls.append(request.full_url)
        return io.BytesIO((dist / request.full_url.rsplit("/", 1)[1]).read_bytes())

    return open_url


@pytest.fixture
def no_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check phrase needs the shipped calibration's model; the test vectors are another."""
    monkeypatch.setattr(setup_impl, "_check", lambda settings, graph: Step("check", "ok", "-"))


@pytest.mark.usefixtures("no_check")
def test_downloads_the_published_graph_and_vectors(
    published: tuple[Published, Path], tmp_path: Path
) -> None:
    entry, dist = published
    settings = Settings(data_dir=tmp_path / "home")
    calls: list[str] = []
    reported: list[Step] = []
    steps = run_setup(
        settings,
        RELEASE,
        published={"fy2099": entry},
        opener=opener(dist, calls),
        report=reported.append,
    )
    assert reported == steps
    assert statuses(steps) == {
        "SQLite": "ok",
        "graph": "done",
        "vectors": "done",
        "L2 model": "ok",  # test:words downloads nothing
        "check": "ok",
    }
    assert read_meta(settings.graph_path("fy2099"))["content_sha256"] == entry.graph
    assert len(calls) == 3
    again = run_setup(settings, RELEASE, published={"fy2099": entry}, opener=opener(dist, calls))
    assert statuses(again)["graph"] == statuses(again)["vectors"] == "ok"
    assert len(calls) == 3  # nothing downloaded again


def test_builds_when_nothing_is_published(
    fixture_release: ReleaseFiles, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture's CMS files are in place (tmp_path/raw/cms/fy2099): no fetch is needed."""
    fetched: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        setup_impl,
        "fetch_release",
        lambda release, raw, kinds, opener: fetched.append(tuple(kinds)),
    )
    steps = run_setup(Settings(data_dir=tmp_path), RELEASE, published={})
    assert fetched == [("tables-index", "order")]  # the two files the build reads
    assert statuses(steps)["graph"] == "done"
    assert "built from the CMS files" in steps[1].detail
    (vectors,) = (s for s in steps if s.name == "vectors")
    assert vectors.status == "skipped"
    assert "none is published" in vectors.detail
    assert steps[-1].detail.startswith("L1: ")


@pytest.mark.usefixtures("no_check")
def test_a_present_graph_is_kept(
    published: tuple[Published, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, dist = published
    settings = Settings(data_dir=tmp_path / "home")
    calls: list[str] = []
    run_setup(settings, RELEASE, published={"fy2099": entry}, opener=opener(dist, calls), l2=False)
    assert len(calls) == 1  # the graph only
    steps = run_setup(settings, RELEASE, published={"fy2099": entry}, l2=False)
    assert statuses(steps) == {"SQLite": "ok", "graph": "ok", "L2": "skipped", "check": "ok"}


@pytest.mark.usefixtures("no_check")
def test_l2_needs_its_extra(
    published: tuple[Published, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, dist = published
    monkeypatch.setattr(setup_impl, "missing", lambda spec: ("onnxruntime",))
    steps = run_setup(
        Settings(data_dir=tmp_path), RELEASE, published={"fy2099": entry}, opener=opener(dist, [])
    )
    (l2,) = (s for s in steps if s.name == "L2")
    assert l2.status == "skipped"
    assert "uv sync --extra embeddings" in l2.detail


@pytest.mark.usefixtures("no_check")
def test_downloads_the_model(
    published: tuple[Published, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, dist = published
    asked: list[tuple[str, str | None]] = []
    monkeypatch.setattr(setup_impl, "downloaded", lambda spec, revision=None: False)
    monkeypatch.setattr(
        setup_impl, "download_model", lambda spec, revision=None: asked.append((spec, revision))
    )
    steps = run_setup(
        Settings(data_dir=tmp_path), RELEASE, published={"fy2099": entry}, opener=opener(dist, [])
    )
    assert statuses(steps)["L2 model"] == "done"
    assert asked == [("test:words", None)]


def test_needs_fts5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_impl, "fts5", lambda: False)
    with pytest.raises(SetupError, match="FTS5"):
        run_setup(Settings(data_dir=tmp_path), RELEASE, published={})
    result = CliRunner().invoke(cli.app, ["setup", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "FTS5" in result.output


def test_this_python_has_fts5() -> None:
    assert setup_impl.fts5()
