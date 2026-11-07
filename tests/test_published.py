"""Task 10.5: published graphs and vectors, packaged reproducibly and installed only verified."""

import hashlib
import io
import shutil
import urllib.request
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from text2icd import __version__, cli
from text2icd.embeddings import Vectors, normalize
from text2icd.fetch import FetchError
from text2icd.graph.model import load, read_meta
from text2icd.published import (
    BASE,
    HEADER,
    Published,
    PublishedError,
    fetch_published,
    load_published,
    package,
    registry_with,
)
from text2icd.resolve.retrieve import build_vectors


class Words:
    model = "test/words"
    dim = 16

    def encode(self, batch: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(batch), 16), dtype=np.float32)
        for row, text in enumerate(batch):
            for word in text.casefold().split():
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 16] += 1
        return normalize(vectors)


@pytest.fixture
def built(graph_file: Path, tmp_path: Path) -> Path:
    """The fixture graph under its release file name, with its vectors."""
    path = tmp_path / "built" / "text2icd-graph-fy2099.sqlite"
    path.parent.mkdir()
    shutil.copy(graph_file, path)
    with load(path=path) as graph:
        build_vectors(graph, Words(), "test:words")
    return path


@pytest.fixture
def packaged(built: Path, tmp_path: Path) -> tuple[Published, Path]:
    dist = tmp_path / "dist"
    return package(built, dist, __version__), dist


def serve(dist: Path, published: Published, calls: list[str] | None = None) -> object:
    def opener(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        assert request.full_url.startswith(f"{BASE}{published.tag}/")
        if calls is not None:
            calls.append(request.full_url)
        return io.BytesIO((dist / request.full_url.rsplit("/", 1)[1]).read_bytes())

    return opener


def test_package_is_reproducible_and_listed(built: Path, tmp_path: Path) -> None:
    first = package(built, tmp_path / "a", "0.2.0")
    again = package(built, tmp_path / "b", "0.2.0")
    assert first == again  # the same bytes: the hashes match
    assert first.tag == "data-fy2099-s1"
    assert first.graph == read_meta(built)["content_sha256"]
    assert [f.name for f in first.files] == [
        "text2icd-graph-fy2099.sqlite.xz",
        "text2icd-graph-fy2099.vectors.npy.gz",
        "text2icd-graph-fy2099.vectors.json.gz",
    ]
    assert load_published(first.toml()) == {"fy2099": first}


def test_fetch_installs_verified_files(packaged: tuple[Published, Path], tmp_path: Path) -> None:
    published, dist = packaged
    graph = tmp_path / "home" / "processed" / "text2icd-graph-fy2099.sqlite"
    calls: list[str] = []
    installed = fetch_published(published, graph, opener=serve(dist, published, calls))
    assert [(i.path.name, i.downloaded) for i in installed] == [
        ("text2icd-graph-fy2099.vectors.json", True),
        ("text2icd-graph-fy2099.vectors.npy", True),
        ("text2icd-graph-fy2099.sqlite", True),
    ]
    assert read_meta(graph)["content_sha256"] == published.graph
    # No download, partial or packed file is left.
    assert sorted(p.name for p in graph.parent.iterdir()) == sorted(i.path.name for i in installed)
    # Present and verified: nothing is downloaded again.
    again = fetch_published(published, graph, opener=serve(dist, published, calls))
    assert [i.downloaded for i in again] == [False, False, False]
    assert len(calls) == 3


def test_graph_only(packaged: tuple[Published, Path], tmp_path: Path) -> None:
    published, dist = packaged
    graph = tmp_path / "home" / "text2icd-graph-fy2099.sqlite"
    installed = fetch_published(published, graph, vectors=False, opener=serve(dist, published))
    assert [i.path.name for i in installed] == ["text2icd-graph-fy2099.sqlite"]


def test_a_tampered_download_is_refused(packaged: tuple[Published, Path], tmp_path: Path) -> None:
    published, dist = packaged
    xz = dist / "text2icd-graph-fy2099.sqlite.xz"
    xz.write_bytes(xz.read_bytes()[:-1] + b"\0")
    graph = tmp_path / "home" / "text2icd-graph-fy2099.sqlite"
    with pytest.raises(FetchError, match="does not match"):
        fetch_published(published, graph, vectors=False, opener=serve(dist, published))
    assert list(graph.parent.iterdir()) == []  # nothing left behind


def test_a_graph_of_other_content_is_not_installed(
    packaged: tuple[Published, Path], tmp_path: Path
) -> None:
    """The download verifies, but it is not the graph the entry names: refused after unpacking,
    and an existing graph is left as it was."""
    published, dist = packaged
    other = replace(published, graph="0" * 64)
    graph = tmp_path / "home" / "text2icd-graph-fy2099.sqlite"
    graph.parent.mkdir()
    graph.write_bytes(b"the user's own file")
    with pytest.raises(PublishedError, match="does not unpack to the published"):
        fetch_published(other, graph, vectors=False, opener=serve(dist, other))
    assert graph.read_bytes() == b"the user's own file"
    assert sorted(p.name for p in graph.parent.iterdir()) == [graph.name]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (('tag = "data-fy2099-s1"', 'tag = "../../evil"'), "not a data release tag"),
        (("text2icd-graph-fy2099.sqlite.xz", "text2icd-graph-fy2027.sqlite.xz"), "not a file of"),
        (("text2icd-graph-fy2099.sqlite.xz", "text2icd-graph-fy2099.sqlite.zip"), "not a file of"),
    ],
)
def test_registry_is_checked(
    packaged: tuple[Published, Path], change: tuple[str, str], message: str
) -> None:
    published, _ = packaged
    with pytest.raises(PublishedError, match=message):
        load_published(published.toml().replace(*change))


def test_shipped_registry_parses() -> None:
    assert isinstance(load_published(), dict)


def test_fetch_graph_command_without_a_published_graph(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.app, ["data", "fetch-graph", "-r", "fy2027", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "no published graph for fy2027" in result.output
    assert "build-graph -r fy2027" in result.output


def test_package_command(built: Path, tmp_path: Path) -> None:
    dist = tmp_path / "out"
    result = CliRunner().invoke(
        cli.app, ["data", "package", "--graph", str(built), "-o", str(dist)]
    )
    assert result.exit_code == 0, result.output
    entry = load_published((dist / "fy2099.toml").read_text("utf-8"))["fy2099"]
    assert {f.name for f in entry.files} <= {p.name for p in dist.iterdir()}


def test_registry_with_adds_or_replaces(packaged: tuple[Published, Path]) -> None:
    published, _ = packaged
    shipped = (
        Path(__file__).resolve().parents[1] / "src/text2icd/resources/graphs.toml"
    ).read_text()
    assert shipped == HEADER  # what the workflow writes into
    text = registry_with(published, shipped)
    assert text.startswith(HEADER)
    assert load_published(text) == {"fy2099": published}
    newer = replace(published, text2icd="9.9.9")
    assert load_published(registry_with(newer, text)) == {"fy2099": newer}
