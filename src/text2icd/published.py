"""Published data: a release's graph and vectors files, built by the data-release workflow and
attached to a GitHub Release, so that a user downloads them instead of building (task 10.5).

`resources/graphs.toml` lists what this version of text2icd can download, per release:

    [fy2027]
    tag = "data-fy2027-s1"      # the GitHub Release
    schema = 1                  # the graph's schema_version
    graph = "0f4d9e66..."       # the graph's content hash, which its vectors were built for
    text2icd = "0.2.0"          # the version that built them

    [[fy2027.files]]
    name = "text2icd-graph-fy2027.sqlite.xz"
    sha256 = "..."              # of the file as downloaded
    size = 10312345

The graph is compressed with xz (75 MB to 10 MB); the vectors with gzip, since float16 hardly
compresses and xz takes 4 s to unpack them. Only files under `BASE` are fetched. Each one is
checked as downloaded (SHA-256 and size; xz and gzip add their own checksums), unpacked beside its
target, checked again (the graph's content hash and schema; the graph the vectors were built for;
the matrix's shape), and only then renamed onto the target. A graph and vectors that do not match
are never used together, whatever is left after a failure: the vectors record their graph.
"""

import gzip
import lzma
import re
import shutil
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from text2icd.fetch import CHUNK, Opener, download, sha256_of
from text2icd.graph.model import GraphError, read_meta
from text2icd.resolve.retrieve import VectorsError, VectorsMeta, read_vectors_meta, vectors_path

BASE = "https://github.com/a2psilva/text2icd/releases/download/"
REGISTRY = "graphs.toml"
TAG = re.compile(r"^data-fy\d{4}(-apr)?-s\d+$")
NAME = re.compile(
    r"^text2icd-graph-(fy\d{4}(?:-apr)?)\.(sqlite|vectors\.npy|vectors\.json)\.(xz|gz)$"
)
KINDS = ("vectors.json", "vectors.npy", "sqlite")  # the order they are installed in
MAX_UNPACKED = 2 << 30  # bytes a file may unpack to; a graph is under 100 MB


HEADER = """\
# Published data: the graph and vectors files of each release that this version of text2icd can
# download (`text2icd data fetch-graph`, `text2icd setup`), attached to GitHub Releases by the
# data-release workflow (.github/workflows/data-release.yml), which writes this file
# (`text2icd data package --registry`) and opens the PR. The format is described in
# src/text2icd/published.py. With no entry for a release, text2icd builds its graph instead.
"""


class PublishedError(Exception):
    """The registry is malformed, or a published file is missing or does not verify."""


@dataclass(frozen=True, slots=True)
class PublishedFile:
    name: str  # as attached to the GitHub Release
    sha256: str
    size: int

    @property
    def target(self) -> str:
        """The file it unpacks to."""
        return self.name.rsplit(".", 1)[0]

    @property
    def kind(self) -> str:
        """sqlite, vectors.npy or vectors.json."""
        found = NAME.fullmatch(self.name)
        assert found is not None
        return found.group(2)


@dataclass(frozen=True, slots=True)
class Published:
    release: str
    tag: str
    schema: int
    graph: str  # the graph's content hash
    text2icd: str
    files: tuple[PublishedFile, ...]

    def url(self, file: PublishedFile) -> str:
        return f"{BASE}{self.tag}/{file.name}"

    def toml(self) -> str:
        """Its entry in `graphs.toml`."""
        lines = [
            f"[{self.release}]",
            f'tag = "{self.tag}"',
            f"schema = {self.schema}",
            f'graph = "{self.graph}"',
            f'text2icd = "{self.text2icd}"',
        ]
        for file in self.files:
            lines += [
                "",
                f"[[{self.release}.files]]",
                f'name = "{file.name}"',
                f'sha256 = "{file.sha256}"',
                f"size = {file.size}",
            ]
        return "\n".join(lines) + "\n"


def load_published(text: str | None = None) -> dict[str, Published]:
    """The registry (`resources/graphs.toml`, or `text`), checked: tags, file names, hashes."""
    if text is None:
        text = resources.files("text2icd.resources").joinpath(REGISTRY).read_text("utf-8")
    try:
        data: dict[str, Any] = tomllib.loads(text)
        return {release: _entry(release, entry) for release, entry in data.items()}
    except (tomllib.TOMLDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PublishedError(f"{REGISTRY}: {exc}") from exc


def _entry(release: str, entry: dict[str, Any]) -> Published:
    files = tuple(
        PublishedFile(str(f["name"]), str(f["sha256"]), int(f["size"])) for f in entry["files"]
    )
    published = Published(
        release=release,
        tag=str(entry["tag"]),
        schema=int(entry["schema"]),
        graph=str(entry["graph"]),
        text2icd=str(entry["text2icd"]),
        files=files,
    )
    if not TAG.fullmatch(published.tag):
        raise ValueError(f"{release}: {published.tag!r} is not a data release tag")
    for file in files:
        found = NAME.fullmatch(file.name)
        if found is None or found.group(1) != release:
            raise ValueError(f"{release}: {file.name!r} is not a file of this release")
        if not re.fullmatch(r"[0-9a-f]{64}", file.sha256):
            raise ValueError(f"{release}: {file.name} has no SHA-256")
    if sorted(f.kind for f in files) != sorted(KINDS):
        raise ValueError(f"{release}: a graph, its vectors and their .json file are needed")
    return published


def registry_with(published: Published, text: str) -> str:
    """The registry `text` with `published` added, or replacing its release's entry."""
    entries = load_published(text)
    entries[published.release] = published
    return HEADER + "".join("\n" + entries[release].toml() for release in sorted(entries))


# -- Packaging (the data-release workflow) -----------------------------------------------------


def package(graph_path: Path, out: Path, text2icd: str, tag: str | None = None) -> Published:
    """Compress a built graph and its vectors files into `out`, and describe them."""
    meta = read_meta(graph_path)
    release, schema = meta["release"], int(meta["schema_version"])
    vectors = vectors_path(graph_path)
    try:
        if read_vectors_meta(vectors).graph != meta["content_sha256"]:
            raise PublishedError(f"{vectors} was built for another graph")
    except VectorsError as exc:
        raise PublishedError(str(exc)) from exc
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for source, suffix in (
        (graph_path, ".xz"),
        (vectors, ".gz"),
        (vectors.with_suffix(".json"), ".gz"),
    ):
        packed = out / (source.name + suffix)
        _compress(source, packed)
        files.append(PublishedFile(packed.name, sha256_of(packed), packed.stat().st_size))
    return Published(
        release=release,
        tag=tag or f"data-{release}-s{schema}",
        schema=schema,
        graph=meta["content_sha256"],
        text2icd=text2icd,
        files=tuple(files),
    )


def _compress(source: Path, packed: Path) -> None:
    """Compress reproducibly: the same file gives the same bytes (no name or time stored)."""
    with source.open("rb") as raw, packed.open("wb") as fh:
        if packed.suffix == ".xz":
            with lzma.open(fh, "wb", preset=6) as xz:
                shutil.copyfileobj(raw, xz, CHUNK)
        else:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=fh, mtime=0) as gz:
                shutil.copyfileobj(raw, gz, CHUNK)


# -- Fetching ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Installed:
    path: Path
    downloaded: bool  # False: already present and verified


def fetch_published(
    published: Published, graph: Path, *, vectors: bool = True, opener: Opener | None = None
) -> list[Installed]:
    """Install the release's graph at `graph` (and its vectors beside it), unless verified copies
    are there."""
    done = []
    for file in sorted(published.files, key=lambda f: KINDS.index(f.kind)):
        if not vectors and file.kind != "sqlite":
            continue
        target = graph.with_name(file.target)
        if _verified(published, file.kind, target, graph):
            done.append(Installed(target, False))
            continue
        url = published.url(file)
        if not url.startswith(BASE):
            raise PublishedError(f"{url} is not a text2icd data release")
        packed = target.with_name(file.name)
        try:
            download(url, file.sha256, file.size, packed, opener=opener)
            _unpack(packed, target, published, file.kind, graph)
        finally:
            packed.unlink(missing_ok=True)
        done.append(Installed(target, True))
    return done


def _unpack(packed: Path, target: Path, published: Published, kind: str, graph: Path) -> None:
    partial = target.with_name(target.name + ".partial")
    try:
        with (
            (
                lzma.open(packed, "rb") if packed.suffix == ".xz" else gzip.open(packed, "rb")
            ) as source,
            partial.open("wb") as out,
        ):
            written = 0
            for chunk in iter(lambda: source.read(CHUNK), b""):
                written += len(chunk)
                if written > MAX_UNPACKED:
                    raise PublishedError(f"{packed.name} unpacks to over {MAX_UNPACKED} bytes")
                out.write(chunk)
        if not _verified(published, kind, partial, graph):
            raise PublishedError(f"{packed.name} does not unpack to the published {target.name}")
        partial.replace(target)
    except (OSError, EOFError, lzma.LZMAError) as exc:
        raise PublishedError(f"{packed.name} could not be unpacked: {exc}") from exc
    finally:
        partial.unlink(missing_ok=True)


def _verified(published: Published, kind: str, path: Path, graph: Path) -> bool:
    """Whether `path` holds the published file of this kind: a graph with its content hash and
    schema; a vectors JSON file built for that graph; a matrix of the shape that file gives."""
    if not path.is_file():
        return False
    if kind == "sqlite":
        try:
            meta = read_meta(path)
        except GraphError:
            return False
        return (meta.get("content_sha256"), meta.get("schema_version")) == (
            published.graph,
            str(published.schema),
        )
    if kind == "vectors.json":
        found = _vectors_meta(path)
        return found is not None and found.graph == published.graph
    described = _vectors_meta(vectors_path(graph).with_suffix(".json"))
    if described is None or described.graph != published.graph:
        return False
    try:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return False
    return bool(matrix.dtype == np.float16 and matrix.shape == (described.rows, described.dim))


def _vectors_meta(path: Path) -> VectorsMeta | None:
    try:
        return VectorsMeta.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError, ValidationError):
        return None
