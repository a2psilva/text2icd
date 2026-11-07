"""`text2icd setup` (task 10.9): from an install to a working coder in one command.

1. SQLite: the full-text search the graph relies on (FTS5) must be compiled in.
2. The graph: kept when present; else downloaded when this text2icd lists it
   (`resources/graphs.toml`); else built from the two CMS files it needs (fetched and verified).
3. L2, when the `embeddings` extra is installed: the vectors, downloaded when the published ones
   were built for this graph (a graph built here from other CMS files gets none, and L2 stays
   off with a hint), then the model they were embedded with, at its pinned revision.
4. A check: the resolver codes a known phrase.

Each step reports as it ends, so a download is never silent; an optional part that cannot be set
up is reported and skipped, not an error.
"""

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from text2icd.config import Settings
from text2icd.embeddings import download_model, downloaded, extra_for, missing
from text2icd.extras import install_hint
from text2icd.fetch import GRAPH_FILES, Opener, fetch_release
from text2icd.graph.build import build_graph
from text2icd.graph.model import GraphError, load, read_meta
from text2icd.ontology.files import ReleaseFiles
from text2icd.published import Published, fetch_published, load_published
from text2icd.releases import Release
from text2icd.resolve.resolver import Resolver
from text2icd.resolve.retrieve import DEFAULT_MODEL, VectorsError, read_vectors_meta, vectors_path

CHECK = "type 2 DM with diabetic CKD stage 3a"  # E11.22, N18.31 in every release since FY2021
Status = Literal["ok", "done", "skipped"]


class SetupError(Exception):
    """A required step failed: text2icd cannot code without it."""


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    status: Status  # ok: already there; done: set up now; skipped: optional, with the reason
    detail: str


def fts5() -> bool:
    """Whether this Python's SQLite has FTS5."""
    with closing(sqlite3.connect(":memory:")) as conn:
        try:
            conn.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        except sqlite3.OperationalError:
            return False
    return True


def setup(
    settings: Settings,
    release: Release,
    *,
    build: bool = False,
    l2: bool = True,
    report: Callable[[Step], None] = lambda step: None,
    published: dict[str, Published] | None = None,
    opener: Opener | None = None,
) -> list[Step]:
    """Set up `release` in the settings' data directory. `build`: build the graph even when a
    published one exists. `published` and `opener`: the registry and downloads (tests)."""
    steps: list[Step] = []

    def done(step: Step) -> None:
        steps.append(step)
        report(step)

    if not fts5():
        raise SetupError(
            f"SQLite {sqlite3.sqlite_version} in this Python has no FTS5 full-text search; use "
            "a Python built with it (python.org, Homebrew or `uv python install`)"
        )
    done(Step("SQLite", "ok", f"{sqlite3.sqlite_version}, with FTS5"))
    entry = (load_published() if published is None else published).get(release.id)
    graph = settings.graph_path(release.id)
    done(_graph(settings, release, graph, None if build else entry, opener))
    if l2:
        for step in _l2(graph, entry, opener):
            done(step)
    else:
        done(Step("L2", "skipped", "--no-l2"))
    done(_check(settings, graph))
    return steps


def _graph(
    settings: Settings,
    release: Release,
    graph: Path,
    entry: Published | None,
    opener: Opener | None,
) -> Step:
    if graph.is_file():
        try:
            read_meta(graph)
            return Step("graph", "ok", f"{graph}")
        except GraphError:
            pass  # unreadable: replaced below
    if entry is not None:
        fetch_published(entry, graph, vectors=False, opener=opener)
        return Step("graph", "done", f"{graph}, downloaded ({entry.tag})")
    raw = settings.cms_raw_dir(release.id)
    fetch_release(release, raw, kinds=GRAPH_FILES, opener=opener)
    graph.unlink(missing_ok=True)
    report = build_graph(ReleaseFiles(raw), release.id, graph)
    return Step("graph", "done", f"{report.path}, built from the CMS files")


def _l2(graph: Path, entry: Published | None, opener: Opener | None) -> list[Step]:
    if missing(DEFAULT_MODEL):
        hint = install_hint(extra_for(DEFAULT_MODEL))
        return [Step("L2", "skipped", f"needs the embeddings extra: {hint}")]
    content = read_meta(graph).get("content_sha256")
    vectors = vectors_path(graph)
    steps = []
    try:
        meta = read_vectors_meta(vectors)
        current = meta.graph == content
    except VectorsError:
        current = False
    if current:
        steps.append(Step("vectors", "ok", str(vectors)))
    elif entry is not None and entry.graph == content:
        fetch_published(entry, graph, opener=opener)
        steps.append(Step("vectors", "done", f"{vectors}, downloaded ({entry.tag})"))
    else:
        why = (
            "none is published for this release"
            if entry is None
            else "the published ones are for another graph"
        )
        return [
            Step(
                "vectors",
                "skipped",
                f"{why}; `text2icd graph embed -r {read_meta(graph)['release']}` builds them "
                "(a few minutes). L1 codes without them",
            )
        ]
    meta = read_vectors_meta(vectors)
    if downloaded(meta.model, meta.revision):
        steps.append(Step("L2 model", "ok", meta.model_id))
    else:
        download_model(meta.model, meta.revision)
        steps.append(Step("L2 model", "done", f"{meta.model_id}, downloaded"))
    return steps


def _check(settings: Settings, graph: Path) -> Step:
    with load(path=graph) as loaded, Resolver(settings=settings, graph=loaded) as resolver:
        result = resolver.code(CHECK)
        layers = "L1+L2" if resolver.retriever is not None else "L1"
    codes = " ".join(code.code for code in result.codes) or "no code"
    return Step("check", "ok", f'{layers}: "{CHECK}" -> {codes}')
