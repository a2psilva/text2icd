"""The GitHub Pages demo (task 10.13): `text2icd demo export DIR` writes the UI as static files,
with the answers to the demo's phrases and the graph pages they touch.

The page is the same client as `text2icd serve`: its `source.js` reads files instead of the
service (the page's `text2icd-source` is `static`), and each file holds what the service would
answer:

- `data/index.json`: `/healthz` and `/version`, the examples by group, the result file of each
  (setting, phrase), and the pages held;
- `data/results/<hash>.json`: `POST /v1/code` with `explain`;
- `data/<release>/codes/<code>.json` and `data/<release>/terms/<id>.json`: the explorer's pages.

The pages held are the terms and codes the results name; the codes one link away from those
codes (children, and the codes their notes name or are named by); the terms their
cross-references lead to and the codes they give; and every page's ancestors. The UI shows a
link to anything else as plain text: following every subterm too would add some 1,600 pages.
Search needs the service.

The phrases (`resources/demo.toml`) were written for the demo, so the export holds CMS data and
nothing else. Each lists the codes the deterministic coder gives it on the demo's release; on
that release the export fails when one codes differently.
"""

import hashlib
import json
import shutil
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from text2icd.api.views import code_view, term_view
from text2icd.graph.model import Graph
from text2icd.resolve.resolver import Resolver
from text2icd.text.model import Setting

STATIC = Path(__file__).with_name("api") / "static"
LIVE = '<meta name="text2icd-source" content="live" />'
STATIC_SOURCE = '<meta name="text2icd-source" content="static" />'


class DemoError(ValueError):
    """The demo's phrases cannot be read, or an example codes differently from its list."""


class DemoExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phrase: str = Field(min_length=1)
    codes: tuple[str, ...]  # inpatient
    outpatient: tuple[str, ...] | None = None  # also shown in the outpatient setting

    def settings(self) -> list[tuple[Setting, tuple[str, ...]]]:
        both: list[tuple[Setting, tuple[str, ...] | None]] = [
            ("inpatient", self.codes),
            ("outpatient", self.outpatient),
        ]
        return [(setting, codes) for setting, codes in both if codes is not None]


class DemoGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    examples: tuple[DemoExample, ...] = Field(min_length=1)


class Demo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release: str  # the release whose codes the examples list
    group: tuple[DemoGroup, ...] = Field(min_length=1)


@dataclass
class Exported:
    out: Path
    results: int = 0
    codes: int = 0
    terms: int = 0
    size: int = 0  # bytes written
    checked: bool = False  # whether the examples' codes were checked (the demo's release)


@dataclass
class _Bundle:
    files: dict[str, str] = field(default_factory=dict)  # path under data/ -> JSON text
    results: dict[str, str] = field(default_factory=dict)  # key -> path
    codes: set[str] = field(default_factory=set)
    terms: set[int] = field(default_factory=set)


def load_demo(path: Path | None = None) -> Demo:
    """The demo's phrases: `path`, or the packaged `resources/demo.toml`."""
    try:
        if path is None:
            raw = resources.files("text2icd.resources").joinpath("demo.toml").read_bytes()
        else:
            raw = path.read_bytes()
        return Demo.model_validate(tomllib.loads(raw.decode("utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise DemoError(f"cannot read the demo's phrases: {exc}") from exc


def key(setting: str, episode: str | None, phrase: str) -> str:
    """The result's key in `index.json`; `source.js` computes the same."""
    return f"{setting}|{episode or ''}|{' '.join(phrase.split()).lower()}"


def export(resolver: Resolver, demo: Demo, out: Path) -> Exported:
    """Write the demo to `out`, which must be empty, missing, or an earlier export."""
    graph = resolver.graph
    checked = demo.release == graph.release
    bundle = _Bundle()
    wrong = []
    for group in demo.group:
        for example in group.examples:
            for setting, expected in example.settings():
                result = resolver.code(example.phrase, setting=setting, explain=True)
                found = tuple(c.code for c in result.codes)
                if checked and found != expected:
                    wrong.append(
                        f"{example.phrase!r} ({setting}): {' '.join(found) or 'no code'}, "
                        f"listed {' '.join(expected) or 'no code'}"
                    )
                body = result.model_dump(mode="json")
                text = _json(body)
                name = f"results/{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"
                bundle.files[name] = text
                bundle.results[key(setting, None, example.phrase)] = name
                _named(body, graph, bundle)
    if wrong:
        raise DemoError(
            f"{len(wrong)} examples code differently on {graph.release}:\n  " + "\n  ".join(wrong)
        )
    _close(graph, bundle)
    for code in sorted(bundle.codes):
        bundle.files[f"{graph.release}/codes/{code}.json"] = _json(
            code_view(graph, code).model_dump(mode="json")
        )
    for term in sorted(bundle.terms):
        bundle.files[f"{graph.release}/terms/{term}.json"] = _json(
            term_view(graph, term).model_dump(mode="json")
        )
    bundle.files["index.json"] = _json(
        {
            "health": {
                "status": "ok",
                "release": graph.release,
                "graph": graph.meta.get("content_sha256", ""),
                "fts5": False,
                "review": None,
            },
            "version": resolver.info(),
            "examples": [
                {"group": g.name, "phrases": [e.phrase for e in g.examples]} for g in demo.group
            ],
            "results": bundle.results,
            "pages": {"codes": sorted(bundle.codes), "terms": sorted(bundle.terms)},
        }
    )
    size = _write(out, bundle.files)
    return Exported(
        out=out,
        results=len(bundle.results),
        codes=len(bundle.codes),
        terms=len(bundle.terms),
        size=size,
        checked=checked,
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _named(value: Any, graph: Graph, bundle: _Bundle) -> None:
    """Every Index term and code a result names: `term` ids, and `code`, `source` and `kept`."""
    if isinstance(value, dict):
        for name, item in value.items():
            if name == "term" and isinstance(item, int):
                bundle.terms.add(item)
            elif name in ("code", "source", "kept") and isinstance(item, str) and item in graph:
                bundle.codes.add(item)
            else:
                _named(item, graph, bundle)
    elif isinstance(value, list):
        for item in value:
            _named(item, graph, bundle)


def _close(graph: Graph, bundle: _Bundle) -> None:
    """Add the pages around those the results name (see the module's docstring), then every
    page's ancestors, so that breadcrumbs always lead somewhere."""
    codes, terms = set(bundle.codes), set(bundle.terms)
    for code in sorted(bundle.codes):
        page = code_view(graph, code)
        codes.update(c.key for c in page.children)
        codes.update(e.target for e in page.edges if e.target)
        codes.update(e.source for e in page.named_by)
    for term in sorted(bundle.terms):
        entry = term_view(graph, term)
        terms.update(x.target for x in entry.xrefs if x.target is not None)
        codes.update(p.code for p in entry.pointers if p.code)
    bundle.codes = codes | {a.key for code in codes for a in graph.ancestors(code)}
    bundle.terms = terms | set(_term_ancestors(graph, terms))


def _term_ancestors(graph: Graph, terms: Iterable[int]) -> Iterable[int]:
    for term in terms:
        parent = graph.term_parent(term)
        while parent is not None:
            yield parent.id
            parent = graph.term_parent(parent.id)


def _write(out: Path, files: dict[str, str]) -> int:
    if out.exists() and any(out.iterdir()) and not (out / "data" / "index.json").is_file():
        raise DemoError(f"{out} is not empty and holds no earlier export; choose another directory")
    shutil.rmtree(out / "data", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    size = 0
    for page in sorted(STATIC.iterdir()):
        if page.suffix not in (".html", ".js", ".css"):
            continue
        text = page.read_text("utf-8")
        if page.name == "index.html":
            if text.count(LIVE) != 1:
                raise DemoError(f"{page} has no {LIVE}")
            text = text.replace(LIVE, STATIC_SOURCE)
        size += _put(out / page.name, text)
    size += _put(out / ".nojekyll", "")  # served as they are by GitHub Pages
    for name, text in files.items():
        size += _put(out / "data" / name, text)
    return size


def _put(path: Path, text: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return len(data)
