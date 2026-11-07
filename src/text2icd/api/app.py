"""The HTTP service (task 10.3): `create_app(resolver)`, served by `text2icd serve`.

- `POST /v1/code`: one phrase, `{phrase, setting, episode, explain}`, gives a `CodingResult`;
  `explain` adds the paths and decisions the UI draws.
- `POST /v1/batch`: up to `MAX_BATCH` phrases, `{results: [...]}` in order.
- `GET /v1/{release}/codes/{code}`, `/v1/{release}/terms/{id}`, `/v1/{release}/search?q=`:
  the graph, for the explorer (`api.views`).
- `GET /healthz` (with the FTS5 check) and `GET /version` (what the resolver runs).

The endpoints are synchronous: FastAPI runs them in its thread pool, and the resolver is safe to
share between threads. Phrases are never logged (the resolver logs hashes only, task 9.7), and
`text2icd serve` turns the access log off, since search URLs hold the query. Requests over
`MAX_BODY` bytes are refused before they are read. No CORS headers are sent: the UI is served by
the same origin.
"""

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from text2icd import __version__
from text2icd.api.views import (
    Book,
    CodeView,
    SearchView,
    TermView,
    code_view,
    search_view,
    term_view,
)
from text2icd.resolve.resolver import MAX_PHRASE, Resolver
from text2icd.resolve.tabular import Episode
from text2icd.result import CodingResult
from text2icd.setup import fts5
from text2icd.text.model import Setting

MAX_BATCH = 1_000  # phrases per batch request
MAX_BODY = 2 << 20  # bytes: a full batch of the longest phrases fits


class CodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase: Annotated[str, Field(min_length=1, max_length=MAX_PHRASE)]
    setting: Setting = "inpatient"
    episode: Episode | None = None
    explain: bool = False


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrases: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=MAX_PHRASE)]],
        Field(min_length=1, max_length=MAX_BATCH),
    ]
    setting: Setting = "inpatient"
    episode: Episode | None = None
    explain: bool = False


class BatchResponse(BaseModel):
    results: list[CodingResult]


class Health(BaseModel):
    status: Literal["ok"]
    release: str
    graph: str
    fts5: bool


def create_app(resolver: Resolver) -> FastAPI:
    """The service over `resolver`, which the caller builds, warms and closes."""
    graph = resolver.graph
    app = FastAPI(
        title="text2icd",
        version=__version__,
        description="ICD-10-CM codes for diagnosis phrases, with the Index path to each code.",
    )
    app.add_middleware(GZipMiddleware, minimum_size=1_000)

    @app.middleware("http")
    async def limit_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "a Content-Length is required"}, status_code=411)
            if not length.isdigit() or int(length) > MAX_BODY:
                return JSONResponse({"detail": f"over {MAX_BODY:,} bytes"}, status_code=413)
        return await call_next(request)

    def served(release: str) -> None:
        if release != graph.release:
            raise HTTPException(404, f"this server serves {graph.release}, not {release}")

    @app.post("/v1/code", tags=["coding"])
    def code(request: CodeRequest) -> CodingResult:
        """Code one phrase."""
        return resolver.code(
            request.phrase,
            setting=request.setting,
            episode=request.episode,
            explain=request.explain,
        )

    @app.post("/v1/batch", tags=["coding"])
    def batch(request: BatchRequest) -> BatchResponse:
        """Code up to 1,000 phrases; the results come in the same order."""
        results = resolver.code_many(
            request.phrases,
            setting=request.setting,
            episode=request.episode,
            explain=request.explain,
        )
        return BatchResponse(results=list(results))

    @app.get("/v1/{release}/codes/{code}", tags=["graph"])
    def code_page(release: str, code: str) -> CodeView:
        """A Tabular code: its place, 7th characters, notes, edges both ways, Index terms."""
        served(release)
        try:
            return code_view(graph, code)
        except KeyError as exc:
            raise HTTPException(404, f"{release} has no code {code}") from exc

    @app.get("/v1/{release}/terms/{term}", tags=["graph"])
    def term_page(release: str, term: Annotated[int, Path(ge=0)]) -> TermView:
        """An Index term: its ancestors, subterms, codes, and cross-references both ways."""
        served(release)
        try:
            return term_view(graph, term)
        except (KeyError, IndexError) as exc:
            raise HTTPException(404, f"{release} has no term {term}") from exc

    @app.get("/v1/{release}/search", tags=["graph"])
    def search(
        release: str,
        q: Annotated[str, Query(min_length=1, max_length=200)],
        book: Book | None = None,
        k: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> SearchView:
        """Index terms (of one book, or all) and Tabular texts matching the words of `q`."""
        served(release)
        return search_view(graph, q, book, k)

    @app.get("/healthz", tags=["service"])
    def health() -> Health:
        return Health(
            status="ok",
            release=graph.release,
            graph=graph.meta.get("content_sha256", ""),
            fts5=fts5(),
        )

    @app.get("/version", tags=["service"])
    def version() -> dict[str, object]:
        """What the resolver runs: release, graph hash, calibration, L2 model, L3 model."""
        return resolver.info()

    return app
