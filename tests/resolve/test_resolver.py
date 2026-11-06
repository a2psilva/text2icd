"""Tasks 9.1 to 9.3: the resolver builds the cascade from the settings, codes a phrase with the
setting and episode it is given, caches results in memory, and returns the public result."""

import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.embeddings import Vectors, normalize
from text2icd.graph.model import Graph
from text2icd.resolve import retrieve
from text2icd.resolve.confidence import Calibration, RetrievalStage, read_calibration
from text2icd.resolve.resolver import MAX_PHRASE, Resolver, ResolverError
from text2icd.resolve.retrieve import LazyRetriever, build_vectors
from text2icd.result import CodingResult, schema

SCHEMA = Path(__file__).resolve().parents[2] / "docs" / "schema" / "coding-result.json"
PNEUMONIA = "aspiration pneumonia due to food"
PROPANOL = "poisoning by propanol"


class Words:
    """A stand-in embedder: hashed word counts."""

    model = "test/words"
    dim = 64

    def encode(self, batch: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(batch), 64), dtype=np.float32)
        for row, text in enumerate(batch):
            for word in text.casefold().replace(",", " ").split():
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return normalize(vectors)


def staged(model: str = "test/words") -> Calibration:
    """The shipped table with a retrieval stage fitted for `model`: every phrase escalates."""
    shipped = read_calibration()
    assert shipped is not None
    stage = RetrievalStage.model_validate(
        {
            "features": ("retrieval_first", "coder"),
            "weights": (4.0, 1.0),
            "intercept": -2.0,
            "knots": (-2.0, 0.0, 2.0),
            "values": (0.1, 0.5, 0.9),
            "threshold": 0.95,
            "model": model,
        }
    )
    return shipped.model_copy(update={"threshold": 1.0, "retrieval": stage})


@pytest.fixture
def resolver(graph: Graph) -> Resolver:
    return Resolver(graph=graph)


def test_code_returns_the_public_result(resolver: Resolver, graph: Graph) -> None:
    result = resolver.code(PNEUMONIA)
    assert isinstance(result, CodingResult)
    assert (result.release, result.graph) == ("fy2099", graph.meta["content_sha256"])
    assert [c.code for c in result.codes] == ["J69.0"]
    primary = result.primary
    assert primary is not None
    assert (primary.primary, primary.role, primary.chunks, primary.path) == (
        True,
        "diagnosis",
        (0,),
        0,
    )
    (chunk,) = result.chunks
    assert (chunk.text, chunk.start, chunk.end) == (PNEUMONIA, 0, len(PNEUMONIA))
    assert any(step.code == "J69.0" for step in result.trace)
    assert result.detail is None
    assert result.cached is False


def test_explain_adds_the_paths(resolver: Resolver) -> None:
    result = resolver.code(PNEUMONIA, explain=True)
    assert result.detail is not None
    (path,) = result.detail.paths
    assert path.code == "J69.0"
    assert [s.title for s in path.steps] == ["Pneumonia", "aspiration", "due to", "food"]
    assert path.coded_step == 3
    assert path.chunks == (0,)
    assert path.book == "index"


def test_setting_and_episode_per_phrase(resolver: Resolver) -> None:
    assert resolver.code("possible aspiration pneumonia").codes[0].code == "J69.0"
    outpatient = resolver.code("possible aspiration pneumonia", setting="outpatient")
    assert "J69.0" not in [c.code for c in outpatient.codes]  # IV.H: not coded outpatient
    assert outpatient.setting == "outpatient"
    assert resolver.code(PROPANOL).codes[0].code == "T51.3X1A"
    assert resolver.code(PROPANOL, episode="subsequent").codes[0].code == "T51.3X1D"
    assert sorted(resolver._coders) == ["initial", "subsequent"]
    # One walker for every setting and episode.
    assert resolver.coder("initial").walker is resolver.coder("subsequent").walker


def test_results_are_cached_in_memory(graph: Graph) -> None:
    resolver = Resolver(graph=graph, cache_size=2)
    first = resolver.code(PNEUMONIA)
    again = resolver.code(PNEUMONIA)
    assert (first.cached, again.cached) == (False, True)
    assert again.model_copy(update={"cached": False}) == first
    assert resolver.code(PNEUMONIA, setting="outpatient").cached is False  # another key
    resolver.code(PROPANOL)  # the cache keeps 2: the oldest goes
    assert resolver.code(PNEUMONIA).cached is False
    off = Resolver(graph=graph, cache_size=0)
    off.code(PNEUMONIA)
    assert off.code(PNEUMONIA).cached is False


def test_long_input_is_refused(resolver: Resolver) -> None:
    with pytest.raises(ResolverError, match="at most"):
        resolver.code("pneumonia " * (MAX_PHRASE // 10 + 1))


def test_from_many_threads(resolver: Resolver) -> None:
    phrases = [PNEUMONIA, PROPANOL, "possible aspiration pneumonia"] * 10
    expected = {p: Resolver(graph=resolver.graph, cache_size=0).code(p) for p in set(phrases)}
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda p: resolver.code(p, setting="inpatient"), phrases))
    for phrase, result in zip(phrases, got, strict=True):
        assert result.model_copy(update={"cached": False}) == expected[phrase]


def test_retrieval_is_off_without_vectors(graph: Graph) -> None:
    resolver = Resolver(graph=graph, calibration=staged())
    assert resolver.retriever is None
    assert resolver.warnings == []
    with pytest.raises(ResolverError, match="does not exist"):
        Resolver(graph=graph, calibration=staged(), retrieval=True)
    unstaged = staged().model_copy(update={"retrieval": None})
    with pytest.raises(ResolverError, match="retrieval stage"):
        Resolver(graph=graph, calibration=unstaged, retrieval=True)


def test_retrieval_loads_on_first_use(graph: Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    build_vectors(graph, Words(), "test:words")
    monkeypatch.setattr(retrieve, "load_embedder", lambda spec: Words())
    resolver = Resolver(graph=graph, calibration=staged())
    assert isinstance(resolver.retriever, LazyRetriever)
    assert not resolver.retriever.loaded  # nothing read yet
    assert resolver.info()["retrieval"] == "test/words"
    result = resolver.code(PNEUMONIA, explain=True)
    assert resolver.retriever.loaded
    assert result.escalated[0] == "index"
    assert result.detail is not None
    assert result.detail.pool  # the codes L2 weighed
    # Another model's stage does not fit these vectors.
    with pytest.raises(ValueError, match="fitted with"):
        Resolver(graph=graph, calibration=staged("other/model"))


def test_info_and_lifecycle(graph_file: Path, graph: Graph) -> None:
    info = Resolver(graph=graph).info()
    assert info["release"] == "fy2099"
    assert info["calibration"] == "shipped"
    assert info["llm"] is None
    with Resolver(settings=Settings(), graph=graph):
        pass
    assert graph.search_index("pneumonia")  # a graph passed in stays open


def test_llm_needs_a_provider(graph: Graph) -> None:
    with pytest.raises(ResolverError, match="provider and a model"):
        Resolver(graph=graph, llm=True)


def test_schema_file_is_current() -> None:
    """docs/schema/coding-result.json is `python -m text2icd.result`."""
    assert json.loads(SCHEMA.read_text(encoding="utf-8")) == schema()


def test_code_command_prints_the_result(graph_file: Path) -> None:
    runner = CliRunner()
    args = ["code", PNEUMONIA, "--graph", str(graph_file), "--json"]
    plain = runner.invoke(cli.app, args)
    assert plain.exit_code == 0, plain.output
    result = CodingResult.model_validate_json(plain.output)
    assert [c.code for c in result.codes] == ["J69.0"]
    assert result.detail is None
    explained = runner.invoke(cli.app, [*args, "--explain"])
    assert CodingResult.model_validate_json(explained.output).detail is not None
    text = runner.invoke(cli.app, ["code", PNEUMONIA, "--graph", str(graph_file)])
    assert "J69.0" in text.output
    refused = runner.invoke(cli.app, ["code", PNEUMONIA, "--graph", str(graph_file), "--llm"])
    assert refused.exit_code == 1
    assert "provider and a model" in refused.output
