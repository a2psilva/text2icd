"""Task 9.10: what a threaded server shares (the graph, the coder, the decision cache, the cost
meter) works from many threads at once and gives the same answers as one thread."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from text2icd.graph.model import Graph, GraphError, load
from text2icd.llm.cache import FILE, DecisionCache
from text2icd.llm.cost import CostMeter, Usage, load_prices
from text2icd.llm.decisions import Decision, Offer
from text2icd.resolve import words
from text2icd.resolve.coder import Coder
from text2icd.resolve.words import Speller, WordForms

THREADS = 8
PHRASES = (
    "type 2 diabetes mellitus with diabetic chronic kidney disease",
    "aspiration pneumonia due to food",
    "diabetes due to underlying condition with retinopathy",
    "pneumonia",
    "poisoning by propanol",
    "something no index term names at all",
)


def searches(graph: Graph) -> list[object]:
    return [graph.search_index(p, 10, book=b) for p in PHRASES for b in (None, "drug")]


def test_graph_searches_from_many_threads(graph: Graph) -> None:
    expected = searches(graph)
    with ThreadPoolExecutor(THREADS) as pool:
        got = list(pool.map(lambda _: searches(graph), range(40)))
    assert all(result == expected for result in got)
    # Each thread that searched opened its own connection.
    assert len(graph._opened) > 1


def test_close_from_another_thread(graph_file: Path) -> None:
    graph = load(path=graph_file)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: graph.search_tabular("propanol"), range(8)))
    closer = threading.Thread(target=graph.close)
    closer.start()
    closer.join()
    assert graph._opened == []
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        graph.search_index("pneumonia")


def test_a_file_that_is_not_a_graph(tmp_path: Path) -> None:
    bad = tmp_path / "graph.sqlite"
    bad.write_bytes(b"no database here" * 200)
    with pytest.raises(GraphError, match="not a readable graph file"):
        load(path=bad)


def test_coder_from_many_threads(graph: Graph) -> None:
    coder = Coder.from_graph(graph)
    expected = [coder.code(p).model_dump_json() for p in PHRASES]
    with ThreadPoolExecutor(THREADS) as pool:
        got = list(pool.map(lambda p: coder.code(p).model_dump_json(), PHRASES * 20))
    assert got == expected * 20


def decision(i: int) -> Decision:
    return Decision.of("subterm", f"phrase {i}", [Offer("Amyloid heart", "E85.4"), Offer("Other")])


def test_decision_cache_from_many_threads(tmp_path: Path) -> None:
    with DecisionCache(tmp_path / FILE) as cache:

        def work(i: int) -> str | None:
            cache.put(decision(i), "space", "1" if i % 2 else None)
            choice = cache.get(decision(i), "space")
            assert choice is not None
            return choice.option

        with ThreadPoolExecutor(THREADS) as pool:
            options = list(pool.map(work, range(200)))
        assert options == ["1" if i % 2 else None for i in range(200)]
        assert len(cache) == 200
        mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_two_caches_on_one_file(tmp_path: Path) -> None:
    """Two processes (here two connections) can share the file: one writes, the other reads."""
    with DecisionCache(tmp_path / FILE) as first, DecisionCache(tmp_path / FILE) as second:
        first.put(decision(1), "space", "1")
        choice = second.get(decision(1), "space")
    assert choice is not None
    assert choice.option == "1"


def test_cost_meter_loses_no_charge() -> None:
    meter = CostMeter(load_prices().get("anthropic", "claude-haiku-5-5"), budget_usd=1e9)
    usage = Usage(input_tokens=300, output_tokens=8)
    with ThreadPoolExecutor(THREADS) as pool:
        costs = list(pool.map(lambda _: meter.charge(usage), range(2000)))
    assert meter.calls == 2000
    assert meter.usage.input_tokens == 300 * 2000
    assert meter.spent_usd == pytest.approx(sum(costs))


def test_memos_stay_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(words, "FORMS_CACHE", 10)
    monkeypatch.setattr(words, "SPELLING_CACHE", 10)
    forms = WordForms({"renal": "kidney"}, {})
    speller = Speller({"pneumonia": 5, "diabetes": 3})
    for i in range(100):
        forms.form(f"word{i}")
        speller.correct(f"unknownword{i}")
    assert len(forms._forms) <= 10
    assert len(speller._cache) <= 10
    assert forms.form("renal") == "kidney"
    assert speller.correct("pnuemonia") == "pneumonia"
