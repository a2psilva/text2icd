from collections.abc import Sequence
from typing import Any

import pytest

from text2icd.graph.model import Graph
from text2icd.resolve import choice
from text2icd.resolve.coder import Coder, describe_code_set
from text2icd.resolve.confidence import (
    RETRIEVAL_FEATURES,
    Calibration,
    CalibrationError,
    RetrievalStage,
)
from text2icd.resolve.retrieve import Hits
from text2icd.resolve.tabular import Completion

DIABETES = "type 2 diabetes mellitus with diabetic chronic kidney disease"  # the coder: N11.9


class Fixed:
    """A retriever that finds the same nodes for every phrase."""

    name = "test/fixed"

    def __init__(self, hits: Hits) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(self, queries: Sequence[str], k: int) -> list[Hits]:
        self.queries += queries
        return [list(self.hits[:k]) for _ in queries]


def stage(**changes: Any) -> RetrievalStage:
    """Retrieval's first code wins unless changed."""
    data: dict[str, Any] = {
        "features": ("retrieval_first", "coder"),
        "weights": (4.0, 1.0),
        "intercept": -2.0,
        "knots": (-2.0, 0.0, 2.0),
        "values": (0.1, 0.5, 0.9),
        "threshold": 0.6,
        "model": "test/fixed",
    }
    data.update(changes)
    return RetrievalStage.model_validate(data)


def table(retrieval: RetrievalStage | None) -> Calibration:
    return Calibration.model_validate(
        {
            "features": ("coverage", "margin"),
            "weights": (2.0, 1.0),
            "intercept": -1.0,
            "knots": (-1.0, 0.0, 2.0),
            "values": (0.1, 0.5, 0.9),
            "threshold": 0.6,
            "target": 0.98,
            "retrieval": retrieval,
        }
    )


def test_options() -> None:
    completions = {"n1": "C10", "n2": "A10"}  # n3 does not complete

    def complete(node: str) -> Completion | None:
        code = completions.get(node)
        return Completion(code=code, description=f"{code} title") if code else None

    pool = choice.options(
        ["A10", "B10"],
        0.3,
        [("n1", 0.9), ("n2", 0.8), ("n3", 0.5)],
        complete,
        lambda code: {"x"} if code == "A10" else set(),
        lambda code: "Thing, unspecified",
        frozenset({"x", "y"}),
    )
    assert [o.code for o in pool] == ["A10", "B10", "C10"]  # the coder's first
    a, b, c = (o.features for o in pool)
    assert set(a) == set(RETRIEVAL_FEATURES)
    assert (a["coder"], a["coder_rank"], a["retrieval_rank"], a["similarity"]) == (1, 1, 0.5, 0.8)
    assert (a["coverage"], a["coder_coverage"], a["coder_confidence"]) == (0.5, 0.5, 0.3)
    assert (b["coder"], b["coder_rank"], b["not_retrieved"], b["similarity"]) == (0, 0.5, 1, 0.5)
    assert b["unspecified"] == 1.0  # described by the graph: not retrieved
    assert (c["retrieval_first"], c["similarity_gap"], c["coder_confidence"]) == (1, 0, 0)
    assert c["unspecified"] == 0.0  # described by its completion
    assert a["top_margin"] == pytest.approx(0.1)
    assert a["agreement"] == 0.0  # retrieval's first code is not the coder's
    assert pool[2].node == "n1"
    assert pool[2].similarity == 0.9
    assert not pool[2].index
    assert pool[1].node is None
    assert pool[1].index
    assert not pool[1].coder

    nothing = choice.options([], 0.0, [], complete, lambda c: set(), lambda c: "", frozenset())
    assert nothing == []
    assert choice.choose(nothing, stage()) is None


def test_choose_prefers_the_first_of_equals() -> None:
    pool = choice.options(
        ["A10"],
        0.0,
        [("A10", 0.9)],
        lambda node: Completion(code=node, description=node),
        lambda code: set(),
        lambda code: code,
        frozenset(),
    )
    flat = stage(weights=(0.0, 0.0))
    found = choice.choose(pool, flat)
    assert found is not None
    assert found[0].code == "A10"
    assert found[1] == 0.1


def test_coder_takes_a_retrieved_code(graph: Graph) -> None:
    retriever = Fixed([("E11.22", 0.91), ("J69.0", 0.4)])
    coder = Coder.from_graph(graph, calibration=table(stage()), retriever=retriever)
    code_set = coder.code(DIABETES)
    assert retriever.queries == [code_set.parsed.normalized]
    assert code_set.codes[0] == "E11.22"
    assert code_set.resolver == "retrieval"
    assert (code_set.confidence, code_set.escalate, code_set.escalated) == (0.9, False, ("index",))
    assert code_set.choice["retrieval_first"] == 1.0
    assert code_set.diagnoses[0].candidate.book == choice.BOOK
    assert choice.RETRIEVED in code_set.diagnoses[0].flags
    assert code_set.features  # the coder's features of the new set are still read
    lines = "\n".join(describe_code_set(code_set))
    assert "confidence: 0.90 after retrieval" in lines
    assert "retrieved: " in lines
    assert 'retrieval/retrieved: E11.22 "' in lines
    assert "retrieval/choice: confidence 0.51 below 0.60: a retrieved code chosen" in lines
    assert "the coder's N11.9 0.30" in lines

    low = Coder.from_graph(graph, calibration=table(stage(threshold=0.95)), retriever=retriever)
    still = low.code(DIABETES)
    assert (still.escalate, still.escalated) == (True, ("index", "retrieval"))


def test_coder_keeps_its_code(graph: Graph) -> None:
    keep = stage(weights=(0.0, 4.0))
    coder = Coder.from_graph(graph, calibration=table(keep), retriever=Fixed([("E11.22", 0.9)]))
    code_set = coder.code(DIABETES)
    assert code_set.codes == ("N11.9",)
    assert (code_set.resolver, code_set.confidence, code_set.escalate) == ("index", 0.9, False)
    assert code_set.trace[-1].layer == "retrieval"
    assert "the coder's code kept, N11.9 0.90 of 3 codes" in code_set.trace[-1].detail


def test_retrieval_codes_an_uncoded_phrase(graph: Graph) -> None:
    coder = Coder.from_graph(graph, calibration=table(stage()), retriever=Fixed([("J69.0", 0.8)]))
    code_set = coder.code("pneumonia")
    assert code_set.codes == ("J69.0",)
    assert code_set.resolver == "retrieval"
    assert code_set.choice["uncoded"] == 1.0


def test_retrieval_runs_only_when_escalated_with_a_stage(graph: Graph) -> None:
    retriever = Fixed([("E11.22", 0.9)])
    coder = Coder.from_graph(graph, calibration=table(stage()), retriever=retriever)
    assert coder.code("aspiration pneumonia").codes == ("J69.0",)  # confident: no retrieval
    assert retriever.queries == []
    plain = Coder.from_graph(graph, calibration=table(None), retriever=retriever)
    code_set = plain.code(DIABETES)
    assert (code_set.codes, code_set.escalated, code_set.choice) == (("N11.9",), ("index",), {})
    assert retriever.queries == []
    assert plain.options(code_set)  # the pool, for fitting the stage
    with pytest.raises(CalibrationError, match="fitted with other/model"):
        Coder.from_graph(graph, calibration=table(stage(model="other/model")), retriever=retriever)


def test_retrieval_never_codes_what_the_phrase_rules_out(graph: Graph) -> None:
    retriever = Fixed([("J69.0", 0.95)])
    coder = Coder.from_graph(graph, calibration=table(stage()), retriever=retriever)
    code_set = coder.code("no pneumonia")
    assert code_set.codes == ()
    assert code_set.escalate
    assert retriever.queries == []  # nothing to code: no retrieval
