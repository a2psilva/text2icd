import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.embeddings import Vectors, normalize
from text2icd.eval import retrieval
from text2icd.eval.datasets import Example, write_dataset, write_manifest
from text2icd.graph.model import Graph
from text2icd.resolve import retrieve
from text2icd.resolve.coder import Coder
from text2icd.resolve.confidence import Calibration


class Words:
    """A stand-in embedder: hashed word counts, so texts sharing words are similar."""

    model = "test/words"
    size_bytes = 1234

    def encode(self, batch: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(batch), 64), dtype=np.float32)
        for row, text in enumerate(batch):
            for word in text.casefold().replace(",", " ").split():
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return normalize(vectors)


ROWS = [
    ("aspiration pneumonia", "J69.0", "valid"),
    ("amyloid heart disease", "E85.4", "valid"),
    ("type 2 diabetes mellitus with diabetic chronic kidney disease", "E11.22", "valid"),
    ("aspiration pneumonia due to vomitus", "J69.0", "valid"),
    ("pneumonia", "J18.9", "unmappable"),
]


def _examples(source: str = "test") -> list[Example]:
    return [
        Example.model_validate(
            {
                "id": f"x{i}",
                "phrase": phrase,
                "gold": [{"code": code, "role": "primary", "status": status}],
                "release": "fy2099",
                "source": source,
                "split": "dev",
            }
        )
        for i, (phrase, code, status) in enumerate(ROWS)
    ]


def _coder(graph: Graph) -> Coder:
    """Escalates below 0.6: coverage alone, so the diabetes phrase (half of it walked)
    escalates."""
    calibration = Calibration.model_validate(
        {
            "features": ("coverage",),
            "weights": (1.0,),
            "intercept": 0.0,
            "knots": (0.0, 1.0),
            "values": (0.0, 1.0),
            "threshold": 0.6,
            "target": 0.98,
        }
    )
    return Coder.from_graph(graph, calibration=calibration)


def test_queries(graph: Graph) -> None:
    found = retrieval.queries(_coder(graph), _examples())
    assert len(found) == 4  # the unmappable example is not scored
    assert [q.gold.code for q in found] == ["J69.0", "E85.4", "E11.22", "J69.0"]
    assert [q.right for q in found] == [True, True, False, True]  # N11.9 is not E11.22
    assert [q.escalated for q in found] == [False, False, True, False]
    assert found[0].text == "aspiration pneumonia"
    assert "aspiration" in found[0].evidence.words


def test_completed(graph: Graph) -> None:
    (query,) = retrieval.queries(_coder(graph), _examples()[2:3])
    hits = [("E11", 1.0), ("E11.2", 0.9), ("E11.22", 0.8), ("nope", 0.7), ("T51.3", 0.6)]
    codes = retrieval.completed(_coder(graph).verifier, hits, query.evidence)
    assert codes[0] == "E11.22"  # E11 completes to it, then E11.2 and E11.22 repeat it
    assert len(codes) == len(set(codes))
    assert all(graph.code(code).billable for code in codes)
    assert retrieval.completed(_coder(graph).verifier, hits, query.evidence, depth=1) == ["E11.22"]


def test_measure(graph: Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    coder = _coder(graph)
    found = retrieval.queries(coder, _examples())
    lexical = retrieval.measure("bm25", graph, ("tabular", "index"), found, coder.verifier)
    assert lexical.totals == {"all": 4, "escalated": 1, "coder wrong": 1}
    assert (lexical.dim, lexical.vectors_bytes, lexical.model_bytes) == (None, None, None)
    monkeypatch.setattr(retrieval, "load_embedder", lambda spec, revision=None: Words())
    embedded = retrieval.measure("words", graph, ("tabular", "index"), found, coder.verifier)
    assert embedded.dim == 64
    assert embedded.vectors_bytes == embedded.texts * 64 * 2
    assert embedded.model_bytes == 1234
    assert embedded.found["all"][1] <= embedded.found["all"][5] <= embedded.found["all"][20]
    assert embedded.recall("escalated", 1) == 1.0  # E11.22 first for the diabetes phrase
    assert embedded.latency_ms[0] > 0
    monkeypatch.setattr(retrieve, "load_embedder", lambda spec, revision=None: Words())
    retrieve.build_vectors(graph, Words(), "test:words", sources=("tabular",))
    stored = retrieval.measure("vectors", graph, ("index",), found, coder.verifier)
    assert stored.sources == ("tabular",)  # the file's own texts, whatever --texts says
    assert stored.texts == len(retrieve.texts(graph))
    text = "\n".join([*retrieval.summary(found), *retrieval.table([lexical, embedded])])
    assert "escalated: 1 examples (25.0%), coder accuracy@1 0.0%" in text
    assert "coder wrong: 1 examples (25.0%), coder accuracy@1 0.0%" in text
    assert "| words | tabular+index (" in text
    assert "(64 d)" in text


def test_retrieval_command(graph_file: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "set.jsonl"
    examples = _examples()
    write_dataset(dataset, examples)
    write_manifest(dataset, examples, name="tiny")
    args = ["eval", "retrieval", "-d", str(dataset), "--graph", str(graph_file), "-m", "bm25"]
    result = CliRunner().invoke(cli.app, [*args, "--texts", "tabular", "--texts", "tabular,index"])
    assert result.exit_code == 0, result.output
    assert "4 scored examples" in result.output
    assert "| bm25 | tabular (" in result.output
    assert "| bm25 | tabular+index (" in result.output
    bad = CliRunner().invoke(cli.app, [*args, "--texts", "tabular,drugs"])
    assert bad.exit_code == 2
    assert "use tabular, index or both" in bad.output
