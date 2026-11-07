"""Task 9.6: the same phrase, settings, graph and package give the same result, byte for byte, in
any process. Each run below is a fresh interpreter with its own hash seed, so set and dict orders
that depend on string hashes would show up as a difference."""

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.eval.datasets import read_dataset
from text2icd.graph.build import BuildReport
from text2icd.graph.model import load
from text2icd.resolve.resolver import Resolver

SAMPLE = Path(__file__).resolve().parents[1] / "benchmarks/cms-synthetic/fy2027-sample.jsonl"
PHRASES = [
    "aspiration pneumonia due to food",
    "possible aspiration pneumonia",
    "poisoning by propanol",
    "viral pneumonia, bacterial pneumonia",
    "hx of pneumonia",
    "no pneumonia",
    "lung abscess with pneumonia",
    "tuberculosis of kidney",
    "DM2 w/ CKD 3a, possible UTI 2/2 E. coli",
]
# Prints one SHA-256 per result, with every field the result has (`explain` included).
RUN = """
import hashlib, json, sys
from pathlib import Path
from text2icd.graph.model import load
from text2icd.resolve.resolver import Resolver
phrases = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
with load(path=Path(sys.argv[1])) as graph, Resolver(graph=graph, retrieval=False) as resolver:
    for result in resolver.code_many(phrases, explain=True):
        print(hashlib.sha256(result.model_dump_json().encode()).hexdigest())
"""


def digests(graph: Path, phrases: list[str], seed: str, tmp_path: Path) -> list[str]:
    """The result digests of a fresh interpreter run with this hash seed; in `tmp_path`, so no
    `.env` is read."""
    file = tmp_path / "phrases.json"
    file.write_text(json.dumps(phrases), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-c", RUN, str(graph), str(file)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    return done.stdout.split()


def here(graph: Path, phrases: list[str]) -> list[str]:
    with load(path=graph) as loaded, Resolver(graph=loaded, retrieval=False) as resolver:
        return [
            hashlib.sha256(r.model_dump_json().encode()).hexdigest()
            for r in resolver.code_many(phrases, explain=True)
        ]


def test_same_result_in_any_process(graph_file: Path, tmp_path: Path) -> None:
    expected = here(graph_file, PHRASES)
    assert len(expected) == len(PHRASES)
    for seed in ("0", "1", "4242"):
        assert digests(graph_file, PHRASES, seed, tmp_path) == expected, seed


@pytest.mark.slow
def test_same_result_in_any_process_on_the_sample(
    release_graph: Callable[[str], BuildReport], tmp_path: Path
) -> None:
    """The committed sample (1,067 phrases) on the FY2027 graph."""
    graph = release_graph("fy2027").path
    phrases = [example.phrase for example in read_dataset(SAMPLE)]
    first = digests(graph, phrases, "1", tmp_path)
    assert len(first) == len(phrases)
    second = digests(graph, phrases, "2", tmp_path)
    differ = [i for i, (a, b) in enumerate(zip(first, second, strict=True)) if a != b]
    assert not differ, f"{len(differ)} results differ, the first for {phrases[differ[0]]!r}"
