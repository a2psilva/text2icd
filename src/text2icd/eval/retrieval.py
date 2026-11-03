"""Retrieval benchmark (task 7.4): which local retriever L2 should use, over which texts.
`text2icd eval retrieval`.

The coder codes every scored example first (`Coder.code`). That gives the normalised phrase
(abbreviations expanded) the retrievers read, the evidence Tabular verification completes their
codes with, and whether the coder escalates the phrase. Each retriever then ranks codes for the
normalised phrase (`resolve.retrieve`); the first `NODES` are completed to billable codes as L2
completes them (`TabularVerifier`). Recall@k is the share of examples whose primary gold code is
among the first k distinct completed codes, over three subsets:

- `all` scored examples;
- `escalated`: those the coder marks for escalation, where L2 runs;
- `coder wrong`: those whose primary code the coder gets wrong or gives none.

Also measured: the time to embed the texts, the latency of one phrase (search and completion, one
phrase at a time, after a warm-up), and the size of the vectors in float16 and of the model.
Only aggregates leave this module.
"""

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from text2icd.embeddings import load_embedder
from text2icd.eval.datasets import Example, GoldCode
from text2icd.eval.metrics import matches
from text2icd.eval.mimic.audit import wilson
from text2icd.graph.model import Graph
from text2icd.resolve.coder import Coder
from text2icd.resolve.retrieve import (
    EmbeddingRetriever,
    Hits,
    LexicalRetriever,
    Retriever,
    Source,
    load_vectors,
    texts,
)
from text2icd.resolve.tabular import Evidence, TabularVerifier

DEPTHS = (1, 5, 20)
NODES = 50  # codes retrieved per phrase, completed until max(DEPTHS) billable codes
SUBSETS = ("all", "escalated", "coder wrong")
LATENCY_SAMPLE = 200
_BATCH = 2048  # phrases searched at once
_FLOAT16 = 2


@dataclass(frozen=True, slots=True)
class Query:
    """One scored example as the retrievers see it."""

    text: str  # the normalised phrase
    evidence: Evidence
    gold: GoldCode
    escalated: bool
    right: bool  # the coder's primary code is the gold one


@dataclass(frozen=True, slots=True)
class Measured:
    name: str
    sources: tuple[Source, ...]
    texts: int
    dim: int | None  # embedding dimension; None for BM25
    build_s: float  # embedding the texts
    vectors_bytes: int | None  # the text vectors in float16
    model_bytes: int | None
    latency_ms: tuple[float, float]  # p50, p95 of one phrase
    found: dict[str, dict[int, int]]  # subset -> k -> examples with the gold in the first k
    totals: dict[str, int]  # subset -> examples

    def recall(self, subset: str, k: int) -> float:
        total = self.totals[subset]
        return self.found[subset][k] / total if total else float("nan")


def queries(
    coder: Coder, examples: Sequence[Example], progress: Callable[[int], None] | None = None
) -> list[Query]:
    """Code each scored example (gold codes valid or mapped) and keep what retrieval needs."""
    forms = coder.walker.finder.forms
    out: list[Query] = []
    for example in examples:
        if progress is not None:
            progress(1)
        if any(gold.status == "unmappable" for gold in example.gold):
            continue
        code_set = coder.code(example.phrase)
        parsed = code_set.parsed
        modifiers = tuple(m for condition in parsed.conditions for m in condition.modifiers)
        text = parsed.normalized
        evidence = Evidence(frozenset(forms.content(text)), modifiers, text)
        right = bool(code_set.codes) and matches(code_set.codes[0], example.primary)
        out.append(Query(text, evidence, example.primary, code_set.escalate, right))
    return out


def completed(
    verifier: TabularVerifier, hits: Hits, evidence: Evidence, depth: int = max(DEPTHS)
) -> list[str]:
    """The distinct billable codes the retrieved nodes complete to, in rank order."""
    codes: list[str] = []
    for node, _ in hits:
        found = verifier.complete(node, evidence)
        if found is not None and found.code not in codes:
            codes.append(found.code)
            if len(codes) == depth:
                break
    return codes


def build_retriever(spec: str, graph: Graph, sources: Sequence[Source]) -> Retriever:
    """`bm25`; `vectors`, the graph's vectors file as built (its own texts); or an embedding
    model named as `embeddings.load_embedder` reads it."""
    if spec == "bm25":
        return LexicalRetriever(graph, sources)
    if spec == "vectors":
        return load_vectors(graph)
    return EmbeddingRetriever.embed(load_embedder(spec), texts(graph, sources))


def measure(
    spec: str,
    graph: Graph,
    sources: Sequence[Source],
    found_queries: Sequence[Query],
    verifier: TabularVerifier,
    progress: Callable[[int], None] | None = None,
) -> Measured:
    began = time.perf_counter()
    retriever = build_retriever(spec, graph, sources)
    build_s = time.perf_counter() - began
    found = {subset: dict.fromkeys(DEPTHS, 0) for subset in SUBSETS}
    totals = dict.fromkeys(SUBSETS, 0)
    for start in range(0, len(found_queries), _BATCH):
        batch = found_queries[start : start + _BATCH]
        ranked = retriever.search([q.text for q in batch], NODES)
        for query, hits in zip(batch, ranked, strict=True):
            codes = completed(verifier, hits, query.evidence)
            first = next((i for i, c in enumerate(codes) if matches(c, query.gold)), None)
            for subset in _subsets(query):
                totals[subset] += 1
                for k in DEPTHS:
                    found[subset][k] += first is not None and first < k
        if progress is not None:
            progress(len(batch))
    dim = vectors = model = None
    if isinstance(retriever, EmbeddingRetriever):
        dim = int(retriever.index.vectors.shape[1])
        vectors = len(retriever.index) * dim * _FLOAT16
        model = getattr(retriever.embedder, "size_bytes", None)
    entries = 0
    if isinstance(retriever, EmbeddingRetriever):
        entries, sources = len(retriever.nodes), retriever.sources or sources
    return Measured(
        name=spec,
        sources=tuple(sources),
        texts=entries or len(texts(graph, sources)),
        dim=dim,
        build_s=build_s if dim is not None else 0.0,
        vectors_bytes=vectors,
        model_bytes=model,
        latency_ms=_latency(retriever, verifier, found_queries),
        found=found,
        totals=totals,
    )


def _subsets(query: Query) -> list[str]:
    out = ["all"]
    if query.escalated:
        out.append("escalated")
    if not query.right:
        out.append("coder wrong")
    return out


def _latency(
    retriever: Retriever, verifier: TabularVerifier, found_queries: Sequence[Query]
) -> tuple[float, float]:
    """p50 and p95 of one phrase at a time, over an evenly spread sample."""
    if not found_queries:
        return float("nan"), float("nan")
    step = max(1, len(found_queries) // LATENCY_SAMPLE)
    sample = found_queries[::step][:LATENCY_SAMPLE]
    retriever.search([sample[0].text], NODES)  # warm-up
    times: list[float] = []
    for query in sample:
        began = time.perf_counter()
        completed(verifier, retriever.search([query.text], NODES)[0], query.evidence)
        times.append((time.perf_counter() - began) * 1000)
    if len(times) == 1:
        return times[0], times[0]
    cuts = statistics.quantiles(times, n=20, method="inclusive")
    return statistics.median(times), cuts[18]


def summary(found_queries: Sequence[Query]) -> list[str]:
    """How many examples each subset holds, and the coder's accuracy@1 on it."""
    lines = []
    for subset in SUBSETS:
        chosen = [q for q in found_queries if subset in _subsets(q)]
        right = sum(q.right for q in chosen)
        share = len(chosen) / len(found_queries) if found_queries else float("nan")
        accuracy = right / len(chosen) if chosen else float("nan")
        lines.append(
            f"{subset}: {len(chosen):,} examples ({share:.1%}), coder accuracy@1 {accuracy:.1%}"
        )
    return lines


def table(results: Sequence[Measured]) -> list[str]:
    """A markdown table: recall by subset, latency and size."""
    head = (
        "| Retriever | Texts | Recall@1 / @5 / @20 | Escalated @1 / @5 / @20 | "
        "Coder wrong @1 / @20 | p50 / p95 | Vectors | Model |"
    )
    lines = [head, "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        every = " / ".join(f"{r.recall('all', k):.1%}" for k in DEPTHS)
        low, high = wilson(r.found["all"][20], r.totals["all"])
        escalated = " / ".join(f"{r.recall('escalated', k):.1%}" for k in DEPTHS)
        wrong = f"{r.recall('coder wrong', 1):.1%} / {r.recall('coder wrong', 20):.1%}"
        latency = f"{r.latency_ms[0]:.1f} / {r.latency_ms[1]:.1f} ms"
        lines.append(
            f"| {r.name} | {'+'.join(r.sources)} ({r.texts:,}) | {every} ({low:.1%}-{high:.1%}) "
            f"| {escalated} | {wrong} | {latency} | {_size(r.vectors_bytes, r.dim)} "
            f"| {_size(r.model_bytes)} |"
        )
    return lines


def _size(size: int | None, dim: int | None = None) -> str:
    if size is None:
        return "-"
    text = f"{size / 1e6:.0f} MB" if size >= 1e6 else f"{size / 1e3:.0f} kB"
    return f"{text} ({dim} d)" if dim is not None else text
