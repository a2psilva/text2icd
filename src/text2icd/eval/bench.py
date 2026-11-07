"""Speed of a system (task 9.8): start-up, memory and latency per phrase, on one thread.

`text2icd eval bench` loads the graph, builds the system as `eval run` does, and codes the
phrases one at a time, timing each. The first phrase is reported apart, since it pays for what
the graph builds lazily. Times are wall-clock on the machine it runs on, so compare runs made on
one machine. Only example ids and numbers are reported, never a phrase.
"""

import math
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from text2icd.config import Settings
from text2icd.eval.datasets import Example
from text2icd.eval.metrics import latency_summary
from text2icd.eval.runner import predict
from text2icd.eval.systems import Options, SystemSetupError, build_system
from text2icd.graph.model import load

SLOWEST = 5
# What a library user imports to code phrases, timed in a fresh interpreter.
_IMPORT = (
    "import time; began = time.perf_counter(); import text2icd.resolve.coder; "
    "print(time.perf_counter() - began)"
)


class Bench(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str
    options: dict[str, str]
    release: str
    dataset: str
    phrases: int  # examples coded per round
    rounds: int
    errors: int
    import_s: float | None  # None when not measured
    graph_s: float
    setup_s: float  # building the system: walker, verifier, calibration, L2 when on
    first_ms: float  # the first phrase coded
    latency_ms: dict[str, float]  # every phrase after the first: p50, p95, p99, max
    phrases_per_s: float  # phrases after the first, over the time spent coding them
    rss_mb: dict[str, float]  # peak resident memory after setup and at the end
    slowest: tuple[tuple[str, float], ...]  # example ids with their slowest time, in ms

    def lines(self) -> list[str]:
        options = "".join(f" -O {k}={v}" for k, v in sorted(self.options.items()))
        imported = "" if self.import_s is None else f"import {self.import_s:.2f} s, "
        latency = " / ".join(f"{k} {v:.1f}" for k, v in self.latency_ms.items())
        slowest = ", ".join(f"{id_} {ms:.1f} ms" for id_, ms in self.slowest)
        return [
            f"{self.system}{options} on {self.dataset}, {self.release}: "
            f"{self.phrases:,} phrases x {self.rounds}, {self.errors} errors",
            f"  start-up: {imported}graph {self.graph_s:.2f} s, setup {self.setup_s:.2f} s, "
            f"first phrase {self.first_ms:.1f} ms",
            f"  memory: {self.rss_mb['setup']:.0f} MB peak after setup, "
            f"{self.rss_mb['end']:.0f} MB at the end",
            f"  latency ms: {latency}; {self.phrases_per_s:,.0f} phrases/s on one thread",
            f"  slowest: {slowest}",
        ]


def bench(
    graph_path: Path,
    system: str,
    settings: Settings,
    options: Options,
    examples: Sequence[Example],
    dataset: str,
    rounds: int = 1,
    imports: bool = True,
) -> Bench:
    """Time `system` over `examples`, `rounds` times. Systems that call an LLM are refused: a
    benchmark should not spend money, and their time is the provider's."""
    if not examples:
        raise ValueError("no examples to time")
    import_s = import_seconds() if imports else None
    began = time.perf_counter()
    with load(path=graph_path) as graph:
        graph_s = time.perf_counter() - began
        began = time.perf_counter()
        built = build_system(system, graph, settings, options)
        built.warm()  # L2 loads on its first phrase otherwise
        setup_s = time.perf_counter() - began
        rss_setup = peak_rss_mb()
        try:
            if built.uses_llm:
                raise SystemSetupError("eval bench times local systems only; this one calls an LLM")
            predictions = [p for _ in range(rounds) for p in predict(built, examples)]
        finally:
            built.close()
        release = graph.release
    latencies = [p.latency_ms for p in predictions]
    warm = latencies[1:] or latencies
    slowest: dict[str, float] = {}
    for prediction in predictions[1:]:
        slowest[prediction.id] = max(prediction.latency_ms, slowest.get(prediction.id, 0.0))
    top = sorted(slowest.items(), key=lambda item: -item[1])[:SLOWEST]
    return Bench(
        system=system,
        options=dict(options),
        release=release,
        dataset=dataset,
        phrases=len(examples),
        rounds=rounds,
        errors=sum(p.error is not None for p in predictions),
        import_s=import_s,
        graph_s=graph_s,
        setup_s=setup_s,
        first_ms=latencies[0],
        latency_ms=latency_summary(warm),
        phrases_per_s=len(warm) / (sum(warm) / 1000) if sum(warm) else 0.0,
        rss_mb={"setup": rss_setup, "end": peak_rss_mb()},
        slowest=tuple((id_, round(ms, 3)) for id_, ms in top),
    )


def import_seconds() -> float | None:
    """Seconds to import the coder in a fresh interpreter; None if that fails."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", _IMPORT], capture_output=True, text=True, check=True, timeout=120
        )
        return float(done.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (getrusage counts bytes on macOS and
    kibibytes on Linux); NaN where there is no getrusage (Windows)."""
    try:
        import resource
    except ImportError:
        return math.nan
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak * 1024 / 1e6
