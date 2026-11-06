import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.eval import bench as bench_module
from text2icd.eval.bench import Bench, bench, import_seconds, peak_rss_mb
from text2icd.eval.datasets import Example, read_dataset, write_dataset, write_manifest
from text2icd.eval.metrics import latency_summary
from text2icd.eval.systems import SystemSetupError

PHRASES = (
    "type 2 diabetes mellitus with diabetic chronic kidney disease",
    "aspiration pneumonia",
    "diabetes due to underlying condition with retinopathy",
)


def example(i: int, phrase: str) -> Example:
    data: dict[str, Any] = {
        "id": f"x{i:02d}",
        "phrase": phrase,
        "gold": [{"code": "E11.22", "role": "primary"}],
        "release": "fy2099",
        "source": "test",
        "split": "dev",
    }
    return Example.model_validate(data)


EXAMPLES = [example(i, phrase) for i, phrase in enumerate(PHRASES)]


def test_latency_summary() -> None:
    assert latency_summary([]) == {}
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert list(summary) == ["p50", "p95", "p99", "max"]
    assert summary["p50"] == 3.0
    assert summary["max"] == 100.0
    assert summary["p95"] <= summary["p99"] <= summary["max"]


def test_bench_times_every_phrase(graph_file: Path) -> None:
    result = bench(graph_file, "coder", Settings(), {}, EXAMPLES, "tiny (dev)", 2, imports=False)
    assert (result.phrases, result.rounds, result.errors) == (3, 2, 0)
    assert result.release == "fy2099"
    assert result.import_s is None
    assert result.graph_s > 0
    assert result.setup_s > 0
    assert result.first_ms > 0
    assert set(result.latency_ms) == {"p50", "p95", "p99", "max"}
    assert result.phrases_per_s > 0
    # Each example once, with its slowest round; ids only, never the phrase.
    assert sorted(id_ for id_, _ in result.slowest) == ["x00", "x01", "x02"]
    text = "\n".join(result.lines())
    assert "phrases/s" in text
    assert not any(phrase in text for phrase in PHRASES)
    assert Bench.model_validate_json(result.model_dump_json()) == result


def test_bench_refuses_llm_systems(graph_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = bench_module.build_system

    def with_llm(*args: Any) -> Any:
        system = real(*args)
        system.uses_llm = True
        return system

    monkeypatch.setattr(bench_module, "build_system", with_llm)
    with pytest.raises(SystemSetupError, match="local systems only"):
        bench(graph_file, "bm25", Settings(), {}, EXAMPLES, "tiny", imports=False)


def test_bench_needs_examples(graph_file: Path) -> None:
    with pytest.raises(ValueError, match="no examples"):
        bench(graph_file, "coder", Settings(), {}, [], "empty", imports=False)


def test_import_and_memory_probes() -> None:
    seconds = import_seconds()
    assert seconds is not None
    assert 0 < seconds < 60
    assert peak_rss_mb() > 1


def test_bench_command(graph_file: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "set.jsonl"
    write_dataset(dataset, EXAMPLES)
    write_manifest(dataset, EXAMPLES, name="tiny")
    out = tmp_path / "bench.json"
    args = ["eval", "bench", "-d", str(dataset), "--graph", str(graph_file), "--no-import"]
    result = CliRunner().invoke(cli.app, [*args, "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "3 phrases x 1, 0 errors" in result.output
    assert json.loads(out.read_text())["system"] == "coder"


def test_bench_command_reports_a_bad_graph(tmp_path: Path) -> None:
    bad = tmp_path / "graph.sqlite"
    bad.write_bytes(b"not a database, just bytes" * 100)
    result = CliRunner().invoke(cli.app, ["eval", "bench", "--graph", str(bad), "--no-import"])
    assert result.exit_code == 1
    assert "not a readable graph file" in result.output


SAMPLE = Path("benchmarks/cms-synthetic/fy2027-sample.jsonl")


@pytest.mark.slow
@pytest.mark.benchmark
def test_coder_meets_the_targets(release_graph: Any) -> None:
    """Task 9.8 on one core: the deterministic coder at p50 under 20 ms and at least 200
    phrases per second, on the committed sample."""
    report = release_graph("fy2027")
    result = bench(report.path, "coder", Settings(), {}, read_dataset(SAMPLE), "sample")
    assert result.errors == 0
    assert result.latency_ms["p50"] < 20
    assert result.phrases_per_s >= 200
