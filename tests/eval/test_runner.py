import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.eval import runner as runner_module
from text2icd.eval.datasets import Example, write_dataset, write_manifest
from text2icd.eval.metrics import report_to_json, score
from text2icd.eval.predictions import Prediction
from text2icd.eval.report import by_group, comparison, fmt, selective
from text2icd.eval.runner import (
    RunError,
    chapter_of,
    evaluate,
    git_state,
    is_restricted,
    predict,
    read_metrics,
    read_run,
    resolve_dataset,
    select,
    write_run,
)
from text2icd.eval.systems import BM25System, System, SystemSetupError, build_system
from text2icd.graph.model import Graph

DIABETES = "type 2 diabetes mellitus with diabetic chronic kidney disease"


def example(i: int, phrase: str, code: str, split: str = "dev", **changes: Any) -> Example:
    data: dict[str, Any] = {
        "id": f"x{i:02d}",
        "phrase": phrase,
        "gold": [{"code": code, "role": "primary"}],
        "release": "fy2099",
        "source": "test",
        "split": split,
    }
    data.update(changes)
    return Example.model_validate(data)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "set.jsonl"
    examples = [
        example(1, DIABETES, "E11.22"),
        example(2, "diabetes due to underlying condition with retinopathy", "E08.3211"),
        example(3, "aspiration pneumonia", "J69.0", split="test"),
    ]
    write_dataset(path, examples)
    write_manifest(path, examples, name="tiny")
    return path


def test_bm25(graph: Graph) -> None:
    system = BM25System(graph, {})
    prediction = system.predict(DIABETES)
    assert prediction.primary == "E11.22"
    assert all(graph.code(code).billable for code in prediction.ranked)
    assert 0 < (prediction.confidence or 0) <= 1
    assert system.predict("zzzz").codes == ()
    assert system.settings() == {"k": 50, "top": 5, "tokenizer": "porter"}
    assert BM25System(graph, {"top": "1"}).predict(DIABETES).alternates == ()


def test_build_system(graph: Graph) -> None:
    assert build_system("bm25", graph, Settings(), {"k": "10"}).settings()["k"] == 10
    with pytest.raises(SystemSetupError, match="unknown system 'nope'; known systems: bm25"):
        build_system("nope", graph, Settings(), {})
    with pytest.raises(SystemSetupError, match="unknown options for bm25: model"):
        build_system("bm25", graph, Settings(), {"model": "x"})
    with pytest.raises(SystemSetupError, match="option k must be an integer"):
        build_system("bm25", graph, Settings(), {"k": "many"})


class Flaky(System):
    name = "flaky"

    def predict(self, phrase: str) -> Prediction:
        if "bad" in phrase:
            raise RuntimeError("cannot")
        return Prediction(id="", codes=("E11.22",), latency_ms=5.0)


class Short(System):
    name = "short"

    def predict(self, phrase: str) -> Prediction:
        return Prediction(id="")

    def predict_many(self, phrases: Sequence[str]) -> Iterator[Prediction]:
        yield Prediction(id="", codes=("E11.22",))


def test_predict_keeps_going_after_an_error() -> None:
    examples = [example(1, "good", "E11.22"), example(2, "bad", "E11.22"), example(3, "ok", "A00")]
    seen = []
    out = predict(Flaky(), examples, progress=seen.append)
    assert [p.id for p in out] == ["x01", "x02", "x03"]
    assert out[1].error == "RuntimeError: cannot"
    assert out[2].codes == ("E11.22",)
    assert out[0].latency_ms == 5.0
    assert len(seen) == 3
    short = predict(Short(), examples)
    assert short[1].error == "no prediction returned"


def test_select() -> None:
    examples = [example(i, f"p{i}", "E11.22", split="dev" if i % 3 else "test") for i in range(30)]
    dev = select(examples, "dev", None, 0)
    assert len(dev) == 20
    assert all(e.split == "dev" for e in dev)
    assert len(select(examples, "all", None, 0)) == 30
    picked = select(examples, "dev", 5, 0)
    assert len(picked) == 5
    assert picked == select(examples, "dev", 5, 0)
    assert [e.id for e in picked] == sorted(e.id for e in picked)
    assert picked != select(examples, "dev", 5, 1)


def test_resolve_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    named = settings.eval_dir() / "mimic-silver.jsonl"
    versioned = settings.eval_dir() / "cms-synthetic-fy2027.jsonl"
    named.parent.mkdir(parents=True)
    named.write_text("")
    versioned.write_text("")
    monkeypatch.setattr(runner_module, "BENCHMARKS", tmp_path / "benchmarks")
    committed = tmp_path / "benchmarks" / "cms-synthetic" / "fy2027-sample.jsonl"
    committed.parent.mkdir(parents=True)
    committed.write_text("")
    assert resolve_dataset(str(named), "fy2027", settings) == named
    assert resolve_dataset("mimic-silver", "fy2027", settings) == named
    assert resolve_dataset("cms-synthetic", "fy2027", settings) == versioned
    assert resolve_dataset("cms-synthetic-sample", "fy2027", settings) == committed
    with pytest.raises(RunError, match="no dataset 'other'"):
        resolve_dataset("other", "fy2027", settings)


def test_evaluate_and_result_files(graph: Graph, dataset: Path, tmp_path: Path) -> None:
    run, report = evaluate(BM25System(graph, {}), graph, dataset, resamples=50)
    header = run.header
    assert header["release"] == "fy2099"
    assert header["dataset"]["name"] == "tiny"
    assert header["dataset"]["split"] == "dev"
    assert header["dataset"]["examples"] == 2
    assert header["dataset"]["restricted"] is False
    assert header["system"] == {
        "name": "bm25",
        "uses_llm": False,
        "k": 50,
        "top": 5,
        "tokenizer": "porter",
    }
    assert set(header["git"]) == {"commit", "dirty"}
    assert [p.id for p in run.predictions] == ["x01", "x02"]
    assert report.metrics["accuracy@1"].value == 0.5
    assert list(report.groups) == ["4"]

    path = tmp_path / "results" / "run.jsonl"
    metrics = write_run(path, run, report)
    assert metrics.name == "run.metrics.json"
    assert read_run(path).predictions == run.predictions
    for source in (path, metrics):
        loaded_header, loaded = read_metrics(source)
        assert loaded_header == header
        assert report_to_json(loaded) == report_to_json(report)
    assert "phrase" not in path.read_text()

    test_run, _ = evaluate(BM25System(graph, {}), graph, dataset, split="test", resamples=0)
    assert test_run.header["dataset"]["examples"] == 1


def test_evaluate_refuses(graph: Graph, tmp_path: Path) -> None:
    other = tmp_path / "other.jsonl"
    write_dataset(other, [example(1, DIABETES, "E11.22", release="fy2027")])
    with pytest.raises(RunError, match="the dataset is for fy2027 and the graph for fy2099"):
        evaluate(BM25System(graph, {}), graph, other)
    only_dev = tmp_path / "dev.jsonl"
    write_dataset(only_dev, [example(1, DIABETES, "E11.22")])
    with pytest.raises(RunError, match="no test examples"):
        evaluate(BM25System(graph, {}), graph, only_dev, split="test")
    not_results = tmp_path / "plain.jsonl"
    not_results.write_text('{"id": "x"}\n')
    with pytest.raises(RunError, match="not a result file"):
        read_run(not_results)


def test_chapter_and_restriction(graph: Graph) -> None:
    chapter = chapter_of(graph)
    assert chapter(example(1, "x", "E11.22")) == "4"
    mapped = example(
        2,
        "x",
        "E11.2",
        gold=[{"code": "Q99.9", "role": "primary", "status": "mapped", "mapped_to": ["E11.22"]}],
    )
    assert chapter(mapped) == "4"
    assert chapter(example(3, "x", "Z99.99")) == "unknown"
    assert not is_restricted([example(1, "x", "E11.22")])
    assert is_restricted([example(1, "x", "E11.22", source="mimic-silver")])


def test_git_state_outside_a_repository(tmp_path: Path) -> None:
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def test_markdown(graph: Graph, dataset: Path) -> None:
    run, report = evaluate(BM25System(graph, {}), graph, dataset, resamples=50)
    lines = comparison([(run.header, report)])
    assert lines[0].startswith("| System | Dataset | n | Acc@1 | Acc@5 |")
    assert lines[2].startswith("| bm25 | tiny (dev) | 2 | 50.0% (")
    assert lines[-1].startswith("- bm25: fy2099, commit ")
    plain = comparison([(run.header, report)], intervals=False)
    assert "| 50.0% |" in plain[2]
    chapters = by_group(run.header, report)
    assert chapters[0] == "bm25 on tiny"
    assert chapters[-1].startswith("| 4 | 2 | 50.0%")
    shown = selective([(run.header, report)])
    assert shown[0].startswith("| System | Dataset | 100% kept | 90% kept | 80% kept |")
    assert shown[2].startswith("| bm25 | tiny (dev) | 50.0% |")
    examples = [example(1, DIABETES, "E11.22"), example(2, "x", "V29.88XA")]
    unscored = score(examples, {}, lambda code: True, resamples=0)
    assert selective([(run.header, unscored)]) == []  # no confidence, no table
    assert fmt(unscored.metrics["accuracy@1"], "share") == "0.0%"
    assert fmt(None, "share") == "n/a"


cli_runner = CliRunner()


def test_run_and_report_commands(graph_file: Path, dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / "run.jsonl"
    args = ["eval", "run", "-d", str(dataset), "-s", "bm25", "--graph", str(graph_file)]
    args += ["-o", str(output), "--resamples", "20", "-O", "k=20"]
    result = cli_runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "| bm25 | tiny (dev) | 2 |" in result.output
    header = json.loads(output.read_text().splitlines()[0])["run"]
    assert header["system"]["k"] == 20

    report_file = tmp_path / "report.md"
    shown = cli_runner.invoke(cli.app, ["eval", "report", str(output), "--by-chapter"])
    assert shown.exit_code == 0, shown.output
    assert "bm25 on tiny" in shown.output
    written = cli_runner.invoke(cli.app, ["eval", "report", str(output), "-o", str(report_file)])
    assert written.exit_code == 0
    assert report_file.read_text().startswith("| System |")


def test_run_command_errors(graph_file: Path, dataset: Path, tmp_path: Path) -> None:
    base = ["eval", "run", "-d", str(dataset), "--graph", str(graph_file)]
    unknown = cli_runner.invoke(cli.app, [*base, "-s", "nope"])
    assert unknown.exit_code == 1
    assert "unknown system 'nope'" in unknown.output
    bad_option = cli_runner.invoke(cli.app, [*base, "-s", "bm25", "-O", "k"])
    assert bad_option.exit_code == 2
    missing = cli_runner.invoke(
        cli.app, ["eval", "run", "-d", "nothing", "-s", "bm25", "--graph", str(graph_file)]
    )
    assert missing.exit_code == 1
    assert "no dataset 'nothing'" in missing.output
    no_report = cli_runner.invoke(cli.app, ["eval", "report", str(tmp_path / "none.jsonl")])
    assert no_report.exit_code == 1
