"""The evaluation runner: run a system over a dataset and keep the result with its provenance.

A result file is JSONL: the first line is `{"run": {...}}` (git commit, release, graph hash,
dataset hash, split, system settings, times), then one prediction per line, by example id. Result
files hold no phrases, only ids and codes. The metrics are written next to it
(`<run>.metrics.json`), so reports do not need the dataset again.

Results of MIMIC-IV datasets stay under the git-ignored data directory like the datasets.
"""

import json
import platform
import random
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import JsonValue

from text2icd import __version__
from text2icd.config import Settings
from text2icd.eval.datasets import Example, dataset_hash, read_dataset, read_manifest
from text2icd.eval.metrics import Report, report_from_json, report_to_json, score
from text2icd.eval.mimic.splits import FrozenSplitError, check_frozen
from text2icd.eval.predictions import Prediction
from text2icd.eval.systems import System
from text2icd.graph.model import Graph
from text2icd.llm.providers import EndpointNotAllowedError, LLMUnavailableError, check_endpoint

SplitChoice = Literal["dev", "test", "all"]
RESTRICTED_SOURCES = ("mimic",)
BENCHMARKS = Path("benchmarks")


class RunError(ValueError):
    """A run that cannot start: an unknown dataset, a release mismatch, nothing to evaluate."""


def resolve_dataset(name: str, release: str, settings: Settings) -> Path:
    """A dataset by path or by name: `<eval dir>/<name>.jsonl`, `<eval dir>/<name>-<release>.jsonl`,
    or for `<set>-sample`, the committed `benchmarks/<set>/<release>-sample.jsonl`."""
    candidates = [Path(name), settings.eval_dir() / f"{name}.jsonl"]
    candidates.append(settings.eval_dir() / f"{name}-{release}.jsonl")
    if name.endswith("-sample"):
        candidates.append(BENCHMARKS / name.removesuffix("-sample") / f"{release}-sample.jsonl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise RunError(f"no dataset {name!r} (looked for {tried})")


def select(
    examples: Sequence[Example], split: SplitChoice, limit: int | None, seed: int
) -> list[Example]:
    """The examples of a split; with a limit, a seeded random sample of them, in id order."""
    chosen = [e for e in examples if split == "all" or e.split == split]
    if limit is not None and limit < len(chosen):
        chosen = random.Random(f"{seed}:select").sample(chosen, limit)
    return sorted(chosen, key=lambda e: e.id)


def is_restricted(examples: Sequence[Example]) -> bool:
    return any(e.source.startswith(RESTRICTED_SOURCES) for e in examples)


def predict(
    system: System, examples: Sequence[Example], progress: Callable[[int], None] | None = None
) -> list[Prediction]:
    """Run the system over the examples. A failure on one example is kept as an error on it."""
    out: list[Prediction] = []
    phrases = [e.phrase for e in examples]
    stream: Iterator[Prediction] = system.predict_many(phrases)
    for example in examples:
        start = time.perf_counter()
        try:
            prediction = next(stream)
        except StopIteration:
            prediction = Prediction(id=example.id, error="no prediction returned")
        except LLMUnavailableError:  # bad credentials or no access: every call would fail
            raise
        except Exception as exc:
            prediction = Prediction(id=example.id, error=f"{type(exc).__name__}: {exc}")
            stream = system.predict_many(phrases[len(out) + 1 :])
        elapsed = (time.perf_counter() - start) * 1000
        update: dict[str, Any] = {"id": example.id}
        if not prediction.latency_ms:
            update["latency_ms"] = elapsed
        out.append(prediction.model_copy(update=update))
        if progress is not None:
            progress(1)
    return out


def git_state(cwd: Path | None = None) -> dict[str, JsonValue]:
    """The commit the code was run from, and whether tracked files had changes."""

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


@dataclass(frozen=True, slots=True)
class Run:
    header: dict[str, Any]
    predictions: list[Prediction]

    @property
    def path_label(self) -> str:
        dataset = self.header["dataset"]
        return f"{self.header['system']['name']} on {dataset['name']} ({dataset['split']})"


def provenance(
    *,
    system: System,
    graph: Graph,
    dataset_path: Path,
    dataset: Sequence[Example],
    chosen: Sequence[Example],
    split: SplitChoice,
    limit: int | None,
    seed: int,
    started: datetime,
) -> dict[str, JsonValue]:
    manifest = read_manifest(dataset_path)
    return {
        "text2icd": __version__,
        "git": git_state(),
        "release": graph.release,
        "graph_sha256": graph.meta.get("content_sha256"),
        "dataset": {
            "name": _dataset_name(dataset_path, manifest),
            "path": str(dataset_path),
            "sha256": dataset_hash(dataset),
            "split": split,
            "limit": limit,
            "seed": seed,
            "examples": len(chosen),
            "selection_sha256": dataset_hash(chosen),
            "restricted": is_restricted(dataset),
        },
        "system": {"name": system.name, "uses_llm": system.uses_llm, **system.settings()},
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def dataset_name(path: Path) -> str:
    """The name of the dataset at `path`, from its manifest when it has one."""
    return _dataset_name(path, read_manifest(path))


def _dataset_name(path: Path, manifest: dict[str, JsonValue]) -> str:
    name = manifest.get("name")
    if not isinstance(name, str) or not name:
        return path.name.removesuffix(".jsonl")
    return f"{name}-sample" if "sample_of" in manifest else name


def default_output(settings: Settings, header: dict[str, Any]) -> Path:
    stamp = datetime.fromisoformat(header["started"]).strftime("%Y%m%dT%H%M%SZ")
    dataset = header["dataset"]
    name = f"{header['system']['name']}-{dataset['split']}-{stamp}.jsonl"
    return settings.eval_dir() / "results" / str(dataset["name"]) / name


def write_run(path: Path, run: Run, report: Report) -> Path:
    """Write the result file and its metrics; returns the metrics path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"run": run.header}, ensure_ascii=False) + "\n")
        for prediction in run.predictions:
            data = prediction.model_dump(mode="json", exclude_defaults=True)
            fh.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    metrics = metrics_path(path)
    payload = {"run": run.header, "report": report_to_json(report)}
    metrics.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def metrics_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".jsonl") + ".metrics.json")


def read_run(path: Path) -> Run:
    with path.open(encoding="utf-8") as fh:
        first = json.loads(fh.readline())
        if not isinstance(first, dict) or "run" not in first:
            raise RunError(f"{path} is not a result file (no run header)")
        predictions = [Prediction.model_validate_json(line) for line in fh if line.strip()]
    return Run(first["run"], predictions)


def read_metrics(path: Path) -> tuple[dict[str, Any], Report]:
    """The run header and report of a result file (or of its `.metrics.json`)."""
    target = path if path.name.endswith(".metrics.json") else metrics_path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    return data["run"], report_from_json(data["report"])


def chapter_of(graph: Graph) -> Callable[[Example], str]:
    """The chapter of an example's primary gold code (of its first replacement if mapped)."""

    def chapter(example: Example) -> str:
        gold = example.primary
        for code in (gold.code, *gold.mapped_to):
            if code in graph:
                ancestors = graph.ancestors(code)
                return ancestors[-1].key if ancestors else graph.code(code).key
        return "unknown"

    return chapter


def billable_in(graph: Graph) -> Callable[[str], bool]:
    return lambda code: code in graph and graph.code(code).billable


def evaluate(
    system: System,
    graph: Graph,
    dataset_path: Path,
    *,
    split: SplitChoice = "dev",
    limit: int | None = None,
    seed: int = 0,
    resamples: int = 1000,
    progress: Callable[[int], None] | None = None,
    frozen_dir: Path | None = None,
    require_compliant_endpoint: bool = True,
) -> tuple[Run, Report]:
    """Run a system over a dataset split and score it. With `frozen_dir`, a test run on a
    dataset whose frozen test split has changed is refused. A system that sends phrases to a
    hosted service may run on restricted data (MIMIC-IV) only through an approved endpoint."""
    dataset = read_dataset(dataset_path)
    releases = {e.release for e in dataset}
    if releases and releases != {graph.release}:
        found = ", ".join(sorted(releases))
        raise RunError(
            f"the dataset is for {found} and the graph for {graph.release}; "
            f"rewrite it with `text2icd eval reconcile --dataset ... -r {graph.release}`"
        )
    chosen = select(dataset, split, limit, seed)
    if not chosen:
        raise RunError(f"no {split} examples in {dataset_path}")
    if frozen_dir is not None and split != "dev":
        name = _dataset_name(dataset_path, read_manifest(dataset_path))
        try:
            check_frozen(frozen_dir, name, dataset)
        except FrozenSplitError as exc:
            raise RunError(str(exc)) from exc
    if system.endpoint is not None:
        try:
            check_endpoint(
                system.endpoint,
                restricted=is_restricted(dataset),
                require_compliant=require_compliant_endpoint,
            )
        except EndpointNotAllowedError as exc:
            raise RunError(str(exc)) from exc
    started = datetime.now(UTC)
    predictions = predict(system, chosen, progress)
    header = provenance(
        system=system,
        graph=graph,
        dataset_path=dataset_path,
        dataset=dataset,
        chosen=chosen,
        split=split,
        limit=limit,
        seed=seed,
        started=started,
    )
    report = score(
        chosen,
        {p.id: p for p in predictions},
        billable_in(graph),
        group_of=chapter_of(graph),
        resamples=resamples,
        seed=seed,
    )
    return Run(header, predictions), report
