"""Markdown tables comparing evaluation runs (`text2icd eval report`). Only aggregate numbers."""

import math
from collections.abc import Sequence
from typing import Any

from text2icd.eval.metrics import Estimate, Report

# Columns of the comparison table: (heading, metric, kind).
COLUMNS = (
    ("Acc@1", "accuracy@1", "share"),
    ("Acc@5", "accuracy@5", "share"),
    ("Category", "category accuracy", "share"),
    ("Code-set F1", "code-set F1", "share"),
    ("Invalid", "invalid-code rate", "share"),
    ("ECE", "calibration error", "share"),
    ("LLM calls", "LLM calls per phrase", "number"),
    ("$/1k", "cost per 1,000 phrases", "dollars"),
)


def fmt(estimate: Estimate | None, kind: str, interval: bool = True) -> str:
    if estimate is None or not estimate.defined:
        return "n/a"
    if kind == "share":
        value = f"{estimate.value:.1%}"
        bounds = f"{estimate.low:.1%}-{estimate.high:.1%}"
    elif kind == "dollars":
        value = f"${estimate.value:.4f}" if estimate.value < 0.01 else f"${estimate.value:.2f}"
        bounds = f"{estimate.low:.4f}-{estimate.high:.4f}"
    else:
        value = f"{estimate.value:.2f}"
        bounds = f"{estimate.low:.2f}-{estimate.high:.2f}"
    show = interval and not math.isnan(estimate.low) and estimate.low != estimate.high
    return f"{value} ({bounds})" if show else value


def comparison(runs: Sequence[tuple[dict[str, Any], Report]], intervals: bool = True) -> list[str]:
    """One row per run."""
    head = ["System", "Dataset", "n", *(c[0] for c in COLUMNS), "p50 ms", "p95 ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for header, report in runs:
        dataset = header["dataset"]
        cells = [
            system_label(header),
            f"{dataset['name']} ({dataset['split']})",
            f"{report.scored:,}",
            *(fmt(report.metrics.get(metric), kind, intervals) for _, metric, kind in COLUMNS),
            _latency(report, "p50"),
            _latency(report, "p95"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines += _notes(runs)
    return lines


def selective(runs: Sequence[tuple[dict[str, Any], Report]]) -> list[str]:
    """Selective accuracy: accuracy@1 of the most confident share of the phrases, for the runs
    whose system reports a confidence (none: no lines)."""
    shown = [(header, report) for header, report in runs if report.selective]
    if not shown:
        return []
    coverages = sorted({c for _, report in shown for c in report.selective}, reverse=True)
    head = ["System", "Dataset", *(f"{c:.0%} kept" for c in coverages)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for header, report in shown:
        dataset = header["dataset"]
        cells = [system_label(header), f"{dataset['name']} ({dataset['split']})"]
        cells += [fmt(report.selective.get(c), "share", False) for c in coverages]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def escalation(runs: Sequence[tuple[dict[str, Any], Report]]) -> list[str]:
    """Accuracy on the phrases the coder keeps and on those it escalates, for the runs whose
    system says (none: no lines). L2 and L3 change only the escalated ones."""
    shown = [(h, r) for h, r in runs if "kept" in r.subsets or "escalated" in r.subsets]
    if not shown:
        return []
    head = ["System", "Dataset"]
    for title in ("Kept", "Escalated"):
        head += [title, "Acc@1", "Code-set F1"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for header, report in shown:
        dataset = header["dataset"]
        cells = [system_label(header), f"{dataset['name']} ({dataset['split']})"]
        for name in ("kept", "escalated"):
            part = report.subsets.get(name)
            if part is None:
                cells += ["0", "n/a", "n/a"]
                continue
            cells += [f"{part.scored:,}"]
            cells += [fmt(part.metrics.get(m), "share") for m in ("accuracy@1", "code-set F1")]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def by_group(header: dict[str, Any], report: Report, label: str = "Chapter") -> list[str]:
    """One row per group (chapter) of a single run."""
    head = [label, "n", "Acc@1", "Acc@5", "Category", "Code-set F1"]
    lines = [f"{system_label(header)} on {header['dataset']['name']}", ""]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, group in sorted(report.groups.items(), key=lambda item: _group_key(item[0])):
        cells = [name, f"{group.scored:,}"]
        cells += [fmt(group.metrics.get(m), "share") for m in _GROUP_METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


_GROUP_METRICS = ("accuracy@1", "accuracy@5", "category accuracy", "code-set F1")


def _group_key(name: str) -> tuple[int, str]:
    return (int(name), "") if name.isdigit() else (10_000, name)


def system_label(header: dict[str, Any]) -> str:
    """The system as the tables name it, which is also its key in an expectations file."""
    system = header["system"]
    model = system.get("model")
    label = f"{system['name']} ({model})" if model else str(system["name"])
    if system.get("retrieval"):
        label += " + L2"
    llm = system.get("llm")
    if isinstance(llm, dict):
        label += f" + L3 ({llm.get('model')})"
    return label


def _latency(report: Report, key: str) -> str:
    value = report.latency_ms.get(key)
    return "n/a" if value is None else f"{value:.1f}"


def _notes(runs: Sequence[tuple[dict[str, Any], Report]]) -> list[str]:
    notes = []
    for header, report in runs:
        dataset = header["dataset"]
        git = header.get("git") or {}
        commit = (git.get("commit") or "unknown")[:10] + (" (dirty)" if git.get("dirty") else "")
        extra = []
        if report.unmappable:
            extra.append(f"{report.unmappable:,} unmappable left out")
        if report.mapped:
            extra.append(f"{report.mapped:,} scored by mapped match")
        if report.missing or report.errors:
            extra.append(f"{report.missing:,} missing, {report.errors:,} errors")
        tail = f"; {'; '.join(extra)}" if extra else ""
        notes.append(
            f"- {system_label(header)}: {header['release']}, commit {commit}, dataset "
            f"{dataset['sha256'][:12]}, {header['started']}{tail}"
        )
    return notes
