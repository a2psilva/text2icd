"""Command-line interface for text2icd."""

import itertools
import sqlite3
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from text2icd import __version__
from text2icd.config import Settings
from text2icd.embeddings import EmbeddingsUnavailableError, OnnxEmbedder
from text2icd.eval import report as eval_report
from text2icd.eval import synthetic
from text2icd.eval.datasets import (
    DatasetError,
    Example,
    check_dataset,
    read_dataset,
    read_manifest,
    summarize,
    write_dataset,
    write_manifest,
)
from text2icd.eval.main_terms import main_term_recall
from text2icd.eval.mimic import abbreviations as mimic_abbreviations
from text2icd.eval.mimic import align as mimic_align
from text2icd.eval.mimic import annotate, audit
from text2icd.eval.mimic.files import (
    DIAGNOSES,
    DISCHARGE,
    MimicFilesMissingError,
    mimic_file,
    read_notes,
)
from text2icd.eval.mimic.gold import admission_codes
from text2icd.eval.mimic.sections import SectionStats
from text2icd.eval.mimic.silver import SOURCE as SILVER
from text2icd.eval.mimic.silver import build_silver
from text2icd.eval.mimic.splits import FrozenSplitError, check_frozen, freeze
from text2icd.eval.runner import (
    RunError,
    SplitChoice,
    billable_in,
    chapter_of,
    default_output,
    evaluate,
    read_metrics,
    resolve_dataset,
    select,
    write_run,
)
from text2icd.eval.systems import SYSTEMS, SystemSetupError, build_system
from text2icd.eval.versions import Reconciled, Reconciler, load_reconciler, reconcile_example
from text2icd.fetch import FetchError, fetch_release, is_complete
from text2icd.graph.build import build_graph
from text2icd.graph.integrity import check_graph, failed
from text2icd.graph.model import Graph, GraphError, graph_file, load
from text2icd.graph.search import DEFAULT_TOKENIZER, Tokenizer
from text2icd.graph.show import show_code, show_term
from text2icd.llm.cost import PriceError
from text2icd.llm.providers import LLMUnavailableError
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.index import IndexSource
from text2icd.ontology.order import billable_codes, parse_order_file
from text2icd.ontology.stats import release_report
from text2icd.releases import Release, load_registry
from text2icd.text.abbreviations import default_lexicon
from text2icd.text.terms import vocabulary

app = typer.Typer(
    name="text2icd",
    help="Map diagnosis phrases to ICD-10-CM codes.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(
    help="Download and inspect the CMS ICD-10-CM release files.", no_args_is_help=True
)
app.add_typer(data_app, name="data")
graph_app = typer.Typer(help="Inspect a built knowledge graph.", no_args_is_help=True)
app.add_typer(graph_app, name="graph")
eval_app = typer.Typer(help="Evaluation datasets, runs and reports.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
mimic_app = typer.Typer(
    help="Build the local MIMIC-IV evaluation sets. Prints aggregate counts only.",
    no_args_is_help=True,
)
eval_app.add_typer(mimic_app, name="mimic")
audit_app = typer.Typer(
    help="Review a random sample of silver labels by hand and estimate their precision.",
    no_args_is_help=True,
)
mimic_app.add_typer(audit_app, name="audit")
gold_app = typer.Typer(
    help="Code a stratified sample of phrases by hand (docs/annotation-guide.md).",
    no_args_is_help=True,
)
mimic_app.add_typer(gold_app, name="gold")
abbrev_app = typer.Typer(
    help="Abbreviations in MIMIC-IV discharge diagnoses: mining (local only).",
    no_args_is_help=True,
)
mimic_app.add_typer(abbrev_app, name="abbrev")

DataDirOption = Annotated[
    Path | None,
    typer.Option("--data-dir", help="Data directory (default: TEXT2ICD_DATA_DIR or ./data)."),
]
ReleaseOption = Annotated[
    str | None,
    typer.Option("--release", "-r", help="Release id. Default: TEXT2ICD_RELEASE or newest."),
]
# Problems printed per file; the rest are counted.
PROBLEM_LIMIT = 5

GraphFileOption = Annotated[
    Path | None,
    typer.Option("--graph", help="Graph file (default: the release's file in the data directory)."),
]


@app.callback()
def main() -> None:
    """Map diagnosis phrases to ICD-10-CM codes."""


@app.command()
def version() -> None:
    """Print the installed text2icd version."""
    typer.echo(__version__)


@data_app.command("releases")
def data_releases(data_dir: DataDirOption = None) -> None:
    """List the known releases and whether each is fully present locally."""
    settings = _settings(data_dir)
    registry = load_registry()
    for release in registry.releases.values():
        local = "present" if is_complete(release, settings.cms_raw_dir(release.id)) else "missing"
        default = "  (default)" if release.id == registry.default else ""
        typer.echo(
            f"{release.id:<11} {release.scope:<8} effective {release.effective}  {local}{default}"
        )


@data_app.command("fetch")
def data_fetch(
    release: Annotated[
        list[str] | None,
        typer.Option(
            "--release", "-r", help="Release id to fetch; repeatable. Default: TEXT2ICD_RELEASE."
        ),
    ] = None,
    history: Annotated[
        bool, typer.Option("--history", help="Also fetch the FY2016 to FY2025 code lists.")
    ] = False,
    data_dir: DataDirOption = None,
) -> None:
    """Download, verify and unpack CMS release files into the data directory."""
    settings = _settings(data_dir)
    registry = load_registry()
    try:
        targets = (
            [registry.get(r) for r in release] if release else [registry.get(settings.release)]
        )
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--release") from exc
    if history:
        targets += [r for r in registry.by_scope("history") if r not in targets]

    for target in targets:
        _fetch_one(target, settings.cms_raw_dir(target.id))


@data_app.command("stats")
def data_stats(
    release: Annotated[
        list[str] | None,
        typer.Option("--release", "-r", help="Release id; repeatable. Default: TEXT2ICD_RELEASE."),
    ] = None,
    data_dir: DataDirOption = None,
) -> None:
    """Parse a fetched release, print what it contains and run the consistency checks."""
    settings = _settings(data_dir)
    registry = load_registry()
    try:
        targets = (
            [registry.get(r) for r in release] if release else [registry.get(settings.release)]
        )
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--release") from exc
    failed = False
    for target in targets:
        if target.scope != "full":
            raise typer.BadParameter(f"{target.id} is a history release", param_hint="--release")
        raw_dir = settings.cms_raw_dir(target.id)
        _require_fetched(target, raw_dir)
        report = release_report(ReleaseFiles(raw_dir), target.id)
        typer.echo("\n".join(report.lines()))
        failed = failed or not report.ok
    if failed:
        raise typer.Exit(code=1)


@app.command("build-graph")
def build_graph_command(
    release: Annotated[
        str | None,
        typer.Option("--release", "-r", help="Release id. Default: TEXT2ICD_RELEASE or newest."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file (default: data/processed/text2icd-graph-<release>.sqlite).",
        ),
    ] = None,
    tokenizer: Annotated[
        Tokenizer,
        typer.Option(help="FTS5 tokenizer: porter (stemmed words) or unicode61 (exact words)."),
    ] = DEFAULT_TOKENIZER,
    data_dir: DataDirOption = None,
) -> None:
    """Build the knowledge-graph SQLite file of a fetched release."""
    settings = _settings(data_dir)
    target = _full_release(release or settings.release)
    raw_dir = settings.cms_raw_dir(target.id)
    _require_fetched(target, raw_dir)
    report = build_graph(
        ReleaseFiles(raw_dir),
        target.id,
        output or settings.graph_path(target.id),
        tokenizer=tokenizer,
    )
    typer.echo("\n".join(report.lines()))


@graph_app.command("show")
def graph_show(
    path: Annotated[str, typer.Argument(help='Index path, such as "Pneumonia > aspiration".')],
    book: Annotated[IndexSource, typer.Option("--book", "-b", help="Index book.")] = "index",
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Print an Index term: its codes, cross-references and subterms."""
    with _open_graph(release, graph_path, data_dir) as graph:
        term = graph.term_at(path, book)
        if term is None:
            typer.echo(f"error: no {book} term at {path!r}", err=True)
            hits = graph.search_index(path, 5, book=book)
            if hits:
                closest = "; ".join(graph.term(hit.term).path for hit in hits)
                typer.echo(f"closest: {closest}", err=True)
            raise typer.Exit(code=1)
        typer.echo("\n".join(show_term(graph, term)))


@graph_app.command("code")
def graph_code(
    code: Annotated[str, typer.Argument(help="Tabular code, such as E11.22 or E1122.")],
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Print a Tabular code: its place, 7th characters, notes, edges and who names it."""
    with _open_graph(release, graph_path, data_dir) as graph:
        if code not in graph:
            typer.echo(f"error: {code} is not a code of {graph.release}", err=True)
            raise typer.Exit(code=1)
        typer.echo("\n".join(show_code(graph, graph.code(code))))


@graph_app.command("check")
def graph_check(
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Run the integrity checks on a built graph; exit 1 if any check fails."""
    path = _graph_path(release, graph_path, data_dir)
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        results = check_graph(conn)
    typer.echo(str(path))
    for result in results:
        typer.echo("\n".join(result.lines()))
    outcomes = Counter(result.outcome for result in results)
    typer.echo(
        f"{len(results)} checks: {outcomes['pass']} passed, {outcomes['warn']} warned, "
        f"{outcomes['fail']} failed"
    )
    if failed(results):
        raise typer.Exit(code=1)


@eval_app.command("validate")
def eval_validate(
    paths: Annotated[list[Path], typer.Argument(help="Dataset files (JSONL).")],
    codes: Annotated[
        bool,
        typer.Option(
            "--codes/--no-codes",
            help="Check gold codes against the billable codes of the dataset's release.",
        ),
    ] = True,
    data_dir: DataDirOption = None,
) -> None:
    """Check that dataset files follow the format; print what each one holds."""
    settings = _settings(data_dir)
    bad = False
    for path in paths:
        typer.echo(str(path))
        try:
            examples = read_dataset(path)
        except (OSError, DatasetError) as exc:
            typer.echo(f"  error: {exc}", err=True)
            bad = True
            continue
        summary = summarize(examples)
        typer.echo("\n".join(summary.lines()))
        billable = None
        if codes and summary.release:
            try:
                order = ReleaseFiles(settings.cms_raw_dir(summary.release)).order
                billable = billable_codes(parse_order_file(order)).__contains__
            except FileNotFoundError:
                typer.echo(f"  gold codes not checked: {summary.release} is not fetched")
        problems = check_dataset(examples, billable)
        for problem in problems[:PROBLEM_LIMIT]:
            typer.echo(f"  problem: {problem}", err=True)
        if len(problems) > PROBLEM_LIMIT:
            typer.echo(f"  ... and {len(problems) - PROBLEM_LIMIT} more problems", err=True)
        if billable is not None and not problems:
            typer.echo(f"  gold codes checked against {summary.release}: ok")
        bad = bad or bool(problems)
    if bad:
        raise typer.Exit(code=1)


@eval_app.command("reconcile")
def eval_reconcile(
    codes: Annotated[
        list[str] | None, typer.Argument(help="Codes to look up, such as N18.3 or N183.")
    ] = None,
    dataset: Annotated[
        Path | None, typer.Option("--dataset", help="Rewrite this dataset's gold code statuses.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Where to write the rewritten dataset.")
    ] = None,
    release: ReleaseOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Show how codes from older releases relate to a release (valid, mapped or unmappable),
    or rewrite a dataset's gold codes against it."""
    if (dataset is None) != (output is None):
        raise typer.BadParameter("--dataset and --output go together", param_hint="--output")
    if not codes and dataset is None:
        raise typer.BadParameter("give codes or --dataset", param_hint="CODES")
    settings = _settings(data_dir)
    target = _full_release(release or settings.release)
    try:
        reconciler = load_reconciler(target.id, settings)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for code in codes or []:
        try:
            typer.echo(_reconciled(reconciler.reconcile(code), reconciler))
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    if dataset is not None and output is not None:
        try:
            examples = [reconcile_example(e, reconciler) for e in read_dataset(dataset)]
        except (OSError, DatasetError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        write_dataset(output, examples)
        typer.echo(f"{output}: {len(examples):,} examples against {target.id}")
        typer.echo("\n".join(summarize(examples).lines()))


@eval_app.command("synth")
def eval_synth(
    release: ReleaseOption = None,
    seed: Annotated[int, typer.Option(help="Seed for 7th characters and perturbations.")] = 0,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Full set (default: data/processed/eval/cms-synthetic-<release>.jsonl).",
        ),
    ] = None,
    sample_output: Annotated[
        Path | None,
        typer.Option("--sample-output", help="Also write a stratified sample here."),
    ] = None,
    sample_size: Annotated[int, typer.Option(help="Examples in the sample.")] = (
        synthetic.SAMPLE_SIZE
    ),
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Generate the cms-synthetic dataset from a release's graph (and optionally a sample)."""
    settings = _settings(data_dir)
    with _open_graph(release, graph_path, data_dir) as graph:
        examples = synthetic.generate(graph, seed)
        graph_sha256 = graph.meta.get("content_sha256", "")
        target = output or settings.eval_dir() / f"{synthetic.SOURCE}-{graph.release}.jsonl"
    made_with: dict[str, JsonValue] = {
        "name": synthetic.SOURCE,
        "seed": seed,
        "graph_sha256": graph_sha256,
        "text2icd": __version__,
    }
    write_dataset(target, examples)
    full = summarize(examples)
    write_manifest(target, examples, **made_with)
    typer.echo(f"{target}")
    typer.echo("\n".join(synthetic.describe(examples)))
    typer.echo("\n".join(full.lines()))
    if sample_output is not None:
        chosen = synthetic.sample(examples, sample_size, seed)
        write_dataset(sample_output, chosen)
        write_manifest(sample_output, chosen, **made_with, sample_of=full.sha256)
        typer.echo(f"{sample_output}")
        typer.echo("\n".join(synthetic.describe(chosen)))
        typer.echo("\n".join(summarize(chosen).lines()))


@eval_app.command("run")
def eval_run(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset file or name (cms-synthetic-sample, cms-synthetic, mimic-silver, ...).",
        ),
    ],
    system: Annotated[
        str, typer.Option("--system", "-s", help=f"System: {', '.join(sorted(SYSTEMS))}.")
    ],
    split: Annotated[
        SplitChoice, typer.Option(help="Split to run on; test is for reported numbers only.")
    ] = "dev",
    limit: Annotated[
        int | None, typer.Option(min=1, help="Run on a seeded random sample of this many.")
    ] = None,
    option: Annotated[
        list[str] | None, typer.Option("--option", "-O", help="System option key=value.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for --limit and the bootstrap.")] = 0,
    resamples: Annotated[int, typer.Option(min=0, help="Bootstrap resamples.")] = 1000,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Result file (default: under data/processed/eval)."),
    ] = None,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Run a system over a dataset split, write the result file and print the metrics."""
    settings = _settings(data_dir)
    options = _options(option or [])
    with _open_graph(release, graph_path, data_dir) as graph:
        try:
            path = resolve_dataset(dataset, graph.release, settings)
            runner_system = build_system(system, graph, settings, options)
        except (
            RunError,
            SystemSetupError,
            LLMUnavailableError,
            PriceError,
            EmbeddingsUnavailableError,
        ) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        try:
            run, report = evaluate(
                runner_system,
                graph,
                path,
                split=split,
                limit=limit,
                seed=seed,
                resamples=resamples,
                progress=_progress() if runner_system.uses_llm else None,
                frozen_dir=settings.eval_dir(),
                require_compliant_endpoint=settings.require_dua_compliant_endpoint,
            )
        except (RunError, DatasetError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        finally:
            runner_system.close()
    target = output or default_output(settings, run.header)
    metrics = write_run(target, run, report)
    typer.echo(f"{target}\n{metrics}")
    typer.echo("\n".join(eval_report.comparison([(run.header, report)])))


@eval_app.command("report")
def eval_report_command(
    results: Annotated[list[Path], typer.Argument(help="Result files (or their .metrics.json).")],
    by_chapter: Annotated[
        bool, typer.Option("--by-chapter", help="Also a per-chapter table for each run.")
    ] = False,
    intervals: Annotated[
        bool, typer.Option("--intervals/--no-intervals", help="Show 95% intervals.")
    ] = True,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the markdown here.")
    ] = None,
) -> None:
    """Render a markdown table comparing runs."""
    try:
        runs = [read_metrics(path) for path in results]
    except (OSError, KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    lines = eval_report.comparison(runs, intervals)
    if by_chapter:
        for header, report in runs:
            lines += ["", *eval_report.by_group(header, report)]
    text = "\n".join(lines) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(str(output))
    else:
        typer.echo(text, nl=False)


@eval_app.command("main-terms")
def eval_main_terms(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset file or name.")],
    split: Annotated[SplitChoice, typer.Option(help="Split to measure on.")] = "dev",
    limit: Annotated[
        int | None, typer.Option(min=1, help="Measure on a seeded random sample of this many.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for --limit.")] = 0,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Main-term recall of the graph's Index search on a dataset split (aggregate counts)."""
    settings = _settings(data_dir)
    with _open_graph(release, graph_path, data_dir) as graph:
        try:
            path = resolve_dataset(dataset, graph.release, settings)
            chosen = select(read_dataset(path), split, limit, seed)
        except (RunError, DatasetError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        tokenizer = graph.meta.get("fts_tokenizer", "unknown")
        results = main_term_recall(graph, chosen)
    typer.echo(f"{path.name} ({split}), tokenizer {tokenizer}, {len(chosen):,} examples")
    typer.echo("\n".join(result.line() for result in results))


def _progress(every: int = 50) -> Callable[[int], None]:
    """Report progress on stderr every `every` phrases, for slow (LLM) runs."""
    done = [0]

    def step(n: int) -> None:
        done[0] += n
        if done[0] % every == 0:
            typer.echo(f"  {done[0]:,} phrases", err=True)

    return step


def _options(pairs: list[str]) -> dict[str, str]:
    options = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"expected key=value, got {pair!r}", param_hint="--option")
        options[key.strip()] = value.strip()
    return options


@mimic_app.command("sections")
def mimic_sections(
    limit: Annotated[int | None, typer.Option(min=1, help="Read only the first N notes.")] = None,
    data_dir: DataDirOption = None,
) -> None:
    """Count the discharge diagnosis sections and items the extractor finds (no note text)."""
    settings = _settings(data_dir)
    try:
        path = mimic_file(settings, DISCHARGE)
    except MimicFilesMissingError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    stats = SectionStats()
    for number, note in enumerate(read_notes(path), start=1):
        stats.add(note.text)
        if limit is not None and number >= limit:
            break
    typer.echo("\n".join(stats.lines()))


@mimic_app.command("silver")
def mimic_silver(
    release: ReleaseOption = None,
    model: Annotated[
        str, typer.Option(help="Aligner embedding model (Hugging Face repo with an ONNX export).")
    ] = mimic_align.DEFAULT_MODEL,
    model_file: Annotated[str, typer.Option(help="ONNX file in the repo.")] = (
        "onnx/model_quantized.onnx"
    ),
    min_score: Annotated[float, typer.Option(help="Lowest similarity kept.")] = (
        mimic_align.MIN_SCORE
    ),
    min_margin: Annotated[float, typer.Option(help="Lowest margin over the next code.")] = (
        mimic_align.MIN_MARGIN
    ),
    limit: Annotated[int | None, typer.Option(min=1, help="Read only the first N notes.")] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Default: data/processed/eval/mimic-silver.jsonl."),
    ] = None,
    data_dir: DataDirOption = None,
) -> None:
    """Build mimic-silver: discharge diagnosis items aligned to their admission's billed codes.
    Local only; prints aggregate counts."""
    settings = _settings(data_dir)
    target = _full_release(release or settings.release)
    try:
        notes_path = mimic_file(settings, DISCHARGE)
        codes_path = mimic_file(settings, DIAGNOSES)
        reconciler = load_reconciler(target.id, settings)
        embedder = OnnxEmbedder(model, model_file)
    except (FileNotFoundError, EmbeddingsUnavailableError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    codes = admission_codes(codes_path)
    notes = read_notes(notes_path)
    if limit is not None:
        notes = itertools.islice(notes, limit)
    examples, stats = build_silver(
        notes, codes, reconciler, embedder, min_score=min_score, min_margin=min_margin
    )
    target_path = output or settings.eval_dir() / f"{SILVER}.jsonl"
    try:
        check_frozen(settings.eval_dir(), SILVER, examples)
    except FrozenSplitError as exc:
        typer.echo(f"error: {exc}; not written", err=True)
        raise typer.Exit(code=1) from exc
    write_dataset(target_path, examples)
    write_manifest(
        target_path,
        examples,
        name=SILVER,
        aligner=embedder.model,
        min_score=min_score,
        min_margin=min_margin,
        notes_limit=limit,
        sources={
            "discharge": {"file": notes_path.name, "bytes": notes_path.stat().st_size},
            "diagnoses": {"file": codes_path.name, "bytes": codes_path.stat().st_size},
        },
        text2icd=__version__,
    )
    typer.echo(str(target_path))
    typer.echo("\n".join(stats.lines()))
    typer.echo("\n".join(summarize(examples).lines()))


@mimic_app.command("freeze")
def mimic_freeze(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset file or name.")],
    data_dir: DataDirOption = None,
) -> None:
    """Freeze a dataset's test split: record its hash, so it cannot change unnoticed."""
    settings = _settings(data_dir)
    try:
        path = resolve_dataset(dataset, load_registry().get(settings.release).id, settings)
        examples = read_dataset(path)
        name = str(read_manifest(path).get("name") or path.name.removesuffix(".jsonl"))
        digest = freeze(settings.eval_dir(), name, examples)
    except (RunError, DatasetError, FrozenSplitError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    tests = sum(e.split == "test" for e in examples)
    typer.echo(f"froze the test split of {name}: {tests:,} examples, sha256 {digest}")


AuditFileOption = Annotated[
    Path | None,
    typer.Option(
        "--file", "-f", help="Review file (default: data/processed/eval/audit/mimic-silver.csv)."
    ),
]


def _audit_file(settings: Settings, path: Path | None) -> Path:
    return path or settings.eval_dir() / "audit" / f"{SILVER}.csv"


@audit_app.command("sample")
def audit_sample(
    size: Annotated[int, typer.Option("--size", "-n", min=1, help="Rows to review.")] = (
        audit.SAMPLE_SIZE
    ),
    seed: Annotated[int, typer.Option(help="Seed of the random sample.")] = 0,
    file: AuditFileOption = None,
    release: ReleaseOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Draw a random sample of mimic-silver dev examples into a review file."""
    settings = _settings(data_dir)
    target = _audit_file(settings, file)
    try:
        dataset = resolve_dataset(SILVER, "", settings)
        examples = read_dataset(dataset)
        reconciler = load_reconciler(_full_release(release or settings.release).id, settings)
        count = audit.write_review(
            target,
            audit.sample_for_review(examples, size, seed),
            lambda code: reconciler.descriptions.get(code, ""),
        )
    except (RunError, DatasetError, FileNotFoundError, FileExistsError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{target}: {count} rows to review (local only; never share this file)")


@audit_app.command("review")
def audit_review(file: AuditFileOption = None, data_dir: DataDirOption = None) -> None:
    """Give a verdict on each row in the terminal; progress is saved after each answer."""
    path = _audit_file(_settings(data_dir), file)
    if not path.is_file():
        typer.echo(f"error: {path} not found; run `text2icd eval mimic audit sample`", err=True)
        raise typer.Exit(code=1)
    rows = audit.read_review(path)
    typer.echo(audit.REVIEW_HELP)
    given = audit.review(
        rows,
        ask=lambda prompt: input(prompt),
        say=typer.echo,
        save=lambda current: audit.save_review(path, current),
    )
    typer.echo(f"{given} verdicts given; " + audit.precision(rows).line())


@audit_app.command("score")
def audit_score(file: AuditFileOption = None, data_dir: DataDirOption = None) -> None:
    """The silver-label precision with a 95% interval, overall and by alignment score."""
    path = _audit_file(_settings(data_dir), file)
    if not path.is_file():
        typer.echo(f"error: {path} not found; run `text2icd eval mimic audit sample`", err=True)
        raise typer.Exit(code=1)
    rows = audit.read_review(path)
    typer.echo(audit.precision(rows).line())
    typer.echo("\n".join(audit.by_score(rows, [0.70, 0.80, 0.90])))


def _annotation_dir(settings: Settings) -> Path:
    return settings.eval_dir() / "annotation"


@gold_app.command("sample")
def gold_sample(
    size: Annotated[int, typer.Option("--size", "-n", min=3, help="Phrases to code.")] = (
        annotate.SAMPLE_SIZE
    ),
    seed: Annotated[int, typer.Option(help="Seed of the sample.")] = 0,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Draw the phrases to code from mimic-silver, stratified by code frequency and chapter."""
    settings = _settings(data_dir)
    folder = _annotation_dir(settings)
    if annotate.read_answers(folder / "answers.jsonl"):
        typer.echo(f"error: {folder / 'answers.jsonl'} already has answers; keep it", err=True)
        raise typer.Exit(code=1)
    with _open_graph(release, graph_path, data_dir) as graph:
        try:
            examples = read_dataset(resolve_dataset(SILVER, graph.release, settings))
            reconciler = load_reconciler(graph.release, settings)
            billed = admission_codes(
                mimic_file(settings, DIAGNOSES), {str(e.meta["hadm_id"]) for e in examples}
            )
        except (RunError, DatasetError, FileNotFoundError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        def choices(example: Example) -> list[tuple[str, str]]:
            found: dict[str, str] = {}
            for code in billed.get(str(example.meta["hadm_id"]), []):
                result = reconciler.reconcile(code.code)
                for current in (code.code,) if result.status == "valid" else result.mapped_to:
                    found.setdefault(current, graph.code(current).description)
            return list(found.items())

        candidates = annotate.sample_candidates(
            examples, choices, chapter_of(graph), size=size, seed=seed
        )
    annotate.write_jsonl(folder / "candidates.jsonl", candidates)
    bands = Counter(c.band for c in candidates)
    typer.echo(
        f"{folder / 'candidates.jsonl'}: {len(candidates)} phrases "
        f"(head {bands['head']}, torso {bands['torso']}, tail {bands['tail']}; "
        f"{len({c.chapter for c in candidates})} chapters). Local only."
    )


@gold_app.command("annotate")
def gold_annotate(
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Code the sampled phrases in the terminal; answers are saved after each phrase."""
    settings = _settings(data_dir)
    folder = _annotation_dir(settings)
    if not (folder / "candidates.jsonl").is_file():
        typer.echo("error: no candidates; run `text2icd eval mimic gold sample`", err=True)
        raise typer.Exit(code=1)
    candidates = annotate.read_candidates(folder / "candidates.jsonl")
    with _open_graph(release, graph_path, data_dir) as graph:

        def lookup(words: str) -> list[str]:
            hits = graph.search_index(words, 8)
            lines = []
            for hit in hits:
                term = graph.term(hit.term)
                codes = ", ".join(p.written for p in graph.pointers(hit.term)) or "-"
                lines.append(f"{term.book}: {term.path}  [{codes}]")
            return lines or ["no Index term found"]

        def show(code: str) -> list[str]:
            if code not in graph:
                return [f"{code} is not a code of {graph.release}"]
            return show_code(graph, graph.code(code))[:30]

        typer.echo(annotate.HELP)
        session = annotate.Session(
            candidates=candidates,
            answers=annotate.read_answers(folder / "answers.jsonl"),
            billable=billable_in(graph),
            save=annotate.answers_to_jsonl(folder / "answers.jsonl"),
            lookup=lookup,
            show=show,
            ask=lambda prompt: input(prompt),
            say=typer.echo,
        )
        given = session.run()
    typer.echo(f"{given} phrases answered this time; {len(session.answers)} of {len(candidates)}")


@gold_app.command("build")
def gold_build(
    release: ReleaseOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Write mimic-gold.jsonl from the answers so far."""
    settings = _settings(data_dir)
    target_release = _full_release(release or settings.release).id
    folder = _annotation_dir(settings)
    if not (folder / "candidates.jsonl").is_file():
        typer.echo("error: no candidates; run `text2icd eval mimic gold sample`", err=True)
        raise typer.Exit(code=1)
    candidates = annotate.read_candidates(folder / "candidates.jsonl")
    answers = annotate.read_answers(folder / "answers.jsonl")
    examples, verdicts = annotate.build_gold(candidates, answers, target_release)
    try:
        check_frozen(settings.eval_dir(), annotate.SOURCE, examples)
    except FrozenSplitError as exc:
        typer.echo(f"error: {exc}; not written", err=True)
        raise typer.Exit(code=1) from exc
    target = settings.eval_dir() / f"{annotate.SOURCE}.jsonl"
    write_dataset(target, examples)
    write_manifest(target, examples, name=annotate.SOURCE, text2icd=__version__)
    counts = ", ".join(f"{k} {verdicts[k]}" for k in ("coded", "skip", "unsure", "unanswered"))
    typer.echo(f"{target}: {len(examples)} examples ({counts})")
    typer.echo(annotate.agreement_with_silver(examples))
    typer.echo("\n".join(summarize(examples).lines()))


@abbrev_app.command("mine")
def abbrev_mine(
    limit: Annotated[int | None, typer.Option(min=1, help="Read only N dev notes.")] = None,
    min_notes: Annotated[
        int, typer.Option(min=1, help="Show only words used in this many notes.")
    ] = mimic_abbreviations.MIN_NOTES,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Count the words of dev discharge diagnoses unknown to the lexicon and the CMS files.

    Writes data/interim/abbreviation-candidates.tsv (git-ignored) and prints aggregate counts,
    showing only words used in --min-notes notes or more (at least 50).
    """
    if min_notes < mimic_abbreviations.MIN_NOTES:
        typer.echo(f"error: --min-notes is at least {mimic_abbreviations.MIN_NOTES}", err=True)
        raise typer.Exit(code=2)
    settings = _settings(data_dir)
    try:
        path = mimic_file(settings, DISCHARGE)
    except MimicFilesMissingError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    with _open_graph(release, graph_path, data_dir) as graph:
        known = vocabulary(graph)
    mined = mimic_abbreviations.mine(read_notes(path), default_lexicon(), known, limit=limit)
    target = settings.data_dir / "interim" / mimic_abbreviations.CANDIDATES
    written = mimic_abbreviations.write_candidates(target, mined)
    typer.echo("\n".join(mined.lines(min_notes)))
    typer.echo(f"{target}: {written:,} words used in 2 or more notes (local only)")


def _reconciled(result: Reconciled, reconciler: Reconciler) -> str:
    line = f"{result.code}  {result.status}"
    if result.billable_in:
        line += f"  (billable {reconciler.spans(result.billable_in)})"
    if result.mapped_to:
        line += f" -> {', '.join(result.mapped_to)} [{result.how}]"
    elif result.status == "unmappable":
        line += f": {result.how}"
    return line


def _graph_path(release: str | None, graph_path: Path | None, data_dir: Path | None) -> Path:
    try:
        if graph_path is not None:
            return graph_file(path=graph_path)
        settings = _settings(data_dir)
        return graph_file(_full_release(release or settings.release).id, settings=settings)
    except GraphError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _open_graph(release: str | None, graph_path: Path | None, data_dir: Path | None) -> Graph:
    path = _graph_path(release, graph_path, data_dir)
    try:
        return load(path=path)
    except GraphError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _full_release(release_id: str | None) -> Release:
    try:
        target = load_registry().get(release_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--release") from exc
    if target.scope != "full":
        raise typer.BadParameter(f"{target.id} is a history release", param_hint="--release")
    return target


def _require_fetched(release: Release, raw_dir: Path) -> None:
    if not is_complete(release, raw_dir):
        typer.echo(
            f"error: {release.id} is not fetched; run `text2icd data fetch -r {release.id}`",
            err=True,
        )
        raise typer.Exit(code=1)


def _fetch_one(release: Release, raw_dir: Path) -> None:
    typer.echo(f"{release.id}: {raw_dir}")
    try:
        results = fetch_release(release, raw_dir)
    except FetchError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for result in results:
        action = "downloaded" if result.downloaded else "verified"
        typer.echo(f"  {result.kind:<17} {action}, {result.extracted} new file(s) unpacked")


def _settings(data_dir: Path | None) -> Settings:
    return Settings(data_dir=data_dir) if data_dir is not None else Settings()
