"""Command-line interface for text2icd."""

import itertools
import sqlite3
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Annotated, cast

import typer
from pydantic import JsonValue

from text2icd import __version__
from text2icd.config import Settings
from text2icd.embeddings import EmbeddingsUnavailableError, OnnxEmbedder, load_embedder
from text2icd.eval import report as eval_report
from text2icd.eval import retrieval, synthetic
from text2icd.eval.bench import bench
from text2icd.eval.calibrate import (
    FOLDS,
    TARGET,
    CalibrationOutputError,
    CalibrationUnavailableError,
    choice_lines,
    collect,
    collect_choices,
    escalated,
    fit,
    fit_choice,
    refuse_output,
)
from text2icd.eval.calibrate import lines as calibration_lines
from text2icd.eval.check import check, check_lines, load_expectations
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
from text2icd.eval.index import index_stats
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
from text2icd.eval.parsing import parse_stats
from text2icd.eval.runner import (
    RunError,
    SplitChoice,
    billable_in,
    chapter_of,
    dataset_name,
    default_output,
    evaluate,
    git_state,
    is_restricted,
    read_metrics,
    resolve_dataset,
    select,
    write_run,
)
from text2icd.eval.systems import (
    DEFAULT_EMBEDDING_MODEL,
    SYSTEMS,
    CoderSystem,
    SystemSetupError,
    build_system,
)
from text2icd.eval.versions import Reconciled, Reconciler, load_reconciler, reconcile_example
from text2icd.fetch import FetchError, fetch_release, is_complete
from text2icd.graph.build import build_graph
from text2icd.graph.integrity import check_graph, failed
from text2icd.graph.model import Graph, GraphError, graph_file, load, read_meta
from text2icd.graph.search import DEFAULT_TOKENIZER, Tokenizer
from text2icd.graph.show import show_code, show_term
from text2icd.llm.cost import PriceError
from text2icd.llm.providers import LLMUnavailableError
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.index import IndexSource
from text2icd.ontology.order import billable_codes, parse_order_file
from text2icd.ontology.stats import release_report
from text2icd.releases import Release, load_registry
from text2icd.resolve.coder import Coder, describe_code_set
from text2icd.resolve.confidence import (
    Calibration,
    CalibrationError,
    read_calibration,
    write_calibration,
)
from text2icd.resolve.resolver import Resolver, ResolverError
from text2icd.resolve.retrieve import DEFAULT_MODEL as DEFAULT_RETRIEVAL_MODEL
from text2icd.resolve.retrieve import (
    DEFAULT_SOURCES,
    EmbeddingRetriever,
    Source,
    VectorsError,
    build_vectors,
    load_vectors,
    vectors_path,
)
from text2icd.resolve.retrieve import SOURCES as RETRIEVAL_SOURCES
from text2icd.resolve.walk import IndexWalker, describe_walk
from text2icd.text.abbreviations import default_lexicon
from text2icd.text.model import Setting
from text2icd.text.parse import Parser, describe_parsed
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
    help="Abbreviations in MIMIC-IV discharge diagnoses: mining and a hand review of expansions.",
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


@app.command("parse")
def parse_command(
    phrase: Annotated[str, typer.Argument(help="A diagnosis phrase.")],
    setting: Annotated[
        str,
        typer.Option(help="inpatient (uncertain diagnoses coded as if established) or outpatient."),
    ] = "inpatient",
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the parsed phrase as JSON.")
    ] = False,
    no_graph: Annotated[
        bool, typer.Option("--no-graph", help="Parse without the graph's Index terms.")
    ] = False,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Show how text understanding reads a phrase: conditions, relations and trace."""
    if setting not in ("inpatient", "outpatient"):
        typer.echo("error: --setting is inpatient or outpatient", err=True)
        raise typer.Exit(code=2)
    if no_graph:
        parser = Parser(setting="outpatient" if setting == "outpatient" else "inpatient")
    else:
        with _open_graph(release, graph_path, data_dir) as graph:
            parser = Parser.from_graph(
                graph, "outpatient" if setting == "outpatient" else "inpatient"
            )
    parsed = parser.parse(phrase)
    if as_json:
        typer.echo(parsed.model_dump_json(indent=2))
        return
    typer.echo("\n".join(describe_parsed(parsed)))


@app.command("code")
def code_command(
    phrase: Annotated[str, typer.Argument(help="A diagnosis phrase.")],
    setting: Annotated[
        str,
        typer.Option(help="inpatient (uncertain diagnoses coded as if established) or outpatient."),
    ] = "inpatient",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the result as JSON (docs/schema/coding-result.json)."),
    ] = False,
    explain: Annotated[
        bool,
        typer.Option(
            "--explain", help="With --json: add each code's Index path and the L2/L3 decisions."
        ),
    ] = False,
    use_retrieval: Annotated[
        bool | None,
        typer.Option(
            "--retrieval/--no-retrieval",
            help="L2 for an escalated phrase: needs the graph's vectors file and a calibration "
            "table with a retrieval stage (default: when both are there).",
        ),
    ] = None,
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm",
            help="L3 for a phrase still escalated: sends it to TEXT2ICD_LLM_PROVIDER, at a cost "
            "(TEXT2ICD_LLM_BUDGET_USD caps it).",
        ),
    ] = False,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Code a phrase: the Index walk and Tabular verification (L1), then local retrieval (L2)
    when the coder is unsure, and the LLM navigator (L3) only with --llm."""
    if setting not in ("inpatient", "outpatient"):
        typer.echo("error: --setting is inpatient or outpatient", err=True)
        raise typer.Exit(code=2)
    chosen: Setting = "outpatient" if setting == "outpatient" else "inpatient"
    settings = _settings(data_dir)
    with _open_graph(release, graph_path, data_dir) as graph:
        try:
            resolver = Resolver(
                settings=settings, graph=graph, retrieval=use_retrieval, llm=use_llm
            )
        except (ResolverError, CalibrationError, LLMUnavailableError, PriceError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        with resolver:
            for warning in resolver.warnings:
                typer.echo(f"warning: {warning}", err=True)
            try:
                if as_json:
                    result = resolver.code(phrase, setting=chosen, explain=explain)
                    typer.echo(result.model_dump_json(indent=2))
                    return
                code_set = resolver.code_set(phrase, setting=chosen)
            except (ResolverError, EmbeddingsUnavailableError, OSError) as exc:
                typer.echo(f"error: {exc}", err=True)
                raise typer.Exit(code=1) from exc
    typer.echo("\n".join(describe_code_set(code_set)))


@app.command("walk")
def walk_command(
    phrase: Annotated[str, typer.Argument(help="A diagnosis phrase.")],
    setting: Annotated[
        str,
        typer.Option(help="inpatient (uncertain diagnoses coded as if established) or outpatient."),
    ] = "inpatient",
    top: Annotated[int, typer.Option(min=1, help="Candidates shown per diagnosis.")] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Print the walk as JSON.")] = False,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Show how the Alphabetic Index walk codes a phrase: candidates, paths and trace."""
    if setting not in ("inpatient", "outpatient"):
        typer.echo("error: --setting is inpatient or outpatient", err=True)
        raise typer.Exit(code=2)
    with _open_graph(release, graph_path, data_dir) as graph:
        walker = IndexWalker.from_graph(
            graph, "outpatient" if setting == "outpatient" else "inpatient"
        )
        result = walker.walk(phrase)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo("\n".join(describe_walk(result, top)))


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
    embeddings: Annotated[
        bool,
        typer.Option(
            "--embeddings/--no-embeddings",
            help="Also embed the texts for local retrieval (L2), as `graph embed` does.",
        ),
    ] = False,
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
    if embeddings:
        with load(path=report.path) as graph:
            _embed(graph, DEFAULT_RETRIEVAL_MODEL, DEFAULT_SOURCES)


@graph_app.command("embed")
def graph_embed(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="model2vec:<id> or onnx:<id> (Hugging Face Hub)."),
    ] = DEFAULT_RETRIEVAL_MODEL,
    texts_option: Annotated[
        str, typer.Option("--texts", help="Texts to embed: tabular, index or tabular,index.")
    ] = ",".join(DEFAULT_SOURCES),
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Embed the graph's texts for local retrieval (L2): a float16 matrix next to the graph
    file, with a JSON file naming the model and the graph it belongs to."""
    sources = _sources(texts_option)
    with _open_graph(release, graph_path, data_dir) as graph:
        _embed(graph, model, sources)


def _embed(graph: Graph, model: str, sources: tuple[Source, ...]) -> None:
    try:
        embedder = load_embedder(model)
    except (ValueError, EmbeddingsUnavailableError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"embedding {graph.release} texts with {model}", err=True)
    built = build_vectors(
        graph, embedder, model, sources=sources, progress=_progress(32_768, "texts")
    )
    typer.echo("\n".join(built.lines()))


def _retriever(graph: Graph) -> EmbeddingRetriever:
    """L2's retriever over the graph's vectors file."""
    try:
        return load_vectors(graph)
    except (VectorsError, ValueError, EmbeddingsUnavailableError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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
            notes = runner_system.warnings()
        except (RunError, DatasetError, LLMUnavailableError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        finally:
            runner_system.close()
    for note in notes:
        typer.echo(f"warning: {note}", err=True)
    target = output or default_output(settings, run.header)
    metrics = write_run(target, run, report)
    typer.echo(f"{target}\n{metrics}")
    typer.echo("\n".join(eval_report.comparison([(run.header, report)])))
    if table := eval_report.selective([(run.header, report)]):
        typer.echo("\n".join(["Selective accuracy@1", "", *table]))
    if table := eval_report.escalation([(run.header, report)]):
        typer.echo("\n".join(["", "Kept and escalated by the coder", "", *table]))


@eval_app.command("bench")
def eval_bench(
    dataset: Annotated[
        str, typer.Option("--dataset", "-d", help="Dataset file or name.")
    ] = "cms-synthetic-sample",
    system: Annotated[
        str, typer.Option("--system", "-s", help=f"System: {', '.join(sorted(SYSTEMS))}.")
    ] = "coder",
    split: Annotated[SplitChoice, typer.Option(help="Split to time.")] = "all",
    limit: Annotated[
        int | None, typer.Option(min=1, help="Time a seeded random sample of this many.")
    ] = None,
    option: Annotated[
        list[str] | None, typer.Option("--option", "-O", help="System option key=value.")
    ] = None,
    rounds: Annotated[int, typer.Option(min=1, help="Times to code every phrase.")] = 1,
    seed: Annotated[int, typer.Option(help="Seed for --limit.")] = 0,
    imports: Annotated[
        bool, typer.Option("--import/--no-import", help="Also time the import in a new process.")
    ] = True,
    as_json: Annotated[
        Path | None, typer.Option("--json", help="Also write the numbers to this JSON file.")
    ] = None,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Time a system on one thread: start-up, peak memory, latency per phrase (p50 to max) and
    phrases per second (task 9.8). Prints example ids and numbers only."""
    settings = _settings(data_dir)
    options = _options(option or [])
    path = _graph_path(release, graph_path, data_dir)
    try:
        release_id = read_meta(path)["release"]
        dataset_path = resolve_dataset(dataset, release_id, settings)
        examples = select(read_dataset(dataset_path), split, limit, seed)
        result = bench(
            path, system, settings, options, examples, f"{dataset} ({split})", rounds, imports
        )
    except (
        RunError,
        DatasetError,
        SystemSetupError,
        EmbeddingsUnavailableError,
        GraphError,
        ValueError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("\n".join(result.lines()))
    if as_json is not None:
        as_json.parent.mkdir(parents=True, exist_ok=True)
        as_json.write_text(result.model_dump_json(indent=2) + "\n", "utf-8")


@eval_app.command("calibrate")
def eval_calibrate(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset file or name.")],
    split: Annotated[SplitChoice, typer.Option(help="Split to fit on.")] = "dev",
    limit: Annotated[
        int | None, typer.Option(min=1, help="Fit on a seeded random sample of this many.")
    ] = None,
    target: Annotated[
        float,
        typer.Option(min=0.5, max=1.0, help="Precision of the code sets kept (the threshold)."),
    ] = TARGET,
    folds: Annotated[
        int, typer.Option(min=2, help="Held-out folds, grouped by gold code.")
    ] = FOLDS,
    option: Annotated[
        list[str] | None, typer.Option("--option", "-O", help="Coder option key=value.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for --limit.")] = 0,
    use_retrieval: Annotated[
        bool | None,
        typer.Option(
            "--retrieval/--no-retrieval",
            help="Also fit L2's choice on the escalated examples, with the graph's vectors file "
            "(default: when it exists).",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Table to write (default: under data/processed/calibration). The shipped one "
            "is src/text2icd/resources/calibration.json, fitted on a public dataset only.",
        ),
    ] = None,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Fit the coder's confidence calibration on a split and choose the escalation threshold,
    then L2's choice for the escalated examples; prints aggregate results only."""
    settings = _settings(data_dir)
    with _open_graph(release, graph_path, data_dir) as graph:
        try:
            path = resolve_dataset(dataset, graph.release, settings)
            chosen = select(read_dataset(path), split, limit, seed)
            system = build_system("coder", graph, settings, _options(option or []))
        except (RunError, DatasetError, SystemSetupError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        name = f"{dataset_name(path)} {split} {graph.release}"
        target_path = output or settings.data_dir / "processed" / "calibration" / (
            name.replace(" ", "-") + ".json"
        )
        try:
            refuse_output(target_path, is_restricted(chosen))
        except CalibrationOutputError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        assert isinstance(system, CoderSystem)
        wanted = use_retrieval or (use_retrieval is None and vectors_path(graph.path).is_file())
        retriever = _retriever(graph) if wanted else None
        typer.echo(f"{path.name} ({split}), {len(chosen):,} examples", err=True)
        sample = collect(system.coder, chosen, _progress(5000))
        provenance: dict[str, JsonValue] = {
            "name": name,
            "dataset": path.name,
            "split": split,
            "release": graph.release,
            "options": {
                k: v for k, v in system.settings().items() if k not in ("calibration", "retrieval")
            },
            "commit": git_state().get("commit"),
        }
        try:
            fitted = fit(sample, target=target, folds=folds, fitted=provenance)
            calibration = fitted.calibration
            if retriever is not None:
                coder = Coder(
                    graph, system.coder.walker, system.coder.verifier, calibration, retriever
                )
                escalating = [chosen[i] for i in escalated(sample, calibration)]
                typer.echo(f"retrieval for {len(escalating):,} escalated examples", err=True)
                choices = collect_choices(coder, escalating, _progress(1000))
                staged = fit_choice(
                    choices,
                    fitted,
                    model=retriever.name,
                    folds=folds,
                    fitted={"dataset": path.name, "split": split, "release": graph.release},
                )
                calibration = calibration.model_copy(update={"retrieval": staged.stage})
        except CalibrationUnavailableError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    write_calibration(target_path, calibration)
    typer.echo(f"{target_path}\n{sample.examples:,} scored, {sample.uncoded:,} without a code")
    typer.echo("\n".join(calibration_lines(fitted)))
    if retriever is not None:
        typer.echo("\n".join(choice_lines(staged)))


def _calibration(settings: Settings) -> Calibration | None:
    try:
        return read_calibration(settings.calibration)
    except (OSError, CalibrationError) as exc:
        typer.echo(f"error: calibration table: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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
    if table := eval_report.selective(runs):
        lines += ["Selective accuracy@1", "", *table, ""]
    if table := eval_report.escalation(runs):
        lines += ["Kept and escalated by the coder", "", *table, ""]
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


@eval_app.command("check")
def eval_check(
    results: Annotated[list[Path], typer.Argument(help="Result files (or their .metrics.json).")],
    expected: Annotated[
        Path, typer.Option("--expected", "-e", help="Expectations file (TOML) with the bounds.")
    ],
) -> None:
    """Check runs against the bounds of an expectations file; exit 1 when any fails."""
    try:
        expectations = load_expectations(expected)
        runs = [read_metrics(path) for path in results]
    except (OSError, KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    outcomes = check(runs, expectations)
    typer.echo("\n".join(check_lines(outcomes)))
    ran = {eval_report.system_label(header) for header, _ in runs}
    if skipped := [label for label in expectations.systems if label not in ran]:
        typer.echo(f"not run, not checked: {', '.join(skipped)}")
    failed = sum(not o.ok for o in outcomes)
    if failed:
        typer.echo(f"{failed} of {len(outcomes)} checks failed", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"all {len(outcomes)} checks passed")


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
    """Main-term recall of Index search and of the Index walk's main-term selection on a dataset
    split (aggregate counts)."""
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


@eval_app.command("retrieval")
def eval_retrieval(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset file or name.")],
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Retriever: bm25, vectors (the graph's vectors file, from `graph embed`), "
            "model2vec:<id> or onnx:<id> (Hugging Face Hub). Repeatable.",
        ),
    ] = None,
    texts_option: Annotated[
        list[str] | None,
        typer.Option("--texts", help="Texts to rank: tabular, index or tabular,index. Repeatable."),
    ] = None,
    split: Annotated[SplitChoice, typer.Option(help="Split to measure on.")] = "dev",
    limit: Annotated[
        int | None, typer.Option(min=1, help="Measure on a seeded random sample of this many.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for --limit.")] = 0,
    option: Annotated[
        list[str] | None, typer.Option("--option", "-O", help="Coder option key=value.")
    ] = None,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Recall of local retrievers for L2 (task 7.4): the codes each ranks for a phrase, completed
    to billable codes, over all examples, those the coder escalates and those it gets wrong;
    with latency and size (aggregates only)."""
    specs = model or ["bm25", f"model2vec:{DEFAULT_EMBEDDING_MODEL}"]
    settings_list = [_sources(value) for value in texts_option or ["tabular"]]
    settings = _settings(data_dir)
    with _open_graph(release, graph_path, data_dir) as graph:
        try:
            path = resolve_dataset(dataset, graph.release, settings)
            chosen = select(read_dataset(path), split, limit, seed)
            system = build_system("coder", graph, settings, _options(option or []))
        except (RunError, DatasetError, SystemSetupError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        assert isinstance(system, CoderSystem)
        typer.echo(f"{path.name} ({split}), {len(chosen):,} examples", err=True)
        found = retrieval.queries(system.coder, chosen, _progress(10_000))
        results = []
        for spec in specs:
            for sources in settings_list:
                typer.echo(f"  {spec} over {'+'.join(sources)}", err=True)
                try:
                    results.append(
                        retrieval.measure(spec, graph, sources, found, system.coder.verifier)
                    )
                except (ValueError, FileNotFoundError, EmbeddingsUnavailableError) as exc:
                    typer.echo(f"error: {exc}", err=True)
                    raise typer.Exit(code=1) from exc
    typer.echo(f"{path.name} ({split}), {graph.release}, {len(found):,} scored examples")
    typer.echo("\n".join(retrieval.summary(found)) + "\n")
    typer.echo("\n".join(retrieval.table(results)))


@eval_app.command("index")
def eval_index(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset file or name.")],
    split: Annotated[SplitChoice, typer.Option(help="Split to measure on.")] = "dev",
    limit: Annotated[
        int | None, typer.Option(min=1, help="Measure on a seeded random sample of this many.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for --limit.")] = 0,
    show: Annotated[
        int,
        typer.Option(min=0, help="Show this many missed phrases (public datasets only)."),
    ] = 0,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """How often the Index walk's first candidate is a prefix of the gold code, before Tabular
    verification (aggregate counts, by generator)."""
    settings = _settings(data_dir)
    with _open_graph(release, graph_path, data_dir) as graph:
        try:
            path = resolve_dataset(dataset, graph.release, settings)
            chosen = select(read_dataset(path), split, limit, seed)
        except (RunError, DatasetError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if show and is_restricted(chosen):
            typer.echo("error: --show is refused for restricted (MIMIC-IV) datasets", err=True)
            raise typer.Exit(code=2)
        stats = index_stats(chosen, IndexWalker.from_graph(graph), keep=show)
    typer.echo(f"{path.name} ({split}), {len(chosen):,} examples")
    typer.echo("\n".join(stats.lines()))
    for miss in stats.misses:
        typer.echo(f"  {miss.id}: {miss.phrase} | gold {miss.gold} | {miss.found}")


@eval_app.command("parse")
def eval_parse(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset file or name.")],
    split: Annotated[SplitChoice, typer.Option(help="Split to measure on.")] = "dev",
    limit: Annotated[
        int | None, typer.Option(min=1, help="Measure on a seeded random sample of this many.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for --limit.")] = 0,
    setting: Annotated[str, typer.Option(help="inpatient or outpatient.")] = "inpatient",
    show: Annotated[
        int,
        typer.Option(min=0, help="Show this many mismatched phrases (public datasets only)."),
    ] = 0,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """How text understanding reads a dataset split: aggregate counts, and agreement with the
    expected number of diagnoses (mimic-gold annotations; one per cms-synthetic phrase)."""
    if setting not in ("inpatient", "outpatient"):
        typer.echo("error: --setting is inpatient or outpatient", err=True)
        raise typer.Exit(code=2)
    settings = _settings(data_dir)
    with _open_graph(release, graph_path, data_dir) as graph:
        try:
            path = resolve_dataset(dataset, graph.release, settings)
            chosen = select(read_dataset(path), split, limit, seed)
        except (RunError, DatasetError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if show and is_restricted(chosen):
            typer.echo("error: --show is refused for restricted (MIMIC-IV) datasets", err=True)
            raise typer.Exit(code=2)
        parser = Parser.from_graph(graph, "outpatient" if setting == "outpatient" else "inpatient")
    stats = parse_stats(chosen, parser, keep=show)
    typer.echo(f"{path.name} ({split}), {setting}, {len(chosen):,} examples")
    typer.echo("\n".join(stats.lines()))
    for mismatch in stats.mismatches:
        found = " | ".join(mismatch.conditions)
        typer.echo(
            f"  {mismatch.id}: expected {mismatch.expected}, found {mismatch.found}: {found}"
        )


def _progress(every: int = 50, unit: str = "phrases") -> Callable[[int], None]:
    """Report progress on stderr every `every` items, for slow runs."""
    done = [0]

    def step(n: int) -> None:
        done[0] += n
        if done[0] % every == 0:
            typer.echo(f"  {done[0]:,} {unit}", err=True)

    return step


def _sources(value: str) -> tuple[Source, ...]:
    """`tabular`, `index` or `tabular,index`, as --texts gives them."""
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts or any(part not in RETRIEVAL_SOURCES for part in parts):
        typer.echo(f"error: --texts {value!r}: use tabular, index or both", err=True)
        raise typer.Exit(code=2)
    return cast(tuple[Source, ...], parts)


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


AbbrevFileOption = Annotated[
    Path | None,
    typer.Option("--file", help="Review file (default: <eval dir>/audit/abbreviations.csv)."),
]


def _abbrev_file(settings: Settings, path: Path | None) -> Path:
    return path or settings.eval_dir() / "audit" / mimic_abbreviations.REVIEW


@abbrev_app.command("sample")
def abbrev_sample(
    size: Annotated[int, typer.Option("--size", "-n", min=1, help="Rows to review.")] = (
        mimic_abbreviations.REVIEW_SIZE
    ),
    seed: Annotated[int, typer.Option(help="Seed of the random sample.")] = 0,
    file: AbbrevFileOption = None,
    release: ReleaseOption = None,
    graph_path: GraphFileOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Draw expansions from dev discharge diagnoses into a review file (one per note)."""
    settings = _settings(data_dir)
    target = _abbrev_file(settings, file)
    try:
        path = mimic_file(settings, DISCHARGE)
    except MimicFilesMissingError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    with _open_graph(release, graph_path, data_dir) as graph:
        known = vocabulary(graph)
    rows = mimic_abbreviations.sample_expansions(
        read_notes(path), default_lexicon(), known, size, seed
    )
    try:
        count = mimic_abbreviations.write_expansion_review(target, rows)
    except FileExistsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{target}: {count} rows to review (local only; never share this file)")


@abbrev_app.command("review")
def abbrev_review(file: AbbrevFileOption = None, data_dir: DataDirOption = None) -> None:
    """Judge each expansion in the terminal; progress is saved after each answer."""
    path = _abbrev_file(_settings(data_dir), file)
    if not path.is_file():
        typer.echo(f"error: {path} not found; run `text2icd eval mimic abbrev sample`", err=True)
        raise typer.Exit(code=1)
    rows = audit.read_review(path)
    typer.echo(mimic_abbreviations.REVIEW_HELP)
    given = audit.review(
        rows,
        ask=lambda prompt: input(prompt),
        say=typer.echo,
        save=lambda current: audit.save_review(path, current, mimic_abbreviations.REVIEW_COLUMNS),
        describe=mimic_abbreviations.describe_expansion,
        help_text=mimic_abbreviations.REVIEW_HELP,
    )
    typer.echo(f"{given} verdicts given; " + audit.precision(rows).line("accuracy"))


@abbrev_app.command("score")
def abbrev_score(file: AbbrevFileOption = None, data_dir: DataDirOption = None) -> None:
    """Expansion accuracy with a 95% interval, overall and by how the meaning was chosen."""
    path = _abbrev_file(_settings(data_dir), file)
    if not path.is_file():
        typer.echo(f"error: {path} not found; run `text2icd eval mimic abbrev sample`", err=True)
        raise typer.Exit(code=1)
    typer.echo("\n".join(mimic_abbreviations.accuracy(audit.read_review(path))))


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
