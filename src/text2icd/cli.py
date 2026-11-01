"""Command-line interface for text2icd."""

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from text2icd import __version__
from text2icd.config import Settings
from text2icd.eval.datasets import (
    DatasetError,
    check_dataset,
    read_dataset,
    summarize,
    write_dataset,
)
from text2icd.eval.versions import Reconciled, Reconciler, load_reconciler, reconcile_example
from text2icd.fetch import FetchError, fetch_release, is_complete
from text2icd.graph.build import build_graph
from text2icd.graph.integrity import check_graph, failed
from text2icd.graph.model import Graph, GraphError, graph_file, load
from text2icd.graph.search import DEFAULT_TOKENIZER, Tokenizer
from text2icd.graph.show import show_code, show_term
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.index import IndexSource
from text2icd.ontology.order import billable_codes, parse_order_file
from text2icd.ontology.stats import release_report
from text2icd.releases import Release, load_registry

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
