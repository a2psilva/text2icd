"""Command-line interface for text2icd."""

from pathlib import Path
from typing import Annotated

import typer

from text2icd import __version__
from text2icd.config import Settings
from text2icd.fetch import FetchError, fetch_release, is_complete
from text2icd.graph.build import build_graph
from text2icd.ontology.files import ReleaseFiles
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

DataDirOption = Annotated[
    Path | None,
    typer.Option("--data-dir", help="Data directory (default: TEXT2ICD_DATA_DIR or ./data)."),
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
    data_dir: DataDirOption = None,
) -> None:
    """Build the knowledge-graph SQLite file of a fetched release."""
    settings = _settings(data_dir)
    target = _full_release(release or settings.release)
    raw_dir = settings.cms_raw_dir(target.id)
    _require_fetched(target, raw_dir)
    report = build_graph(ReleaseFiles(raw_dir), target.id, output or settings.graph_path(target.id))
    typer.echo("\n".join(report.lines()))


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
