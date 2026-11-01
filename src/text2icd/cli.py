"""Command-line interface for text2icd."""

import typer

from text2icd import __version__

app = typer.Typer(
    name="text2icd",
    help="Map diagnosis phrases to ICD-10-CM codes.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Map diagnosis phrases to ICD-10-CM codes."""


@app.command()
def version() -> None:
    """Print the installed text2icd version."""
    typer.echo(__version__)
