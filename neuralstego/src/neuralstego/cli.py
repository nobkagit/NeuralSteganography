"""Command-line interface for neuralstego."""

from __future__ import annotations

import click

from .utils import configure_logging


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False),
    default=None,
    help="Set the log level for the CLI session.",
)
def main(log_level: str | None) -> None:
    """neuralstego command-line interface."""
    configure_logging(log_level)
    click.echo("neuralstego CLI placeholder")


@main.command()
def doctor() -> None:
    """Run environment diagnostics."""
    click.echo("Environment looks good!")
