"""Command-line interface for neuralstego."""

from __future__ import annotations

import importlib
import platform
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from .utils import configure_logging

console = Console()


def _rich_echo(message: str, **style_kwargs: Any) -> None:
    """Print a message using Rich's console."""

    console.print(message, **style_kwargs)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version="0.1.0", prog_name="neuralstego")
@click.option(
    "--log-level",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False),
    default=None,
    help="Set the log level for the CLI session.",
)
def main(log_level: str | None) -> None:
    """Entry point for the neuralstego command-line interface."""

    configure_logging(log_level)


@main.command()
@click.option(
    "--cover",
    type=click.Path(path_type=Path, exists=False, dir_okay=False),
    required=True,
    help="Path to the cover image that will carry the hidden message.",
)
@click.option(
    "--message",
    type=str,
    required=True,
    help="Message (or path to message) to embed inside the cover.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    required=True,
    help="Destination path for the generated stego artifact.",
)
def encode(cover: Path, message: str, output: Path) -> None:
    """Encode a message into a cover asset (placeholder)."""

    _rich_echo(
        "[bold green]Encode[/bold green] called with the following parameters:",
    )
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", justify="right")
    table.add_column("Value", style="white")
    table.add_row("Cover", str(cover))
    table.add_row("Message", message)
    table.add_row("Output", str(output))
    console.print(table)


@main.command()
@click.option(
    "--stego",
    type=click.Path(path_type=Path, exists=False, dir_okay=False),
    required=True,
    help="Path to the stego artifact containing the hidden message.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    required=True,
    help="Destination for the extracted message.",
)
def decode(stego: Path, output: Path) -> None:
    """Decode a message from a stego asset (placeholder)."""

    _rich_echo("[bold blue]Decode[/bold blue] called with the following parameters:")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", justify="right")
    table.add_column("Value", style="white")
    table.add_row("Stego", str(stego))
    table.add_row("Output", str(output))
    console.print(table)


@main.command()
def doctor() -> None:
    """Check the execution environment for common dependencies."""

    python_version = platform.python_version()
    checks = {
        "Python": (True, f"Version {python_version}"),
        "torch": _import_with_version_hint("torch", "pip install torch"),
        "transformers": _import_with_version_hint(
            "transformers", "pip install transformers"
        ),
    }

    table = Table(title="neuralstego doctor", header_style="bold magenta")
    table.add_column("Component", style="cyan", justify="right")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    missing_dependencies = []
    for component, (ok, details) in checks.items():
        status = "[green]OK[/]" if ok else "[red]Missing[/]"
        table.add_row(component, status, details)
        if not ok:
            missing_dependencies.append((component, details))

    console.print(table)

    if missing_dependencies:
        _rich_echo("\n[bold yellow]Next steps:[/bold yellow]")
        for component, details in missing_dependencies:
            _rich_echo(f" • Install [bold]{component}[/bold]: {details}")
    else:
        _rich_echo("\n[bold green]All required dependencies are available![/bold green]")


def _import_with_version_hint(module_name: str, install_hint: str) -> tuple[bool, str]:
    """Try to import a module and return success status with details."""

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False, install_hint

    version = getattr(module, "__version__", "unknown version")
    return True, f"Version {version}"
