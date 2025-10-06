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
    "--context",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to the context text file that seeds language generation.",
)
@click.option(
    "--message",
    type=str,
    required=True,
    help="Plaintext message to conceal using GPT-based steganography.",
)
@click.option(
    "--mode",
    type=click.Choice(["arithmetic", "huffman", "bins", "sample"], case_sensitive=False),
    default="arithmetic",
    show_default=True,
    help="Embedding backend to simulate (aligned with the reference scripts).",
)
@click.option(
    "--model",
    type=str,
    default="gpt2-fa",
    show_default=True,
    help="Language model identifier to load (e.g. gpt2-fa).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional path to write the generated stego text; prints to stdout otherwise.",
)
def encode(
    context: Path,
    message: str,
    mode: str,
    model: str,
    output: Path | None,
) -> None:
    """Encode a plaintext message into fluent text (placeholder)."""

    try:
        context_preview = context.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - simple I/O guard
        raise click.ClickException(f"Failed to read context file: {exc}") from exc

    if not context_preview.strip():
        raise click.ClickException("Context file is empty; provide seed text to guide generation.")

    summary = Table(title="encode parameters", header_style="bold magenta")
    summary.add_column("Parameter", style="cyan", justify="right")
    summary.add_column("Value", style="white")
    summary.add_row("Context", str(context))
    summary.add_row("Message", message)
    summary.add_row("Mode", mode.lower())
    summary.add_row("Model", model)
    summary.add_row("Output", str(output) if output else "stdout")

    _rich_echo("[bold green]Encoding message into carrier text[/bold green]")
    console.print(summary)

    preview_excerpt = context_preview.strip().splitlines()[0][:120]
    _rich_echo(f"\nContext preview: [italic]{preview_excerpt}...[/italic]")

    _rich_echo(
        "\n[italic]Actual language-model driven embedding will be integrated in later phases.[/italic]"
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<placeholder stego text>", encoding="utf-8")
        _rich_echo(f"\n[bold green]Placeholder stego text written to[/bold green] {output}")
    else:
        _rich_echo("\n<placeholder stego text>")


@main.command()
@click.option(
    "--stego",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to the stego text produced during encoding.",
)
@click.option(
    "--context",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="The same context file used during encoding.",
)
@click.option(
    "--mode",
    type=click.Choice(["arithmetic", "huffman", "bins"], case_sensitive=False),
    default="arithmetic",
    show_default=True,
    help="Embedding backend expected when decoding stego text.",
)
@click.option(
    "--model",
    type=str,
    default="gpt2-fa",
    show_default=True,
    help="Language model identifier to load for decoding.",
)
def decode(stego: Path, context: Path, mode: str, model: str) -> None:
    """Decode a hidden message from generated stego text (placeholder)."""

    try:
        stego_text = stego.read_text(encoding="utf-8")
        context_text = context.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - simple I/O guard
        raise click.ClickException(f"Failed to read text file: {exc}") from exc

    table = Table(title="decode parameters", header_style="bold magenta")
    table.add_column("Parameter", style="cyan", justify="right")
    table.add_column("Value", style="white")
    table.add_row("Stego", str(stego))
    table.add_row("Context", str(context))
    table.add_row("Mode", mode.lower())
    table.add_row("Model", model)

    _rich_echo("[bold blue]Decoding hidden message from stego text[/bold blue]")
    console.print(table)

    context_excerpt = context_text.strip().splitlines()[0][:120] if context_text.strip() else "(empty)"
    stego_excerpt = stego_text.strip().splitlines()[0][:120] if stego_text.strip() else "(empty)"

    _rich_echo(f"\nContext preview: [italic]{context_excerpt}...[/italic]")
    _rich_echo(f"Stego preview: [italic]{stego_excerpt}...[/italic]")

    _rich_echo(
        "\n[italic]Decoding logic using GPT-based arithmetic coding will arrive in subsequent iterations.[/italic]"
    )


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
        "bitarray": _import_with_version_hint("bitarray", "pip install bitarray"),
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
