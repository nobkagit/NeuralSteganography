#!/usr/bin/env python3
"""Simplified CLI entrypoint for the NeuralSteganography pipeline."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import click

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neuralstego.api import cover_generate, cover_reveal  # noqa: E402
from neuralstego.exceptions import NeuralStegoError, QualityGateError  # noqa: E402
from neuralstego.lm import load_lm  # noqa: E402

DEFAULT_SEED = "در یک گفت‌وگوی کوتاه درباره‌ی فناوری و اخبار روز صحبت می‌کنیم."


def _parse_quality(entries: Iterable[str]) -> Dict[str, object]:
    """Parse ``--quality`` CLI options into a dictionary."""

    parsed: Dict[str, object] = {}
    for entry in entries:
        if "=" not in entry:
            raise click.BadParameter(
                "quality entries must be in the form key=value", param_hint="--quality"
            )
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise click.BadParameter("quality key may not be empty", param_hint="--quality")
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed[key] = lowered == "true"
            continue
        try:
            parsed[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            parsed[key] = float(value)
            continue
        except ValueError:
            parsed[key] = value
    return parsed


@click.command()
@click.option("--secret", "secret_message", prompt=True, help="متن محرمانه‌ای که باید پنهان شود.")
@click.option("--model", "model_name", default="mock", show_default=True, help="نام مدل زبانی (مانند gpt2-fa یا mock).")
@click.option("--seed-text", default=DEFAULT_SEED, show_default=True, help="متن اولیه برای شروع تولید.")
@click.option("--chunk-bytes", default=256, show_default=True, help="حجم هر تکه باینری قبل از رمزگذاری.")
@click.option("--no-crc", is_flag=True, help="غیرفعال کردن CRC در مرحله بسته‌بندی.")
@click.option(
    "--ecc/--no-ecc",
    default=True,
    show_default=True,
    help="فعال یا غیرفعال‌سازی Reed-Solomon ECC.",
)
@click.option("--nsym", default=10, show_default=True, help="تعداد نمادهای ECC در صورت فعال بودن.")
@click.option(
    "--quality",
    multiple=True,
    help="پارامترهای کیفیت (مانند temp=0.9 یا top_k=60).",
)
def main(
    secret_message: str,
    model_name: str,
    seed_text: str,
    chunk_bytes: int,
    no_crc: bool,
    ecc: bool,
    nsym: int,
    quality: Tuple[str, ...],
) -> None:
    """Embed *secret_message* into a cover text and immediately recover it."""

    click.echo("[مرحله ۱] در حال بارگذاری مدل زبانی…", err=True)
    try:
        lm = load_lm(model_name)
    except Exception as exc:  # pragma: no cover - defensive: optional deps
        raise click.ClickException(f"خطا در بارگذاری مدل '{model_name}': {exc}") from exc

    quality_args = _parse_quality(quality) if quality else {}
    ecc_mode = "rs" if ecc else "none"
    use_crc = not no_crc

    click.echo("[مرحله ۲] تولید متن پوششی…", err=True)
    try:
        cover_text = cover_generate(
            secret_message.encode("utf-8"),
            seed_text=seed_text,
            quality=quality_args or None,
            chunk_bytes=chunk_bytes,
            use_crc=use_crc,
            ecc=ecc_mode,
            nsym=nsym,
            lm=lm,
        )
    except QualityGateError as exc:
        details = "، ".join(exc.reasons) if exc.reasons else "علت نامشخص"
        raise click.ClickException(f"کاور تولید نشد: {details}") from exc
    except NeuralStegoError as exc:
        raise click.ClickException(f"خطا در مرحله نهان‌سازی: {exc}") from exc

    click.echo("\n[متن پوششی]")
    click.echo(cover_text)

    click.echo("\n[مرحله ۳] بازیابی پیام محرمانه…", err=True)
    try:
        recovered = cover_reveal(
            cover_text,
            seed_text=seed_text,
            quality=quality_args or None,
            use_crc=use_crc,
            ecc=ecc_mode,
            nsym=nsym,
            lm=lm,
        )
    except NeuralStegoError as exc:
        raise click.ClickException(f"بازیابی پیام با خطا روبه‌رو شد: {exc}") from exc

    try:
        recovered_text = recovered.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise click.ClickException("پیام بازیابی‌شده قابل تبدیل به UTF-8 نیست.") from exc

    click.echo("\n[پیام بازیابی‌شده]")
    click.echo(recovered_text)


if __name__ == "__main__":
    main()
