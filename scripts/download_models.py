#!/usr/bin/env python
"""دانلود یا آماده‌سازی مدل‌های مورد نیاز برای NeuralStego."""
from __future__ import annotations

import argparse
import pathlib
import sys

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - راهنما برای کاربر
    raise SystemExit(
        "transformers نصب نشده است. ابتدا `pip install -e .` را اجرا کنید."
    ) from exc

DEFAULT_MODEL = "HooshvareLab/gpt2-fa"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="شناسه مدل در HuggingFace Hub (پیش‌فرض: %(default)s)",
    )
    parser.add_argument(
        "--target",
        type=pathlib.Path,
        default=pathlib.Path("models"),
        help="مسیر ذخیره مدل (پیش‌فرض: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.target.mkdir(parents=True, exist_ok=True)
    print(f"[download_models] در حال دریافت {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    save_path = args.target / args.model.split("/")[-1]
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"[download_models] مدل در {save_path} ذخیره شد.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
