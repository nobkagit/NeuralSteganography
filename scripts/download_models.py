#!/usr/bin/env python
"""دریافت تنبل (lazy) مدل‌های موردنیاز و گزارش مسیر کش."""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

try:
    from transformers.utils import hub
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "transformers نصب نشده است. ابتدا `pip install transformers` یا `pip install -e .[dev]` را اجرا کنید."
    ) from exc

DEFAULT_MODEL = "HooshvareLab/gpt2-fa"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="شناسهٔ مدل روی HuggingFace Hub (پیش‌فرض: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="شاخه یا تگ موردنظر برای دریافت (پیش‌فرض: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=None,
        help="مسیر دلخواه برای کش مدل. در صورت عدم تعیین از مقدار پیش‌فرض transformers استفاده می‌شود.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    cache_dir = args.cache_dir
    if cache_dir is not None:
        cache_dir = cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir_str: Optional[str] = str(cache_dir)
    else:
        cache_dir_str = None

    print(f"[download_models] دریافت تنبل مدل {args.model} (revision={args.revision})...")
    snapshot_path = pathlib.Path(
        hub.snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=cache_dir_str,
            resume_download=True,
        )
    )

    if cache_dir is None:
        default_cache = pathlib.Path(hub.TRANSFORMERS_CACHE).expanduser().resolve()
        print(f"[download_models] از کش پیش‌فرض transformers استفاده شد: {default_cache}")
    else:
        print(f"[download_models] کش سفارشی مورد استفاده: {cache_dir}")

    print(f"[download_models] مسیر اسنپ‌شات محلی: {snapshot_path}")
    print("[download_models] انجام شد؛ فایل‌ها تنها در کش نگهداری می‌شوند.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
