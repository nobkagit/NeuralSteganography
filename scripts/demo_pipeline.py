"""Run an end-to-end NeuralStego encode/decode demonstration with progress logs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run(title: str, args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print(f"\n[{title}] اجرای فرمان:")
    print(" ", " ".join(shlex.quote(part) for part in args))
    completed = subprocess.run(
        args,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(f"[{title}] خروجی استاندارد:\n{completed.stdout}")
    if completed.stderr:
        print(f"[{title}] خروجی خطا:\n{completed.stderr}")
    return completed


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"[+] فایل نوشته شد: {path}")
    print(f"    محتوا:\n{content}")


def _read_text(path: Path) -> str:
    data = path.read_text(encoding="utf-8")
    print(f"[+] خواندن فایل: {path}")
    print(f"    محتوا:\n{data}")
    return data


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    src_path = root_dir / "src"
    work_dir = Path("/tmp/neuralstego_demo")
    work_dir.mkdir(parents=True, exist_ok=True)

    secret_path = work_dir / "secret.txt"
    tokens_path = work_dir / "stego.json"
    recovered_path = work_dir / "recovered.txt"

    secret_message = "گزارش سری: جلسهٔ اصلی فردا ساعت ۱۵ برگزار می‌شود."
    password = "Pa$$w0rd"
    quality_flags = [
        "--quality",
        "top-k",
        "60",
        "--quality",
        "temperature",
        "0.8",
    ]

    _write_text(secret_path, secret_message)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_path) if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"

    encode_cmd = [
        sys.executable,
        "-m",
        "neuralstego",
        "encode",
        "-p",
        password,
        "-i",
        str(secret_path),
        "-o",
        str(tokens_path),
    ] + quality_flags
    _run("Encode", encode_cmd, env=env)
    _read_text(tokens_path)

    decode_cmd = [
        sys.executable,
        "-m",
        "neuralstego",
        "decode",
        "-p",
        password,
        "-i",
        str(tokens_path),
        "-o",
        str(recovered_path),
    ]
    _run("Decode", decode_cmd, env=env)
    recovered = _read_text(recovered_path)

    print("\n[نتیجه] مقایسهٔ پیام اصلی و بازیابی شده:")
    if recovered == secret_message:
        print("✅ پیام بازیابی شده دقیقاً با پیام اصلی مطابقت دارد.")
    else:
        print("⚠️ پیام بازیابی شده با پیام اصلی تفاوت دارد!")


if __name__ == "__main__":
    main()
