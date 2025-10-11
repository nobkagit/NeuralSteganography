"""Gradio-based user interface for NeuralSteganography."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Optional, Tuple

import gradio as gr

from ..api import DEFAULT_GATE_THRESHOLDS, cover_generate, cover_reveal
from ..exceptions import ConfigurationError, MissingChunksError, NeuralStegoError, QualityGateError
from ..lm import load_lm
from .ui_presets import QUALITY_PRESETS, SEED_PRESETS

DeviceChoice = str

_DEFAULT_DEVICE = os.getenv("NEURALSTEGO_DEVICE", "auto")
_DEVICE_CHOICES: Tuple[DeviceChoice, ...] = ("auto", "cpu", "cuda")

_LM_CACHE: Dict[Tuple[str, Optional[str]], Any] = {}
_LM_LOCK = Lock()


def _write_temp_file(filename: str, data: bytes) -> str:
    """Persist *data* to a temporary file and return its path."""

    temp_dir = Path(tempfile.mkdtemp(prefix="neuralstego_ui_"))
    file_path = temp_dir / filename
    file_path.write_bytes(data)
    return str(file_path)


def _normalise_device(device: Optional[str]) -> str:
    if not device:
        return "auto"
    return device


def _resolve_lm(device: Optional[str], progress: Optional[gr.Progress] = None) -> Any:
    """Load (and cache) the GPT2-fa language model adapter."""

    normalised = _normalise_device(device)
    cache_key = ("gpt2-fa", normalised)

    with _LM_LOCK:
        cached = _LM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if progress is not None:
        progress(0, desc="در حال دانلود مدل GPT2-fa…")

    lm = load_lm("gpt2-fa", device=None if normalised == "auto" else normalised)

    with _LM_LOCK:
        _LM_CACHE[cache_key] = lm

    return lm


def read_secret_input(secret_text: str, secret_file: Any) -> str:
    """Return the secret payload as a UTF-8 string."""

    if secret_file:
        path: Optional[Path] = None
        if isinstance(secret_file, (str, os.PathLike)):
            path = Path(secret_file)
        elif hasattr(secret_file, "name"):
            try:
                path = Path(secret_file.name)  # type: ignore[arg-type]
            except TypeError:
                path = None
        elif isinstance(secret_file, Mapping):
            potential_path = secret_file.get("path") or secret_file.get("name")
            if isinstance(potential_path, str):
                path = Path(potential_path)
            data_obj = secret_file.get("data")
            if isinstance(data_obj, (bytes, bytearray)):
                try:
                    return bytes(data_obj).decode("utf-8")
                except UnicodeDecodeError as exc:  # pragma: no cover - UI feedback path
                    raise ValueError("فایل محرمانه باید با UTF-8 رمزگذاری شده باشد.") from exc
            if isinstance(data_obj, str):
                return data_obj

        if path is not None and path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("فایل محرمانه باید با UTF-8 رمزگذاری شده باشد.") from exc

        raise ValueError("قادر به خواندن فایل محرمانه نیستیم؛ لطفاً دوباره تلاش کنید.")

    if secret_text and secret_text.strip():
        return secret_text

    raise ValueError("برای جاسازی، متن یا فایل محرمانه لازم است.")


def get_quality_params(profile: str, top_k: Any, temperature: Any) -> Dict[str, float]:
    """Build the quality parameter dictionary for cover generation."""

    params: Dict[str, float] = dict(QUALITY_PRESETS.get(profile, {}))

    if top_k not in (None, ""):
        try:
            params["top_k"] = float(top_k)
        except (TypeError, ValueError) as exc:
            raise ValueError("مقدار top_k باید عددی باشد.") from exc
    if temperature not in (None, ""):
        try:
            params["temperature"] = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("مقدار temperature باید عددی باشد.") from exc

    return params


def _gate_thresholds(
    max_ppl: Optional[float],
    max_ngram_repeat: Optional[float],
    min_ttr: Optional[float],
) -> Dict[str, float]:
    thresholds = dict(DEFAULT_GATE_THRESHOLDS)
    if max_ppl is not None:
        thresholds["max_ppl"] = float(max_ppl)
    if max_ngram_repeat is not None:
        thresholds["max_ngram_repeat"] = float(max_ngram_repeat)
    if min_ttr is not None:
        thresholds["min_ttr"] = float(min_ttr)
    return thresholds


def _parse_int(value: Any, *, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def on_generate_cover(
    secret_text: str,
    secret_file: Any,
    password: str,
    seed_preset: str,
    seed_override: str,
    quality_profile: str,
    top_k: Any,
    temperature: Any,
    chunk_bytes: Any,
    crc: str,
    ecc: str,
    nsym: Any,
    enable_quality_gate: bool,
    max_ppl: Optional[float],
    max_ngram_repeat: Optional[float],
    min_ttr: Optional[float],
    regen_attempts: Any,
    device: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    """Handle cover generation button events."""

    if password:
        gr.Warning("رمز عبور فعلاً فقط برای اهداف نمایشی است و اعمال نمی‌شود.")

    try:
        secret = read_secret_input(secret_text, secret_file)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    seed = seed_override.strip() or seed_preset or ""
    try:
        quality = get_quality_params(quality_profile, top_k, temperature)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    chunk_size = _parse_int(chunk_bytes, fallback=256)
    nsym_value = _parse_int(nsym, fallback=10)
    regen_value = max(_parse_int(regen_attempts, fallback=2), 0)

    gate_thresholds = _gate_thresholds(max_ppl, max_ngram_repeat, min_ttr)
    gate_kwargs: Dict[str, Any] = {}
    if enable_quality_gate:
        gate_kwargs["quality_gate"] = True
        gate_kwargs["gate_thresholds"] = gate_thresholds
        gate_kwargs["regen_attempts"] = regen_value
    else:
        gate_kwargs["quality_gate"] = False

    lm = _resolve_lm(device, progress)

    if progress is not None:
        progress(0.4, desc="در حال تولید کاور…")

    try:
        cover_text = cover_generate(
            secret,
            seed_text=seed,
            quality=quality,
            chunk_bytes=chunk_size,
            use_crc=crc == "on",
            ecc=ecc,
            nsym=nsym_value,
            lm=lm,
            **gate_kwargs,
        )
        gate_error: QualityGateError | None = None
    except QualityGateError as exc:
        cover_text = exc.cover_text
        gate_error = exc
    except MissingChunksError as exc:
        raise gr.Error("برخی بسته‌ها مفقود شده‌اند؛ بازتولید با پارامترهای متفاوت را امتحان کنید.") from exc
    except (ConfigurationError, NeuralStegoError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    if gate_error is not None:
        message_parts = ["نگهبان کیفیت کاور تولیدی را رد کرد."]
        if gate_error.reasons:
            message_parts.append("؛ ".join(gate_error.reasons))
        gr.Warning(" ".join(message_parts))

    if progress is not None:
        progress(0.8, desc="در حال آماده‌سازی فایل خروجی…")

    download_path = _write_temp_file("cover.txt", cover_text.encode("utf-8"))
    return cover_text, download_path


def on_reveal_secret(
    cover_text: str,
    password: str,
    seed_preset: str,
    seed_override: str,
    chunk_bytes: Any,
    crc: str,
    ecc: str,
    nsym: Any,
    device: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    """Handle secret reveal button events."""

    if password:
        gr.Warning("رمز عبور در مسیر افشا فعلاً پشتیبانی نمی‌شود.")

    seed = seed_override.strip() or seed_preset or ""
    nsym_value = _parse_int(nsym, fallback=10)

    lm = _resolve_lm(device, progress)

    if progress is not None:
        progress(0.4, desc="در حال بازیابی پیام…")

    try:
        secret_bytes = cover_reveal(
            cover_text,
            seed_text=seed,
            quality=None,
            use_crc=crc == "on",
            ecc=ecc,
            nsym=nsym_value,
            lm=lm,
        )
    except MissingChunksError as exc:
        partial_path = _write_temp_file("secret.partial.txt", exc.partial_payload)
        gr.Warning("برخی بسته‌ها گم شده‌اند؛ بخشی از متن بازیابی شد.")
        try:
            secret_text = exc.partial_payload.decode("utf-8")
        except UnicodeDecodeError:
            raise gr.Error("بخشی از پیام بازیابی شد اما رمزگشایی UTF-8 آن ناممکن است.") from exc
        return secret_text, partial_path
    except (ConfigurationError, NeuralStegoError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    try:
        secret_text = secret_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise gr.Error("پیام بازیابی‌شده UTF-8 معتبر نیست.") from exc

    if progress is not None:
        progress(0.8, desc="در حال آماده‌سازی فایل خروجی…")

    download_path = _write_temp_file("secret.txt", secret_bytes)
    return secret_text, download_path


def _bool_arg(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in {"1", "true", "yes", "on"}:
        return True
    if value_lower in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("boolean flag must be one of true/false/yes/no/on/off/1/0")


def build_interface(default_device: str = _DEFAULT_DEVICE) -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    with gr.Blocks(title="NeuralStego UI", css=".important-note {color: #a67900; font-weight: bold;}") as demo:
        gr.Markdown("""# NeuralStego UI\nمحیط تعاملی برای جاسازی و بازیابی پیام‌های محرمانه.""")

        with gr.Accordion("تنظیمات سامانه", open=False):
            device_dropdown = gr.Dropdown(
                choices=list(_DEVICE_CHOICES),
                value=default_device if default_device in _DEVICE_CHOICES else "auto",
                label="انتخاب دستگاه",
            )
            gr.Markdown(
                "در صورت انتخاب `auto`، برنامه به‌طور خودکار بین CPU و CUDA تصمیم می‌گیرد (در صورت موجود بودن)."
            )

        with gr.Tabs():
            with gr.TabItem("تولید کاور"):
                secret_text = gr.Textbox(label="متن محرمانه", lines=6, placeholder="متن را اینجا وارد کنید…")
                secret_file = gr.File(label="یا فایل متنی UTF-8 بارگذاری کنید")
                password = gr.Textbox(label="رمز عبور (اختیاری)", type="password")

                with gr.Row():
                    seed_preset = gr.Dropdown(
                        choices=SEED_PRESETS,
                        value=SEED_PRESETS[0],
                        label="پیش‌فرض seed",
                    )
                    seed_override = gr.Textbox(label="seed دستی (در صورت نیاز)")

                quality_profile = gr.Dropdown(
                    choices=list(QUALITY_PRESETS.keys()),
                    value="Balanced",
                    label="پروفایل کیفیت",
                )

                with gr.Accordion("پارامترهای پیشرفته", open=False):
                    with gr.Row():
                        top_k = gr.Textbox(label="top_k (override)")
                        temperature = gr.Textbox(label="temperature (override)")
                    chunk_bytes = gr.Number(label="اندازه هر chunk (بایت)", value=256, precision=0)
                    crc = gr.Radio(["on", "off"], label="CRC32", value="on")
                    ecc = gr.Radio(["rs", "none"], label="Error-Correction", value="rs")
                    nsym = gr.Number(label="نمادهای RS (nsym)", value=10, precision=0)

                with gr.Accordion("Quality Gate", open=False):
                    enable_quality_gate = gr.Checkbox(label="فعال‌سازی نگهبان کیفیت", value=True)
                    max_ppl = gr.Number(label="حداکثر Perplexity", value=DEFAULT_GATE_THRESHOLDS["max_ppl"])
                    max_ngram_repeat = gr.Number(
                        label="حداکثر تکرار n-gram",
                        value=DEFAULT_GATE_THRESHOLDS["max_ngram_repeat"],
                    )
                    min_ttr = gr.Number(label="حداقل نسبت نوع به توکن", value=DEFAULT_GATE_THRESHOLDS["min_ttr"])
                    regen_attempts = gr.Number(label="تعداد تلاش مجدد", value=2, precision=0)

                generate_button = gr.Button("تولید کاور", variant="primary")
                cover_output = gr.Textbox(label="متن کاور", lines=8, interactive=False)
                cover_download = gr.File(label="دانلود cover.txt")

            with gr.TabItem("افشای پیام"):
                reveal_cover = gr.Textbox(label="متن کاور", lines=8)
                reveal_password = gr.Textbox(label="رمز عبور (اختیاری)", type="password")
                with gr.Row():
                    reveal_seed_preset = gr.Dropdown(
                        choices=SEED_PRESETS,
                        value=SEED_PRESETS[0],
                        label="پیش‌فرض seed",
                    )
                    reveal_seed_override = gr.Textbox(label="seed دستی")

                gr.Markdown(
                    "<div class=\"important-note\">پارامترهای زیر باید با تنظیمات مرحلهٔ encode هماهنگ باشند؛ در غیر این صورت بازیابی شکست می‌خورد.</div>",
                    elem_id="decode-note",
                )

                with gr.Accordion("پارامترهای پیشرفته", open=False):
                    reveal_chunk_bytes = gr.Number(label="اندازه chunk (برای یادداشت)", value=256, precision=0)
                    reveal_crc = gr.Radio(["on", "off"], label="CRC32", value="on")
                    reveal_ecc = gr.Radio(["rs", "none"], label="Error-Correction", value="rs")
                    reveal_nsym = gr.Number(label="نمادهای RS (nsym)", value=10, precision=0)

                reveal_button = gr.Button("افشای پیام", variant="primary")
                secret_output = gr.Textbox(label="متن محرمانه", lines=6, interactive=False)
                secret_download = gr.File(label="دانلود secret.txt")

        generate_button.click(
            on_generate_cover,
            inputs=[
                secret_text,
                secret_file,
                password,
                seed_preset,
                seed_override,
                quality_profile,
                top_k,
                temperature,
                chunk_bytes,
                crc,
                ecc,
                nsym,
                enable_quality_gate,
                max_ppl,
                max_ngram_repeat,
                min_ttr,
                regen_attempts,
                device_dropdown,
            ],
            outputs=[cover_output, cover_download],
        )

        reveal_button.click(
            on_reveal_secret,
            inputs=[
                reveal_cover,
                reveal_password,
                reveal_seed_preset,
                reveal_seed_override,
                reveal_chunk_bytes,
                reveal_crc,
                reveal_ecc,
                reveal_nsym,
                device_dropdown,
            ],
            outputs=[secret_output, secret_download],
        )

    return demo


def launch(*, port: int = 7860, share: bool = False, server_name: Optional[str] = None, device: Optional[str] = None) -> None:
    """Launch the Gradio application."""

    interface = build_interface(default_device=device or _DEFAULT_DEVICE)
    launch_kwargs: Dict[str, Any] = {"server_port": port, "share": share}
    if server_name:
        launch_kwargs["server_name"] = server_name
    interface.launch(**launch_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the NeuralStego Gradio UI")
    parser.add_argument("--port", type=int, default=7860, help="پورت سرور Gradio")
    parser.add_argument("--share", type=_bool_arg, default=False, help="اشتراک‌گذاری gradio.live را فعال کنید")
    parser.add_argument("--server-name", help="نام یا آدرس سرور برای binding")
    parser.add_argument("--device", choices=_DEVICE_CHOICES, help="دستگاه اجرای مدل")
    args = parser.parse_args()
    launch(port=args.port, share=args.share, server_name=args.server_name, device=args.device)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
