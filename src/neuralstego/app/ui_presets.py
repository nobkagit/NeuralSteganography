"""UI presets for the Gradio application."""

from __future__ import annotations

from typing import Dict, List

SEED_PRESETS: List[str] = [
    "در یک گفت‌وگوی کوتاه درباره‌ی فناوری و اخبار روز صحبت می‌کنیم.",
    "امروز هوا کمی سردتر است و درباره‌ی کتاب‌های جدید صحبت می‌کنیم.",
    "گزارشی کوتاه از رویدادهای خبری اخیر ارائه می‌دهیم.",
]

QUALITY_PRESETS: Dict[str, Dict[str, float]] = {
    "Balanced": {"top_k": 60, "temperature": 0.8},
    "HighQuality": {"top_k": 80, "temperature": 0.7},
    "HighCapacity": {"top_k": 40, "temperature": 0.9},
}

__all__ = ["SEED_PRESETS", "QUALITY_PRESETS"]
