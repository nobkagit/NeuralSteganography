"""Logging utilities for neuralstego."""

from __future__ import annotations

import logging
import os
from typing import Optional

DEFAULT_LOG_LEVEL = "INFO"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure the root logger for the application."""
    log_level = (level or os.getenv("NEURALSTEGO_LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
