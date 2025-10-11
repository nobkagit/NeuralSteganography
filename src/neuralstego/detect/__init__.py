"""Detection utilities for assessing cover text quality."""

from .features import EXPECTED_FEATURES, extract_features
from .guard import GuardResult, QualityGuard
from .classifier import DetectionClassifier

__all__ = [
    "EXPECTED_FEATURES",
    "DetectionClassifier",
    "GuardResult",
    "QualityGuard",
    "extract_features",
]
