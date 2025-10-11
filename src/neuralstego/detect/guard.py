"""Rule-based guard that decides whether a cover text passes quality thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..metrics import (
    LMScorer,
    avg_entropy,
    avg_sentence_len,
    ngram_repeat_ratio,
    type_token_ratio,
)
from .features import extract_features


@dataclass
class GuardResult:
    """Structured response returned by :class:`QualityGuard`."""

    passed: bool
    reasons: List[str]
    metrics: Dict[str, float]
    detector_score: Optional[float] = None


@dataclass
class QualityGuard:
    """Evaluate cover text quality using heuristic thresholds and optional models."""

    lm_scorer: LMScorer = field(default_factory=LMScorer)
    classifier: Optional[object] = None

    def _collect_metrics(self, text: str) -> Dict[str, float]:
        lm_metrics = self.lm_scorer.score(text)
        stats = {
            "ngram_repeat_ratio": ngram_repeat_ratio(text),
            "type_token_ratio": type_token_ratio(text),
            "avg_sentence_len": avg_sentence_len(text),
            "avg_entropy": avg_entropy(text, self.lm_scorer),
        }
        combined = {**lm_metrics, **stats}
        return combined

    def _evaluate_rules(self, features: Dict[str, float], thresholds: Dict[str, float]) -> List[str]:
        reasons: List[str] = []
        if "max_ppl" in thresholds and features["ppl"] > thresholds["max_ppl"]:
            reasons.append(f"ppl {features['ppl']:.2f} exceeds max {thresholds['max_ppl']:.2f}")
        if "max_ngram_repeat" in thresholds and features["ngram_repeat_ratio"] > thresholds["max_ngram_repeat"]:
            reasons.append(
                "ngram repeat ratio "
                f"{features['ngram_repeat_ratio']:.2f} exceeds {thresholds['max_ngram_repeat']:.2f}"
            )
        if "min_ttr" in thresholds and features["type_token_ratio"] < thresholds["min_ttr"]:
            reasons.append(
                f"type-token ratio {features['type_token_ratio']:.2f} below {thresholds['min_ttr']:.2f}"
            )
        if "max_avg_entropy" in thresholds and features["avg_entropy"] > thresholds["max_avg_entropy"]:
            reasons.append(
                f"avg entropy {features['avg_entropy']:.2f} exceeds {thresholds['max_avg_entropy']:.2f}"
            )
        if "min_avg_sentence_len" in thresholds and features["avg_sentence_len"] < thresholds["min_avg_sentence_len"]:
            reasons.append(
                "avg sentence length "
                f"{features['avg_sentence_len']:.2f} below {thresholds['min_avg_sentence_len']:.2f}"
            )
        return reasons

    def evaluate(self, text: str, thresholds: Dict[str, float]) -> GuardResult:
        """Evaluate the provided text against configured thresholds."""

        metrics = self._collect_metrics(text)
        features = extract_features(metrics)
        reasons = self._evaluate_rules(features, thresholds)
        detector_score: Optional[float] = None
        if self.classifier is not None and hasattr(self.classifier, "predict_proba"):
            detector_score = float(self.classifier.predict_proba(features))
            metrics["detector_score"] = detector_score
            if "max_detector_score" in thresholds and detector_score > thresholds["max_detector_score"]:
                reasons.append(
                    "detector score "
                    f"{detector_score:.2f} exceeds {thresholds['max_detector_score']:.2f}"
                )
        passed = not reasons
        return GuardResult(passed, reasons, metrics, detector_score)
