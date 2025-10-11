"""Optional logistic-regression based detector wrapper."""

from __future__ import annotations

import importlib
import importlib.util
import pickle
from dataclasses import dataclass
from typing import List, Sequence

from .features import EXPECTED_FEATURES


@dataclass
class DetectionClassifier:
    """Wrap a scikit-learn logistic regression model for suspiciousness scoring."""

    model: object | None = None
    feature_order: Sequence[str] = EXPECTED_FEATURES

    def _require_sklearn(self) -> object:
        if importlib.util.find_spec("sklearn") is None:
            raise RuntimeError("scikit-learn is required for DetectionClassifier but is not installed")
        return importlib.import_module("sklearn.linear_model")

    def train(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> bytes:
        """Fit a logistic regression model and return a serialized representation."""

        linear_model = self._require_sklearn()
        classifier = linear_model.LogisticRegression(max_iter=1000)
        classifier.fit(X, y)
        self.model = classifier
        return pickle.dumps(classifier)

    def load(self, payload: bytes) -> None:
        """Load a serialized logistic regression model."""

        self.model = pickle.loads(payload)

    def _vectorize(self, features: dict[str, float]) -> List[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_order]

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return the suspiciousness probability for the provided feature mapping."""

        if self.model is None:
            raise RuntimeError("No model has been trained or loaded for DetectionClassifier")
        probabilities = self.model.predict_proba([self._vectorize(features)])
        # We assume label 1 corresponds to "suspicious".
        return float(probabilities[0][1])
