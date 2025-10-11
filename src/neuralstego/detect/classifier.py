"""Optional logistic-regression based detector wrapper."""

from __future__ import annotations

import importlib
import importlib.util
import pickle
from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence, cast

from .features import EXPECTED_FEATURES


class _LogisticRegressionLike(Protocol):
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> Any:  # pragma: no cover - protocol
        ...

    def predict_proba(self, X: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:  # pragma: no cover - protocol
        ...


class _LogisticRegressionFactory(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> _LogisticRegressionLike:  # pragma: no cover - protocol
        ...


class _LinearModelModule(Protocol):
    LogisticRegression: _LogisticRegressionFactory


@dataclass
class DetectionClassifier:
    """Wrap a scikit-learn logistic regression model for suspiciousness scoring."""

    model: _LogisticRegressionLike | None = None
    feature_order: Sequence[str] = EXPECTED_FEATURES

    def _require_sklearn(self) -> _LinearModelModule:
        if importlib.util.find_spec("sklearn") is None:
            raise RuntimeError("scikit-learn is required for DetectionClassifier but is not installed")
        module = importlib.import_module("sklearn.linear_model")
        logistic = getattr(module, "LogisticRegression", None)
        if not callable(logistic):
            raise RuntimeError("scikit-learn.linear_model.LogisticRegression is unavailable")
        return cast(_LinearModelModule, module)

    def train(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> bytes:
        """Fit a logistic regression model and return a serialized representation."""

        linear_model = self._require_sklearn()
        factory = cast(_LogisticRegressionFactory, linear_model.LogisticRegression)
        classifier = factory(max_iter=1000)
        classifier.fit(X, y)
        self.model = classifier
        return pickle.dumps(classifier)

    def load(self, payload: bytes) -> None:
        """Load a serialized logistic regression model."""

        self.model = cast(_LogisticRegressionLike, pickle.loads(payload))

    def _vectorize(self, features: dict[str, float]) -> List[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_order]

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return the suspiciousness probability for the provided feature mapping."""

        if self.model is None:
            raise RuntimeError("No model has been trained or loaded for DetectionClassifier")
        probabilities = self.model.predict_proba([self._vectorize(features)])
        # We assume label 1 corresponds to "suspicious".
        return float(probabilities[0][1])
