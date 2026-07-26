"""Classifier adapters used by the model manager."""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier


class DecisionTreeClassifier:
    """Adapt scikit-learn's decision tree to the local classifier contract."""

    def __init__(self) -> None:
        """Initialize the classifier."""
        self.estimator = SklearnDecisionTreeClassifier(random_state=0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier."""
        self.estimator.fit(x, y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted classes and probabilities for class one."""
        predictions = self.estimator.predict(x)
        class_probabilities = self.estimator.predict_proba(x)
        class_one = np.where(self.estimator.classes_ == 1)[0]
        probabilities = (
            class_probabilities[:, class_one[0]]
            if len(class_one) > 0
            else np.zeros(len(predictions))
        )
        return predictions, probabilities

    def __str__(self) -> str:
        """Generate a string representation of the classifier."""
        return str(self.estimator)
