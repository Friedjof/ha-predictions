"""Regressor adapters used by the model manager."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


class LinearRegressor:
    """Adapt scikit-learn's linear regression to the local model contract."""

    def __init__(self) -> None:
        """Initialize the regressor."""
        self.estimator = SklearnLinearRegression()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the regressor."""
        self.estimator.fit(x, y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, None]:
        """Return numeric predictions without class probabilities."""
        return self.estimator.predict(x), None

    def __str__(self) -> str:
        """Generate a string representation of the regressor."""
        return str(self.estimator)
