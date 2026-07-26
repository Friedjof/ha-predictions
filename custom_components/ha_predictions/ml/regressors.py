"""Regressor adapters used by the model manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

if TYPE_CHECKING:
    import numpy as np


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


class DecisionTreeRegressor:
    """Adapt scikit-learn's decision-tree regression to the local contract."""

    def __init__(self) -> None:
        """Initialize the regressor."""
        self.estimator = SklearnDecisionTreeRegressor(random_state=0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the regressor."""
        self.estimator.fit(x, y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, None]:
        """Return numeric predictions without class probabilities."""
        return self.estimator.predict(x), None

    def __str__(self) -> str:
        """Generate a string representation of the regressor."""
        return str(self.estimator)
