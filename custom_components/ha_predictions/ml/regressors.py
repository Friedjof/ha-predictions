"""Regressor adapters used by the model manager."""

from __future__ import annotations

import numpy as np


class LinearRegressor:
    """Ordinary least-squares linear regression implemented with NumPy."""

    def __init__(self) -> None:
        """Initialize the regressor."""
        self.coefficients: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the regressor."""
        x_with_intercept = np.column_stack((np.ones(len(x)), x))
        self.coefficients = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, None]:
        """Return numeric predictions without class probabilities."""
        if self.coefficients is None:
            msg = "Linear regressor has not been trained"
            raise ValueError(msg)
        x_with_intercept = np.column_stack((np.ones(len(x)), x))
        return x_with_intercept @ self.coefficients, None

    def __str__(self) -> str:
        """Generate a string representation of the regressor."""
        return f"LinearRegressor(coefficients={self.coefficients})"
