"""Tests for the NumPy decision-tree implementations."""

import sys
from pathlib import Path

import numpy as np
import pytest

ha_predictions_path = (
    Path(__file__).parent.parent.parent / "custom_components" / "ha_predictions"
)
sys.path.insert(0, str(ha_predictions_path))

from ml.decision_tree import (  # noqa: E402
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


class TestDecisionTreeClassifier:
    """Test binary classification."""

    def test_predicts_classes_and_probabilities(self) -> None:
        """Separate two classes and return class-one probabilities."""
        model = DecisionTreeClassifier()
        model.fit(
            np.array([[0.0], [1.0], [10.0], [11.0]]),
            np.array([0.0, 0.0, 1.0, 1.0]),
        )

        predictions, probabilities = model.predict(np.array([[0.5], [10.5]]))

        assert predictions.tolist() == [0.0, 1.0]
        assert probabilities.tolist() == [0.0, 1.0]

    def test_exports_feature_rules(self) -> None:
        """Expose readable rules with supplied feature names."""
        model = DecisionTreeClassifier()
        model.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

        rules = model.export_text(["temperature"])

        assert "temperature" in rules
        assert "class" in rules


class TestDecisionTreeRegressor:
    """Test numeric target regression."""

    def test_predicts_training_regions(self) -> None:
        """Predict numeric values from fitted tree leaves."""
        model = DecisionTreeRegressor()
        x = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.arange(8, dtype=float) ** 2
        model.fit(x, y)

        predictions, probabilities = model.predict(np.array([[2.0], [6.0]]))

        assert predictions.tolist() == [4.0, 36.0]
        assert probabilities is None

    def test_rejects_invalid_training_shapes(self) -> None:
        """Reject mismatched feature and target arrays."""
        model = DecisionTreeRegressor()

        with pytest.raises(ValueError, match="invalid dimensions"):
            model.fit(np.array([[1.0], [2.0]]), np.array([1.0]))
