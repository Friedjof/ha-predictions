"""NumPy decision-tree implementations for classification and regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_DIMENSIONS = 2
MIN_SAMPLES_SPLIT = 2


@dataclass
class _Node:
    """A decision node or leaf in a fitted tree."""

    value: float
    probability: float | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        """Return whether this node contains a prediction."""
        return self.feature_index is None


class _DecisionTree:
    """Shared CART-style tree implementation."""

    def __init__(self, *, classification: bool, max_depth: int = 8) -> None:
        self.classification = classification
        self.max_depth = max_depth
        self.root: _Node | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit a decision tree to numeric feature and target arrays."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if (
            x.ndim != FEATURE_DIMENSIONS
            or y.ndim != 1
            or len(x) != len(y)
            or len(y) == 0
        ):
            msg = "Decision-tree training data has invalid dimensions"
            raise ValueError(msg)
        self.root = self._build_tree(x, y, depth=0)

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        leaf = self._make_leaf(y)
        if (
            depth >= self.max_depth
            or len(y) < MIN_SAMPLES_SPLIT
            or self._impurity(y) == 0
        ):
            return leaf

        split = self._best_split(x, y)
        if split is None:
            return leaf

        feature_index, threshold = split
        left_mask = x[:, feature_index] <= threshold
        return _Node(
            value=leaf.value,
            probability=leaf.probability,
            feature_index=feature_index,
            threshold=threshold,
            left=self._build_tree(x[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(x[~left_mask], y[~left_mask], depth + 1),
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        parent_impurity = self._impurity(y)
        best_impurity = parent_impurity
        best_split: tuple[int, float] | None = None

        for feature_index in range(x.shape[1]):
            order = np.argsort(x[:, feature_index], kind="stable")
            values = x[order, feature_index]
            y_sorted = y[order]
            positions = np.flatnonzero(
                (values[:-1] < values[1:])
                & np.isfinite(values[:-1])
                & np.isfinite(values[1:])
            )
            if len(positions) == 0:
                continue

            impurities = self._split_impurities(y_sorted, positions)
            candidate_index = int(np.argmin(impurities))
            candidate_impurity = float(impurities[candidate_index])
            if candidate_impurity >= best_impurity - 1e-12:
                continue

            position = int(positions[candidate_index])
            left_value = values[position]
            right_value = values[position + 1]
            best_impurity = candidate_impurity
            best_split = (
                feature_index,
                float(left_value + (right_value - left_value) / 2),
            )

        return best_split

    def _split_impurities(
        self, y_sorted: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        left_count = positions + 1
        right_count = len(y_sorted) - left_count

        if self.classification:
            cumulative_positive = np.cumsum(y_sorted == 1)
            left_positive = cumulative_positive[positions]
            right_positive = cumulative_positive[-1] - left_positive
            left_ratio = left_positive / left_count
            right_ratio = right_positive / right_count
            left_impurity = 2 * left_ratio * (1 - left_ratio)
            right_impurity = 2 * right_ratio * (1 - right_ratio)
            return (
                left_count * left_impurity + right_count * right_impurity
            ) / len(y_sorted)

        cumulative_sum = np.cumsum(y_sorted)
        cumulative_square_sum = np.cumsum(y_sorted**2)
        left_sum = cumulative_sum[positions]
        right_sum = cumulative_sum[-1] - left_sum
        left_square_sum = cumulative_square_sum[positions]
        right_square_sum = cumulative_square_sum[-1] - left_square_sum
        left_error = left_square_sum - left_sum**2 / left_count
        right_error = right_square_sum - right_sum**2 / right_count
        return (left_error + right_error) / len(y_sorted)

    def _impurity(self, y: np.ndarray) -> float:
        if self.classification:
            positive_ratio = float(np.mean(y == 1))
            return 2 * positive_ratio * (1 - positive_ratio)
        return float(np.var(y))

    def _make_leaf(self, y: np.ndarray) -> _Node:
        if self.classification:
            values, counts = np.unique(y, return_counts=True)
            prediction = float(values[int(np.argmax(counts))])
            return _Node(
                value=prediction,
                probability=float(np.mean(y == 1)),
            )
        return _Node(value=float(np.mean(y)))

    def _leaf_for_row(self, row: np.ndarray) -> _Node:
        if self.root is None:
            msg = "Decision tree has not been trained"
            raise ValueError(msg)

        node = self.root
        while not node.is_leaf:
            if (
                node.feature_index is None
                or node.threshold is None
                or node.left is None
                or node.right is None
            ):
                msg = "Decision tree contains an invalid node"
                raise ValueError(msg)
            node = (
                node.left
                if row[node.feature_index] <= node.threshold
                else node.right
            )
        return node

    def export_text(self, feature_names: list[str]) -> str:
        """Return human-readable rules for the fitted tree."""
        if self.root is None:
            return ""
        lines: list[str] = []
        self._append_rules(self.root, feature_names, lines, depth=0)
        return "\n".join(lines)

    def _append_rules(
        self,
        node: _Node,
        feature_names: list[str],
        lines: list[str],
        depth: int,
    ) -> None:
        prefix = "|   " * depth
        if node.is_leaf:
            label = "class" if self.classification else "value"
            lines.append(f"{prefix}|--- {label}: {node.value:.6g}")
            return

        if (
            node.feature_index is None
            or node.threshold is None
            or node.left is None
            or node.right is None
        ):
            return
        feature_name = (
            feature_names[node.feature_index]
            if node.feature_index < len(feature_names)
            else f"feature_{node.feature_index}"
        )
        lines.append(f"{prefix}|--- {feature_name} <= {node.threshold:.6g}")
        self._append_rules(node.left, feature_names, lines, depth + 1)
        lines.append(f"{prefix}|--- {feature_name} > {node.threshold:.6g}")
        self._append_rules(node.right, feature_names, lines, depth + 1)


class DecisionTreeClassifier(_DecisionTree):
    """Binary decision-tree classifier implemented with NumPy."""

    def __init__(self, max_depth: int = 8) -> None:
        """Initialize the classifier."""
        super().__init__(classification=True, max_depth=max_depth)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted classes and probabilities for class one."""
        leaves = [self._leaf_for_row(row) for row in np.asarray(x, dtype=float)]
        predictions = np.array([leaf.value for leaf in leaves])
        probabilities = np.array([leaf.probability for leaf in leaves], dtype=float)
        return predictions, probabilities

    def __str__(self) -> str:
        """Generate a string representation of the classifier."""
        return f"DecisionTreeClassifier(max_depth={self.max_depth})"


class DecisionTreeRegressor(_DecisionTree):
    """Decision-tree regressor implemented with NumPy."""

    def __init__(self, max_depth: int = 8) -> None:
        """Initialize the regressor."""
        super().__init__(classification=False, max_depth=max_depth)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, None]:
        """Return numeric predictions without class probabilities."""
        predictions = np.array(
            [self._leaf_for_row(row).value for row in np.asarray(x, dtype=float)]
        )
        return predictions, None

    def __str__(self) -> str:
        """Generate a string representation of the regressor."""
        return f"DecisionTreeRegressor(max_depth={self.max_depth})"
