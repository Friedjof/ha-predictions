"""Tests for rebuilding datasets from Home Assistant history."""

import sys
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ha_predictions_path = (
    Path(__file__).parent.parent / "custom_components" / "ha_predictions"
)
sys.path.insert(0, str(ha_predictions_path))

from history import (  # noqa: E402
    DATASET_TIMESTAMP,
    DatasetFilterSettings,
    pivot_state_history,
)


def test_production_rows_are_filtered() -> None:
    """Do not emit training rows while production mode is active."""
    feature = "input_boolean.motion"
    target = "input_boolean.light"
    rows = [
        (1.0, feature, "off"),
        (1.0, target, "off"),
        (3.0, feature, "on"),
        (5.0, target, "on"),
    ]
    modes = [
        (2.0, "select.operation_mode", "PRODUCTION"),
        (4.0, "select.operation_mode", "TRAINING"),
    ]

    result = pivot_state_history(
        rows, [feature, target], target, operation_mode_rows=modes
    )

    assert result.rows == [
        {
            DATASET_TIMESTAMP: "1970-01-01T00:00:01+00:00",
            feature: "off",
            target: "off",
        },
        {
            DATASET_TIMESTAMP: "1970-01-01T00:00:05+00:00",
            feature: "on",
            target: "on",
        },
    ]
    assert result.production_rows_filtered == 1


def test_state_changes_in_production_are_forward_filled() -> None:
    """Keep production states for the first row after training resumes."""
    feature = "input_boolean.motion"
    target = "input_boolean.light"
    rows = [
        (1.0, feature, "off"),
        (1.0, target, "off"),
        (3.0, feature, "on"),
        (5.0, target, "on"),
    ]
    modes = [
        (2.0, "select.operation_mode", "PRODUCTION"),
        (4.0, "select.operation_mode", "TRAINING"),
    ]

    result = pivot_state_history(
        rows, [feature, target], target, operation_mode_rows=modes
    )

    assert result.rows[-1][feature] == "on"


def test_history_before_first_mode_is_training_data() -> None:
    """Treat history before the first operation-mode state as training."""
    feature = "input_boolean.motion"
    target = "input_boolean.light"
    rows = [(1.0, feature, "on"), (1.0, target, "on")]
    modes = [(2.0, "select.operation_mode", "PRODUCTION")]

    result = pivot_state_history(
        rows, [feature, target], target, operation_mode_rows=modes
    )

    assert len(result.rows) == 1
    assert result.production_rows_filtered == 0


def test_date_weekday_and_overnight_time_filters() -> None:
    """Apply calendar filters in the configured local timezone."""
    feature = "input_boolean.motion"
    target = "input_boolean.light"
    timezone = ZoneInfo("Europe/Berlin")
    friday_night = datetime(2026, 7, 24, 23, 30, tzinfo=timezone).timestamp()
    saturday_day = datetime(2026, 7, 25, 12, 0, tzinfo=timezone).timestamp()
    rows = [
        (friday_night, feature, "on"),
        (friday_night, target, "on"),
        (saturday_day, target, "off"),
    ]
    settings = DatasetFilterSettings(
        date_from=date(2026, 7, 24),
        date_to=date(2026, 7, 24),
        time_from=time(22),
        time_to=time(6),
        weekdays=frozenset({4}),
    )

    result = pivot_state_history(
        rows, [feature, target], target, settings=settings, local_timezone=timezone
    )

    assert len(result.rows) == 1
    assert (
        result.rows[0][DATASET_TIMESTAMP]
        == datetime.fromtimestamp(friday_night, UTC).isoformat()
    )
    assert result.filtered_counts["after_date"] == 1


def test_minimum_interval_and_complete_row_filter() -> None:
    """Reject incomplete rows and rows arriving inside the minimum interval."""
    feature = "input_boolean.motion"
    second_feature = "input_number.lux"
    target = "input_boolean.light"
    rows = [
        (1.0, feature, "on"),
        (1.0, target, "on"),
        (2.0, second_feature, "10"),
        (5.0, target, "off"),
        (13.0, target, "on"),
    ]
    settings = DatasetFilterSettings(minimum_interval=10)

    result = pivot_state_history(
        rows, [feature, second_feature, target], target, settings=settings
    )

    assert [row[target] for row in result.rows] == ["on", "on"]
    assert result.filtered_counts == {"incomplete": 1, "minimum_interval": 1}


def test_production_rows_can_be_included() -> None:
    """Allow production history only when explicitly configured."""
    feature = "input_boolean.motion"
    target = "input_boolean.light"
    rows = [(2.0, feature, "on"), (2.0, target, "on")]
    modes = [(1.0, "select.operation_mode", "PRODUCTION")]

    result = pivot_state_history(
        rows,
        [feature, target],
        target,
        operation_mode_rows=modes,
        settings=DatasetFilterSettings(include_production=True),
    )

    assert len(result.rows) == 1
    assert result.production_rows_filtered == 0
