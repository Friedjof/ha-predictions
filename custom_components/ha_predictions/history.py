"""Pure helpers for filtering training data and recorder history."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, tzinfo

TRAINING_MODE = "TRAINING"
PRODUCTION_MODE = "PRODUCTION"
DATASET_TIMESTAMP = "__timestamp"
INVALID_STATES = {"", "unknown", "unavailable"}


@dataclass(frozen=True)
class DatasetFilterSettings:
    """User-configurable filters for automatic training rows."""

    date_from: date | None = None
    date_to: date | None = None
    time_from: time = time.min
    time_to: time = time.max
    weekdays: frozenset[int] = frozenset(range(7))
    minimum_interval: float = 0
    require_complete: bool = True
    include_production: bool = False


@dataclass(frozen=True)
class HistoryPivotResult:
    """Result of converting recorder states into training rows."""

    rows: list[dict]
    filtered_counts: dict[str, int] = field(default_factory=dict)

    @property
    def production_rows_filtered(self) -> int:
        """Return the number of candidate rows excluded in production mode."""
        return self.filtered_counts.get("production", 0)


def filter_reason(  # noqa: PLR0911, PLR0913
    timestamp: float,
    values: dict[str, object],
    entities: list[str],
    *,
    target_entity: str,
    operation_mode: str,
    settings: DatasetFilterSettings,
    local_timezone: tzinfo,
    last_accepted_timestamp: float | None,
) -> str | None:
    """Return why a candidate row is excluded, or None when accepted."""
    target_value = values.get(target_entity)
    if _is_invalid(target_value):
        return "invalid_target"

    if operation_mode == PRODUCTION_MODE and not settings.include_production:
        return "production"

    local_datetime = datetime.fromtimestamp(timestamp, UTC).astimezone(local_timezone)
    if settings.date_from is not None and local_datetime.date() < settings.date_from:
        return "before_date"
    if settings.date_to is not None and local_datetime.date() > settings.date_to:
        return "after_date"
    if local_datetime.weekday() not in settings.weekdays:
        return "weekday"

    local_time = local_datetime.timetz().replace(tzinfo=None)
    if settings.time_from <= settings.time_to:
        in_time_range = settings.time_from <= local_time <= settings.time_to
    else:
        in_time_range = (
            local_time >= settings.time_from or local_time <= settings.time_to
        )
    if not in_time_range:
        return "time"

    valid_values = sum(not _is_invalid(values.get(entity)) for entity in entities)
    if settings.require_complete and valid_values != len(entities):
        return "incomplete"
    if not settings.require_complete and valid_values < 2:  # noqa: PLR2004
        return "incomplete"

    if (
        last_accepted_timestamp is not None
        and timestamp - last_accepted_timestamp < settings.minimum_interval
    ):
        return "minimum_interval"
    return None


def pivot_state_history(  # noqa: PLR0913
    rows: list[tuple],
    entities: list[str],
    target_entity: str,
    *,
    operation_mode_rows: list[tuple] | None = None,
    settings: DatasetFilterSettings | None = None,
    local_timezone: tzinfo = UTC,
) -> HistoryPivotResult:
    """Forward-fill recorder states and apply training dataset filters."""
    if not rows:
        return HistoryPivotResult([])

    settings = settings or DatasetFilterSettings()
    timestamp_data = defaultdict(dict)
    for timestamp, entity_id, state in rows:
        timestamp_data[timestamp][entity_id] = state

    mode_timeline = sorted(operation_mode_rows or [], key=lambda row: row[0])
    mode_index = 0
    current_mode = TRAINING_MODE
    last_state = dict.fromkeys(entities, "")
    last_accepted_timestamp = None
    pivot_rows = []
    filtered_counts: dict[str, int] = defaultdict(int)

    for timestamp in sorted(timestamp_data):
        while (
            mode_index < len(mode_timeline)
            and mode_timeline[mode_index][0] <= timestamp
        ):
            mode = mode_timeline[mode_index][2]
            if mode in (TRAINING_MODE, PRODUCTION_MODE):
                current_mode = mode
            mode_index += 1

        for entity in entities:
            if entity in timestamp_data[timestamp]:
                last_state[entity] = timestamp_data[timestamp][entity]

        reason = filter_reason(
            timestamp,
            last_state,
            entities,
            target_entity=target_entity,
            operation_mode=current_mode,
            settings=settings,
            local_timezone=local_timezone,
            last_accepted_timestamp=last_accepted_timestamp,
        )
        if reason is not None:
            filtered_counts[reason] += 1
            continue

        pivot_rows.append(
            {
                DATASET_TIMESTAMP: datetime.fromtimestamp(timestamp, UTC).isoformat(),
                **{entity: last_state[entity] for entity in entities},
            }
        )
        last_accepted_timestamp = timestamp

    return HistoryPivotResult(pivot_rows, dict(filtered_counts))


def _is_invalid(value: object) -> bool:
    """Return whether a state cannot be used as a complete dataset value."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.casefold() in INVALID_STATES
    return isinstance(value, float) and math.isnan(value)
