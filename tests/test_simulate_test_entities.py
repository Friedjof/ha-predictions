"""Tests for the continuous Home Assistant test-entity simulation."""

import sys
from itertools import pairwise
from pathlib import Path

scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))

from simulate_test_entities import HouseholdSimulator  # noqa: E402


def create_simulator(seed: int = 42) -> HouseholdSimulator:
    """Create a simulator with standard test settings."""
    return HouseholdSimulator(
        seed=seed,
        start_hour=5.0,
        step_minutes=15,
        noise=0.06,
    )


def test_simulation_is_reproducible() -> None:
    """Produce the same situations for the same random seed."""
    first = create_simulator()
    second = create_simulator()

    assert [first.next() for _ in range(20)] == [second.next() for _ in range(20)]


def test_lux_changes_on_every_iteration() -> None:
    """Ensure each loop causes at least one feature-state change."""
    simulator = create_simulator()
    lux_values = [simulator.next().lux for _ in range(200)]

    assert all(
        previous != current
        for previous, current in pairwise(lux_values)
    )


def test_target_is_correlated_but_not_deterministic() -> None:
    """Generate a target that strongly follows the intended household signal."""
    simulator = create_simulator()
    situations = [simulator.next() for _ in range(700)]
    matching_signal = [
        situation
        for situation in situations
        if situation.presence
        and situation.motion
        and situation.lux < 110
        and 6 <= situation.hour < 23
    ]
    other = [situation for situation in situations if situation not in matching_signal]

    matching_rate = sum(item.target for item in matching_signal) / len(matching_signal)
    other_rate = sum(item.target for item in other) / len(other)

    assert matching_rate > 0.75
    assert other_rate < 0.25
    assert any(item.target for item in other)
