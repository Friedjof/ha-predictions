#!/usr/bin/env python3
"""Continuously simulate correlated household states through the HA REST API."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError

from ha_api import set_boolean, set_number

MINUTES_PER_DAY = 24 * 60
WEEKEND_START_DAY = 5
SLEEP_END_HOUR = 6
SLEEP_START_HOUR = 23
MORNING_HOME_END_HOUR = 8
EVENING_HOME_START_HOUR = 17.5
MORNING_ACTIVITY_END_HOUR = 9
EVENING_ACTIVITY_START_HOUR = 17
MAX_LUX = 1000
DARK_LUX_THRESHOLD = 110
PRESENCE_ANOMALY_PROBABILITY = 0.015
TARGET_PERSISTENCE_PROBABILITY = 0.82


@dataclass(frozen=True)
class Situation:
    """One coherent simulated household situation."""

    day: int
    hour: float
    lux: int
    presence: bool
    motion: bool
    target: bool
    scenario: str


class HouseholdSimulator:
    """Generate stateful, correlated household situations."""

    def __init__(
        self,
        *,
        seed: int,
        start_hour: float,
        step_minutes: int,
        noise: float,
    ) -> None:
        """Initialize the simulation."""
        self.rng = random.Random(seed)  # noqa: S311
        self.minute = int(start_hour * 60)
        self.step_minutes = step_minutes
        self.noise = noise
        self.weather = 0.75
        self.presence = True
        self.target = False
        self.previous_lux: int | None = None

    def next(self) -> Situation:  # noqa: PLR0912, PLR0915
        """Advance simulated time and return the next situation."""
        day = (self.minute // MINUTES_PER_DAY) % 7
        minute_of_day = self.minute % MINUTES_PER_DAY
        hour = minute_of_day / 60
        weekend = day >= WEEKEND_START_DAY
        sleeping = hour < SLEEP_END_HOUR or hour >= SLEEP_START_HOUR

        expected_home = (
            weekend
            or hour < MORNING_HOME_END_HOUR
            or hour >= EVENING_HOME_START_HOUR
        )
        if sleeping:
            expected_home = True
        transition_probability = 0.65 if expected_home else 0.35
        if (
            self.presence != expected_home
            and self.rng.random() < transition_probability
        ):
            self.presence = expected_home
        elif self.rng.random() < PRESENCE_ANOMALY_PROBABILITY:
            self.presence = not self.presence

        self.weather = min(
            1.0,
            max(0.25, self.weather + self.rng.uniform(-0.08, 0.08)),
        )
        daylight = max(0.0, math.sin(math.pi * (hour - 6) / 12))
        lux = round(daylight * 900 * self.weather + self.rng.uniform(0, 12))
        lux = max(0, min(MAX_LUX, lux))
        if lux == self.previous_lux:
            lux = lux + 1 if lux < MAX_LUX else lux - 1
        self.previous_lux = lux

        if not self.presence:
            motion_probability = 0.015
        elif sleeping:
            motion_probability = 0.03
        elif (
            SLEEP_END_HOUR <= hour < MORNING_ACTIVITY_END_HOUR
            or EVENING_ACTIVITY_START_HOUR <= hour < SLEEP_START_HOUR
        ):
            motion_probability = 0.62
        else:
            motion_probability = 0.28
        motion = self.rng.random() < motion_probability

        dark = lux < DARK_LUX_THRESHOLD
        desired_target = self.presence and motion and dark and not sleeping
        if self.target and self.presence and dark and not sleeping:
            desired_target = self.rng.random() < TARGET_PERSISTENCE_PROBABILITY
        if self.rng.random() < self.noise:
            desired_target = not desired_target
        self.target = desired_target

        if sleeping:
            scenario = "sleeping"
        elif not self.presence:
            scenario = "away"
        elif motion and dark:
            scenario = "active_dark"
        elif motion:
            scenario = "active_daylight"
        elif dark:
            scenario = "home_idle_dark"
        else:
            scenario = "home_idle_daylight"

        situation = Situation(
            day=day,
            hour=hour,
            lux=lux,
            presence=self.presence,
            motion=motion,
            target=self.target,
            scenario=scenario,
        )
        self.minute += self.step_minutes
        return situation


def publish(args: argparse.Namespace, situation: Situation) -> None:
    """Publish one situation to Home Assistant in feature-to-target order."""
    if args.dry_run:
        return
    base_url = args.url.rstrip("/")
    set_number(base_url, args.token, args.hour, int(situation.hour))
    set_number(base_url, args.token, args.lux, situation.lux)
    set_boolean(base_url, args.token, args.presence, value=situation.presence)
    set_boolean(base_url, args.token, args.motion, value=situation.motion)
    set_boolean(base_url, args.token, args.target, value=situation.target)


def run(args: argparse.Namespace) -> None:
    """Run until interrupted or until the requested iteration limit is reached."""
    simulator = HouseholdSimulator(
        seed=args.seed,
        start_hour=args.start_hour,
        step_minutes=args.step_minutes,
        noise=args.noise,
    )
    iteration = 0
    while args.iterations == 0 or iteration < args.iterations:
        situation = simulator.next()
        publish(args, situation)
        iteration += 1
        print(  # noqa: T201
            f"{iteration:05d} day={situation.day} hour={situation.hour:05.2f} "
            f"lux={situation.lux:4d} presence={int(situation.presence)} "
            f"motion={int(situation.motion)} target={int(situation.target)} "
            f"scenario={situation.scenario}",
            flush=True,
        )
        if args.interval > 0:
            time.sleep(args.interval)


def main() -> None:
    """Parse command-line arguments and run the live simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=os.environ.get("HA_TOKEN"))
    parser.add_argument(
        "--url", default=os.environ.get("HA_URL", "http://localhost:8123")
    )
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=0, help="0 runs forever")
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--start-hour", type=float, default=5.0)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", default="input_boolean.test_prediction_target")
    parser.add_argument("--motion", default="input_boolean.test_motion")
    parser.add_argument("--presence", default="input_boolean.test_presence")
    parser.add_argument("--hour", default="input_number.test_hour")
    parser.add_argument("--lux", default="input_number.test_ambient_lux")
    args = parser.parse_args()

    if not args.dry_run and not args.token:
        parser.error("--token or the HA_TOKEN environment variable is required")
    if args.step_minutes <= 0:
        parser.error("--step-minutes must be greater than zero")
    if not 0 <= args.noise <= 1:
        parser.error("--noise must be between zero and one")

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")  # noqa: T201
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Could not reach Home Assistant: {error}") from error


if __name__ == "__main__":
    main()
