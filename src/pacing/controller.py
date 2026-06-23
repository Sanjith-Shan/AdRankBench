"""Budget pacing controllers for an ad serving simulation.

This module maps to the ads pacing and traffic control area. The goal of a
pacer is to spend a daily budget smoothly across the day so the campaign avoids
exhausting its budget early in the morning. Smooth pacing keeps cost per mille
stable and gives the campaign broad time of day coverage.

Three pacers are provided. PIDPacer uses a proportional integral derivative
feedback loop that tracks an ideal spend trajectory. AsapPacer spends as fast
as possible and is the naive baseline that exhausts budget early. ThrottlePacer
is a simple proportional baseline that targets a flat per slot spend rate.

Every pacer exposes the same interface. The reset method primes the pacer with
the total budget, the number of slots, and optionally the traffic curve so it
can plan an ideal trajectory. The step method returns a throttle probability in
the closed interval from 0 to 1. The throttle is the fraction of eligible
impressions the pacer is willing to serve in that slot.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _ideal_cumulative(traffic: Optional[np.ndarray], n_slots: int) -> np.ndarray:
    """Build the ideal cumulative spend fraction trajectory.

    When a traffic curve is given the ideal trajectory follows the cumulative
    share of traffic so spend is proportional to available demand. When no
    traffic curve is given the ideal trajectory is a straight line from 0 to 1
    which corresponds to a flat per slot spend rate.

    Returns an array of length n_slots whose last value is 1.0.
    """
    if traffic is not None and np.sum(traffic) > 0:
        cum = np.cumsum(np.asarray(traffic, dtype=np.float64))
        return cum / cum[-1]
    # Flat trajectory. Each slot should have spent an equal fraction by its end.
    return (np.arange(1, n_slots + 1, dtype=np.float64)) / float(n_slots)


class BasePacer:
    """Common interface shared by every pacer.

    A pacer is reset once per simulation and then queried slot by slot. The
    reset method records the budget plan. The step method decides a throttle
    probability for the current slot given how much has been spent so far.
    """

    def reset(
        self,
        total_budget: float,
        n_slots: int,
        traffic: Optional[np.ndarray] = None,
    ) -> "BasePacer":
        """Prime the pacer for a new simulation. Returns self."""
        self.total_budget = float(total_budget)
        self.n_slots = int(n_slots)
        self.traffic = None if traffic is None else np.asarray(traffic, dtype=np.float64)
        self.ideal_cum_frac = _ideal_cumulative(self.traffic, self.n_slots)
        return self

    def step(self, slot: int, eligible: float, spent_so_far: float) -> float:
        """Return a throttle probability in [0, 1] for this slot."""
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


class PIDPacer(BasePacer):
    """A proportional integral derivative feedback pacer.

    The pacer tracks an ideal cumulative spend target for the current slot. The
    error is the gap between the target cumulative spend and the actual spend so
    far normalized by the total budget. The PID control law turns that error
    into an adjustment of the throttle probability. A positive error means the
    campaign is behind plan so the throttle rises. A negative error means the
    campaign is ahead of plan so the throttle falls.

    The integral term removes steady state bias and the derivative term damps
    overshoot when traffic spikes. The output is clamped to the closed interval
    from 0 to 1.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        total_budget: float,
        n_slots: int,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        # Prime with the constructor arguments. A traffic aware plan can be set
        # later by calling reset with a traffic curve.
        self.reset(total_budget, n_slots, traffic=None)

    def reset(
        self,
        total_budget: float,
        n_slots: int,
        traffic: Optional[np.ndarray] = None,
    ) -> "PIDPacer":
        super().reset(total_budget, n_slots, traffic)
        # Controller state.
        self.integral = 0.0
        self.prev_error = 0.0
        # Throttle starts at the ideal share of the first slot so the loop has a
        # sensible warm start rather than slamming to 0 or 1.
        self.throttle = float(self.ideal_cum_frac[0]) if self.n_slots > 0 else 0.0
        self.throttle = min(1.0, max(0.0, self.throttle))
        return self

    def step(self, slot: int, eligible: float, spent_so_far: float) -> float:
        if self.total_budget <= 0:
            return 0.0
        # Target cumulative spend by the end of this slot as a budget fraction.
        target_frac = float(self.ideal_cum_frac[min(slot, self.n_slots - 1)])
        actual_frac = spent_so_far / self.total_budget
        error = target_frac - actual_frac

        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.throttle = min(1.0, max(0.0, self.throttle + adjustment))
        return self.throttle

    def get_name(self) -> str:
        return "PID"


class AsapPacer(BasePacer):
    """As soon as possible pacer.

    This baseline always serves every eligible impression until the budget runs
    out. It exhausts budget early in the day which is exactly the failure mode
    smooth pacing is designed to avoid. It is included for comparison.
    """

    def __init__(
        self,
        total_budget: Optional[float] = None,
        n_slots: Optional[int] = None,
    ):
        # Budget and slot count are optional here because run_pacing calls reset
        # with the real values before stepping.
        if total_budget is not None and n_slots is not None:
            self.reset(total_budget, n_slots, traffic=None)

    def step(self, slot: int, eligible: float, spent_so_far: float) -> float:
        # Always willing to spend on every eligible impression. The simulator
        # caps spend at the remaining budget so this naturally stops when the
        # budget is gone.
        return 1.0

    def get_name(self) -> str:
        return "ASAP"


class ThrottlePacer(BasePacer):
    """A simple proportional pacing baseline.

    This pacer aims for a flat per slot spend equal to the total budget divided
    by the number of slots. At each slot it sets the throttle to the ratio of
    the desired per slot spend to the spend that would happen at full throttle.
    It does not use feedback so it drifts when traffic is uneven, which is what
    the PID pacer corrects.
    """

    def __init__(
        self,
        total_budget: Optional[float] = None,
        n_slots: Optional[int] = None,
    ):
        if total_budget is not None and n_slots is not None:
            self.reset(total_budget, n_slots, traffic=None)

    def reset(
        self,
        total_budget: float,
        n_slots: int,
        traffic: Optional[np.ndarray] = None,
    ) -> "ThrottlePacer":
        super().reset(total_budget, n_slots, traffic)
        self.per_slot_target = (
            self.total_budget / self.n_slots if self.n_slots > 0 else 0.0
        )
        return self

    def step(self, slot: int, eligible: float, spent_so_far: float) -> float:
        if eligible <= 0:
            return 0.0
        # Throttle so the expected spend this slot matches the flat target.
        # Spend at full throttle equals the number of eligible impressions.
        throttle = self.per_slot_target / eligible
        return min(1.0, max(0.0, throttle))

    def get_name(self) -> str:
        return "Proportional"
