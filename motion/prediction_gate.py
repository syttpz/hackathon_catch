"""Prediction-consensus checks shared by the active catcher."""

import math
from collections import deque

import numpy as np


class PredictionGate:
    """Require a stable point and absolute arrival time over multiple frames."""

    def __init__(
        self,
        tolerance_mm=15.0,
        arrival_spread_s=0.12,
        min_samples=6,
        min_span_s=0.15,
    ):
        self.tolerance_mm = tolerance_mm
        self.arrival_spread_s = arrival_spread_s
        self.min_samples = min_samples
        self.min_span_s = min_span_s
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def add(self, now, target, arrival):
        target = np.asarray(target, float)
        if not np.isfinite(target).all() or not math.isfinite(arrival):
            self.reset()
            return False
        if self.samples and (
            now - self.samples[-1][0] > 0.1 or now <= self.samples[-1][0]
        ):
            self.reset()
        self.samples.append((now, target, arrival))
        while self.samples and now - self.samples[0][0] > 0.3:
            self.samples.popleft()
        if (
            len(self.samples) < self.min_samples
            or now - self.samples[0][0] < self.min_span_s
        ):
            return False

        targets = np.array([sample[1] for sample in self.samples])
        arrivals = np.array([sample[2] for sample in self.samples])
        centre = np.median(targets, axis=0)
        spread = np.linalg.norm(targets - centre, axis=1)
        agreeing = int(np.sum(spread <= self.tolerance_mm))
        return bool(
            agreeing >= self.min_samples
            and np.linalg.norm(target - centre) <= self.tolerance_mm
            and np.ptp(np.sort(arrivals)[: max(2, agreeing)])
            <= self.arrival_spread_s
        )

    def consensus(self):
        """Return the median of the recent targets."""
        return np.median(np.array([sample[1] for sample in self.samples]), axis=0)

    def progress(self):
        return len(self.samples), self.min_samples
