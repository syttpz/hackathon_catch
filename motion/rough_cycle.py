"""Continuous rough attempts: approach gate and per-throw rearming."""
import numpy as np


class RoughCycle:
    def __init__(self):
        self.ready = True
        self.blocked_until = 0.
        self.clear_since = None

    def fired(self, now):
        self.ready = False
        self.blocked_until = now + 2.
        self.clear_since = None

    def update(self, now, approaching):
        if self.ready:
            return approaching
        if approaching:
            self.clear_since = None
        elif self.clear_since is None:
            self.clear_since = now
        if (now >= self.blocked_until and self.clear_since is not None
                and now-self.clear_since >= .5):
            self.ready = True
        return False


def approaching_bowl(flight, now, mouth):
    if flight is None:
        return False
    point, velocity = flight.at(now)
    delta = np.asarray(mouth)-point
    distance = float(np.linalg.norm(delta))
    return bool(np.isfinite(point).all() and np.isfinite(velocity).all()
                and 100. < distance < 1200.
                and point[2] > mouth[2]+40.
                and np.dot(delta, velocity)/distance > 300.
                and flight.samples >= 4 and flight.residual_mm <= 40.)


class RoughHandoff:
    """Pair recent exposure-time evidence, never extend it on duplicate frames."""
    def __init__(self, max_age=.25):
        self.max_age = max_age
        self.clear()

    def clear(self):
        self.flight = None
        self.wrist_stamp = None
        self.approach_stamp = None

    def observe_wrist(self, stamp, flight):
        if stamp is not None and (self.wrist_stamp is None or stamp > self.wrist_stamp):
            self.wrist_stamp, self.flight = stamp, flight

    def observe_approach(self, stamp):
        if stamp is not None and (self.approach_stamp is None or stamp > self.approach_stamp):
            self.approach_stamp = stamp

    def approach_recent(self, now):
        return self.approach_stamp is not None and 0 <= now-self.approach_stamp <= self.max_age

    def candidate(self, now):
        if (self.approach_recent(now) and self.wrist_stamp is not None
                and 0 <= now-self.wrist_stamp <= self.max_age):
            return self.flight
        return None
