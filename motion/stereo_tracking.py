"""Capture-time two-camera triangulation; no camera or motion RPCs."""
from collections import deque

import numpy as np

from motion.ballistic import BallisticFit


def bearing(pixel, rotation, intrinsics):
    k = intrinsics
    direction = np.asarray(rotation)@np.array([
        (pixel[0]-k.cx)/k.fx, (pixel[1]-k.cy)/k.fy, 1.])
    return direction/np.linalg.norm(direction)


def triangulate(origin_a, ray_a, origin_b, ray_b, max_separation_mm=30.):
    a, b = np.asarray(origin_a, float), np.asarray(origin_b, float)
    u, v = np.asarray(ray_a, float), np.asarray(ray_b, float)
    if not np.isfinite(np.concatenate((a, b, u, v))).all():
        raise ValueError('non-finite stereo ray')
    u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
    if np.linalg.norm(np.cross(u, v)) < np.sin(np.radians(10)):
        raise ValueError('stereo rays nearly parallel')
    distances, *_ = np.linalg.lstsq(np.column_stack((u, -v)), b-a, rcond=None)
    if min(distances) <= 0:
        raise ValueError('stereo intersection behind a camera')
    pa, pb = a+distances[0]*u, b+distances[1]*v
    error = float(np.linalg.norm(pa-pb))
    if error > max_separation_mm:
        raise ValueError(f'stereo rays disagree by {error:.0f}mm')
    return (pa+pb)*.5, error


class StereoTracker:
    def __init__(self, config):
        self.config = config
        self.wrist = deque(maxlen=16)
        self.side = None
        self.pending = deque(maxlen=16)
        self.consumed = None
        self.timestamp = None
        self.sequence = 0
        self.flight = None
        self.status = 'waiting for synchronized camera pair'
        self.fit = BallisticFit(**{k: config[k] for k in (
            'min_samples', 'max_gap_s', 'max_samples', 'max_residual_mm',
            'release_speed_mm_s', 'min_span_s')})

    def add_wrist(self, timestamp, candidates, origin, rotation, intrinsics):
        ray = bearing(candidates[0][0], rotation, intrinsics) if len(candidates) == 1 else None
        self.wrist.append((timestamp, np.asarray(origin).copy(), ray))

    def add_side(self, timestamp, candidates, origin, rotation, intrinsics):
        ray = bearing(candidates[0][0], rotation, intrinsics) if len(candidates) == 1 else None
        self.pending.append((timestamp, np.asarray(origin).copy(), ray))

    def wrist_at(self, timestamp):
        # Interpolate exposure-time rays only across adjacent, unambiguous
        # observations. Do not extrapolate or bridge a missed detection.
        for first, second in zip(self.wrist, list(self.wrist)[1:]):
            t0, o0, r0 = first
            t1, o1, r1 = second
            if t0 <= timestamp <= t1 and 0 < t1-t0 <= .04:
                if r0 is None or r1 is None:
                    raise ValueError('wrist ball missing/ambiguous')
                alpha = (timestamp-t0)/(t1-t0)
                ray = r0*(1-alpha)+r1*alpha
                return o0*(1-alpha)+o1*alpha, ray/np.linalg.norm(ray)
        # Near-simultaneous hardware captures need no interpolation.
        if self.wrist:
            t, origin, ray = min(self.wrist, key=lambda s: abs(s[0]-timestamp))
            if abs(t-timestamp) <= .003 and ray is not None:
                return origin, ray
        raise ValueError('waiting for wrist exposure bracket')

    def reset(self, reason):
        self.fit.reset()
        self.flight = None
        self.status = reason

    def update(self, now):
        if self.pending:
            self.side = self.pending[0]
        if self.side is None:
            return
        timestamp, origin, ray = self.side
        if not 0 <= now-timestamp <= self.config['max_frame_age_s']:
            if self.pending:
                self.pending.popleft()
            self.reset('stale side observation')
            return
        if timestamp == self.consumed:
            return
        if ray is None:
            self.pending.popleft()
            self.consumed = timestamp
            self.sequence += 1
            self.reset('side ball missing/ambiguous')
            return
        try:
            wrist_origin, wrist_ray = self.wrist_at(timestamp)
        except ValueError as error:
            self.status = str(error)
            # No current pair may keep driving a previously fitted trajectory.
            self.flight = None
            if self.wrist and self.wrist[-1][0] > timestamp+.003:
                self.pending.popleft()
                self.consumed = timestamp
                self.fit.reset()
            return
        self.pending.popleft()
        self.consumed = timestamp
        self.sequence += 1
        try:
            point, error = triangulate(origin, ray, wrist_origin, wrist_ray)
            lower, upper = np.asarray(self.config['throw_volume_mm'], float)
            if not ((point >= lower).all() and (point <= upper).all()):
                raise ValueError('stereo point outside throw volume')
        except ValueError as error:
            self.reset(str(error))
            return
        self.timestamp = timestamp
        self.flight = self.fit.add(timestamp, point)
        if self.flight is not None:
            t0, p0 = self.fit.samples[0]
            if np.linalg.norm(point-p0)/(timestamp-t0) < 500.:
                self.flight = None
        self.status = f'BALL {point.round(0).tolist()} mm | ray gap {error:.1f}mm'
        self.status += (' | waiting for flight' if self.flight is None else
                        f' | fit n={self.flight.samples} r={self.flight.residual_mm:.0f}mm')

    def current(self, now):
        if self.timestamp is None or not 0 <= now-self.timestamp <= self.config['max_frame_age_s']:
            return None
        return self.flight
