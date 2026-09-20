"""Read-only rolling-ball interception geometry, all positions in millimeters."""
from collections import deque
import numpy as np


def line_crossing(position, velocity, endpoints, max_seconds=2):
    """Return (seconds, fraction, flange XYZ) or None for no future segment hit."""
    p, v = np.asarray(position, float)[:2], np.asarray(velocity, float)[:2]
    a, b = np.asarray(endpoints, float)
    if not np.isfinite(np.concatenate((p, v, a, b))).all():
        raise ValueError('Nonfinite intercept input')
    d = b[:2] - a[:2]
    matrix = np.column_stack((v, -d))
    scale = np.linalg.norm(v) * np.linalg.norm(d)
    if scale < 1e-9 or abs(np.linalg.det(matrix)) < 1e-6 * scale:
        return None
    seconds, fraction = np.linalg.solve(matrix, a[:2] - p)
    if not 0 < seconds <= max_seconds or not 0 <= fraction <= 1:
        return None
    return float(seconds), float(fraction), a + fraction * (b-a)


def point_on_ball_plane(pixel, intrinsics, origin, rotation, center_z):
    ray = np.array([(pixel[0]-intrinsics.cx)/intrinsics.fx,
                    (pixel[1]-intrinsics.cy)/intrinsics.fy, 1.0])
    direction = rotation @ ray
    if abs(direction[2]) < 1e-6:
        raise ValueError('Viewing ray is parallel to the ball-center plane')
    distance = (center_z-origin[2])/direction[2]
    if distance <= 0:
        raise ValueError('Ball-center plane is behind camera')
    return origin + distance * direction


class RollingFit:
    def __init__(self):
        self.samples = deque(maxlen=12)

    def reset(self):
        self.samples.clear()

    def add(self, timestamp, point):
        if not np.isfinite(timestamp) or not np.isfinite(point).all():
            raise ValueError('Invalid observation')
        if self.samples:
            dt = timestamp - self.samples[-1][0]
            if dt <= 0:
                return None
            if dt > .1 or np.linalg.norm(point-self.samples[-1][1]) > 100:
                self.reset()
        self.samples.append((timestamp, np.asarray(point, float)))
        if len(self.samples) < 6:
            return None
        times = np.array([s[0]-timestamp for s in self.samples])
        if np.ptp(times) < .1:
            return None
        points = np.array([s[1] for s in self.samples])
        design = np.column_stack((np.ones(len(times)), times))
        position, velocity = np.linalg.lstsq(design, points, rcond=None)[0]
        residual = np.sqrt(np.mean(np.sum((design @ np.array([position, velocity])-points)**2, axis=1)))
        if residual > 5:
            return None
        return position, velocity, float(residual)
