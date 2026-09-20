"""Red sphere RGB-D localization and short-horizon trajectory estimation (mm, s)."""
from collections import deque
from dataclasses import dataclass
import asyncio
import time

import cv2
import numpy as np


@dataclass
class Observation:
    timestamp: float
    xyz: np.ndarray
    pixel: tuple[float, float]
    bbox: tuple[int, int, int, int] | None = None
    depth_mm: float | None = None


def detect_red_ball(bgr, depth_mm, intrinsics, radius_mm, timestamp,
                    min_area=80, min_depth_mm=150, max_depth_mm=3000):
    """Return approximate sphere center in camera optical coordinates, or None.

    Images must be aligned and rectified; intrinsics must describe the color image.
    Uses central surface depth plus known sphere radius along the viewing ray.
    Requires one unambiguous red circular candidate with valid central depth.
    """
    if bgr is None or depth_mm.shape != bgr.shape[:2]:
        raise ValueError("Color and depth must have matching aligned dimensions")
    fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
    if not np.isfinite([fx, fy, cx, cy, radius_mm, timestamp]).all() or min(fx, fy, radius_mm) <= 0:
        raise ValueError("Invalid intrinsics, radius or timestamp")
    if (intrinsics["width"], intrinsics["height"]) != (bgr.shape[1], bgr.shape[0]):
        raise ValueError("Intrinsics resolution does not match color image")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 70), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < min_area or perimeter == 0 or 4 * np.pi * area / perimeter**2 < 0.65:
            continue
        _, radius_px = cv2.minEnclosingCircle(contour)
        left, top, width, height = cv2.boundingRect(contour)
        # Inclusive pixel bounds: the midpoint is also the deprojection target.
        right, bottom = left + width - 1, top + height - 1
        u, v = (left + right) / 2, (top + bottom) / 2
        if u-radius_px < 0 or v-radius_px < 0 or u+radius_px >= bgr.shape[1] or v+radius_px >= bgr.shape[0]:
            continue
        yy, xx = np.ogrid[:bgr.shape[0], :bgr.shape[1]]
        central = ((xx-u)**2 + (yy-v)**2 <= max(2, radius_px * 0.2)**2) & (mask > 0)
        values = np.asarray(depth_mm[central], dtype=float)
        valid = values[np.isfinite(values) & (values >= min_depth_mm) & (values <= max_depth_mm)]
        if valid.size < 5 or valid.size < values.size * 0.6:
            continue
        z = float(np.median(valid))
        if np.median(np.abs(valid-z)) > max(5, radius_mm * 0.25):
            continue
        # Reject blobs whose apparent size is inconsistent with the known ball.
        expected_radius = (fx + fy) / 2 * radius_mm / (z + radius_mm)
        if not 0.5 * expected_radius <= radius_px <= 1.8 * expected_radius:
            continue
        ray = np.array([(u-cx)/fx, (v-cy)/fy, 1.0])
        xyz = z * ray + radius_mm * ray / np.linalg.norm(ray)
        candidates.append(Observation(timestamp, xyz, (u, v), (left, top, right, bottom), z))
    return candidates[0] if len(candidates) == 1 else None


class Trajectory:
    """Least-squares velocity with optional known acceleration in WORLD axes.

    Use acceleration=(0,0,0) for short rolling segments, (0,0,-9810) for
    free flight in a Z-up world. Does not model bounces or rolling friction.
    """
    def __init__(self, acceleration=(0, 0, 0), max_age=0.25, max_horizon=0.5):
        self.samples = deque(maxlen=12)
        self.acceleration = np.asarray(acceleration, dtype=float)
        if self.acceleration.shape != (3,) or not np.isfinite(self.acceleration).all():
            raise ValueError("Acceleration must be a finite XYZ vector")
        self.max_age = max_age
        self.max_horizon = max_horizon

    def add(self, observation):
        if not np.isfinite(observation.xyz).all() or not np.isfinite(observation.timestamp):
            raise ValueError("Nonfinite observation")
        if self.samples:
            dt = observation.timestamp - self.samples[-1].timestamp
            if dt <= 0:
                raise ValueError("Duplicate or out-of-order capture timestamp")
            if dt > self.max_age:
                self.samples.clear()
        self.samples.append(observation)

    def predict(self, at, now=None):
        now = time.time() if now is None else now
        if len(self.samples) < 4:
            raise ValueError("Need at least four fresh ball observations")
        latest = self.samples[-1].timestamp
        horizon = at - latest
        if not 0 <= now-latest <= self.max_age or not 0 <= horizon <= self.max_horizon:
            raise ValueError("Stale track or prediction beyond allowed horizon")
        t = np.array([s.timestamp-latest for s in self.samples])
        if np.ptp(t) < 0.06:
            raise ValueError("Need at least 60 ms of observations")
        positions = np.array([s.xyz for s in self.samples])
        design = np.column_stack((np.ones(len(t)), t))
        corrected = positions - 0.5 * t[:, None]**2 * self.acceleration
        coefficients = np.linalg.lstsq(design, corrected, rcond=None)[0]
        residual = float(np.sqrt(np.mean(np.sum((design @ coefficients-corrected)**2, axis=1))))
        if residual > 10:
            raise ValueError("Unstable trajectory: fit residual exceeds 10 mm")
        position, velocity = coefficients
        return (position + velocity*horizon + 0.5*self.acceleration*horizon**2,
                velocity + self.acceleration*horizon, residual)


class BallCamera:
    """Viam RGB-D adapter using the calibrated wrist-camera frame tree.

    Caller must exclusively control the arm. Capture only after settling: Viam
    transforms use current joints, not a historical pose at exposure time.
    """
    def __init__(self, camera, config, machine, arm):
        self.camera = camera
        self.config = config
        self.machine = machine
        self.arm = arm
        self.last_timestamp = None

    async def observe(self):
        from viam.proto.common import Pose, PoseInFrame

        if await self.arm.is_moving(timeout=2):
            raise ValueError("Stop the arm before capturing wrist-camera observations")
        before = np.asarray((await self.arm.get_joint_positions(timeout=2)).values)
        images, metadata = await self.camera.get_images(timeout=2)
        timestamp = metadata.captured_at.seconds + metadata.captured_at.nanos / 1e9
        if not 0 <= time.time()-timestamp <= self.config.get("max_age_s", 0.25):
            raise ValueError("Missing/stale camera timestamp; synchronize client and robot clocks")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Camera returned a duplicate/out-of-order frame")
        self.last_timestamp = timestamp
        sources = {im.name: im for im in images}
        color = sources[self.config["color_source"]]
        depth = sources[self.config["depth_source"]]
        bgr = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Color source must return encoded JPEG or PNG")
        # Viam raw depth decoding preserves integer millimeter depths.
        depth_mm = np.asarray(depth.bytes_to_depth_array(), dtype=float)
        observation = detect_red_ball(bgr, depth_mm, self.config["intrinsics"],
                                      self.config["ball_radius_mm"], timestamp)
        if observation is not None:
            x, y, z = observation.xyz
            world = await asyncio.wait_for(self.machine.transform_pose(
                PoseInFrame(reference_frame=self.config["camera_optical_frame"],
                            pose=Pose(x=float(x), y=float(y), z=float(z), o_z=1)),
                self.config["world_frame"]), timeout=2)
            observation.xyz = np.array([world.pose.x, world.pose.y, world.pose.z])
        after = np.asarray((await self.arm.get_joint_positions(timeout=2)).values)
        if (await self.arm.is_moving(timeout=2) or before.shape != after.shape
                or not before.size or not np.isfinite(before).all() or not np.isfinite(after).all()
                or np.max(np.abs(after-before)) > 0.1):
            raise ValueError("Arm moved during camera capture; discard observation")
        return observation
