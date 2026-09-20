"""Independent fixed-camera observations for catch_plane (no motion commands)."""
import asyncio
import json
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2

from motion.ballistic import BallisticFit
from motion.viam_runtime import decode_color


def solve_side_pose(world, pixels, intrinsics):
    world, pixels = np.asarray(world, float), np.asarray(pixels, float)
    if (world.ndim != 2 or world.shape[1] != 3 or len(world) < 6
            or pixels.shape != (len(world), 2)
            or not np.isfinite(world).all() or not np.isfinite(pixels).all()):
        raise ValueError('Side calibration needs at least six finite 3D/pixel pairs')
    if np.linalg.matrix_rank(world-world.mean(axis=0)) < 3:
        raise ValueError('Side calibration samples must span three dimensions')
    k = intrinsics
    matrix = np.array([[k.fx, 0., k.cx], [0., k.fy, k.cy], [0., 0., 1.]])
    ok, rvec, tvec, indices = cv2.solvePnPRansac(
        world, pixels, matrix, None, iterationsCount=1000,
        reprojectionError=3., flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or indices is None or len(indices) < 4:
        raise ValueError('Side PnP did not converge; re-record the calibration samples')
    chosen = indices.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(world[chosen], pixels[chosen], matrix, None, rvec, tvec)
    projected, _ = cv2.projectPoints(world, rvec, tvec, matrix, None)
    errors = np.linalg.norm(projected.reshape(-1, 2)-pixels, axis=1)
    rotation, _ = cv2.Rodrigues(rvec)
    if not np.isfinite(tvec).all() or not ((world@rotation.T+tvec.ravel())[:, 2] > 0).all():
        raise ValueError('Side PnP produced invalid camera geometry')
    mask = np.zeros(len(world), bool)
    mask[chosen] = True
    return rotation.T, (-rotation.T@tvec).ravel(), errors, mask


def load_calibration(path, camera, intrinsics):
    data = json.loads(Path(path).read_text())
    if data['camera'] != camera:
        raise ValueError('Side calibration belongs to a different camera')
    saved = data['intrinsics']
    for key in ('width', 'height', 'fx', 'fy', 'cx', 'cy'):
        if not np.isclose(saved[key], getattr(intrinsics, key), rtol=1e-4, atol=.01):
            raise ValueError(f'Side calibration intrinsics changed: {key}')
    samples = data['samples']
    rotation, origin, errors, inliers = solve_side_pose(
        [s['world'] for s in samples], [s['pixel'] for s in samples], intrinsics)
    good = errors[inliers]
    trusted = (int(inliers.sum()) >= 6 and inliers.mean() >= .75
               and np.isfinite(good).all() and np.sqrt(np.mean(good**2)) <= 2.
               and good.max() <= 5.)
    report = (f'{inliers.sum()}/{len(inliers)} inliers, '
              f'RMS {np.sqrt(np.mean(good**2)):.2f}px, worst inlier {good.max():.2f}px')
    validation = data.get('independent_validation')
    if validation is not None:
        error = float(validation['error_px'])
        trusted = trusted and np.isfinite(error) and error <= 5.
        report += f', independent validation {error:.2f}px'
    correction = data.get('preview_origin_offset_world_mm')
    if correction is not None:
        offset = np.asarray(correction, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Invalid preview camera origin offset')
        origin = origin + offset
        trusted = False
        report += f', preview-only world offset {offset.tolist()}mm'
    return origin, rotation, bool(trusted), report


def new_fit(config):
    return BallisticFit(**{k: config[k] for k in (
        'min_samples', 'max_gap_s', 'max_samples', 'max_residual_mm',
        'release_speed_mm_s', 'min_span_s')})


def moving_fit(fit, timestamp, point, minimum_speed):
    flight = fit.add(timestamp, point)
    if flight is None:
        return None
    # A gravity-constrained short fit can invent downward velocity for a static
    # blob. Require measured displacement, not just the fitted velocity.
    first_t, first_p = fit.samples[0]
    if np.linalg.norm(point-first_p)/max(timestamp-first_t, 1e-9) < minimum_speed:
        return None
    return flight


def pixel_motion(history, since, minimum_speed):
    samples = [(t, p) for t, p in history if t >= since]
    if len(samples) < 2:
        return False
    span = samples[-1][0]-samples[0][0]
    return bool(span > 0 and np.linalg.norm(samples[-1][1]-samples[0][1])/span >= minimum_speed)


class SideTracker:
    def __init__(self, camera, config, intrinsics, origin, rotation, detect, locate,
                 color_source='color'):
        self.camera, self.config, self.intrinsics = camera, config, intrinsics
        self.origin, self.rotation = origin, rotation
        self.detect, self.locate, self.color_source = detect, locate, color_source
        self.fit = new_fit(config)
        self.flight = None
        self.timestamp = None
        self.sequence = 0
        self.status = 'waiting for first frame'
        self.frames = self.seen = 0
        self.previous = None
        self.last_capture = None
        self.pixels = deque(maxlen=config['max_samples'])
        self.stereo = None

    def invalidate(self, status):
        self.fit.reset()
        self.flight = None
        self.previous = None
        self.pixels.clear()
        self.status = status
        self.sequence += 1

    def current(self, now):
        if self.timestamp is None or not 0 <= now-self.timestamp <= self.config['max_frame_age_s']:
            return None
        return self.flight

    async def run(self):
        while True:
            try:
                images, meta = await self.camera.get_images(timeout=.3)
                now = time.monotonic()
                stamp = meta.captured_at.seconds+meta.captured_at.nanos/1e9
                age = time.time()-stamp
                if not np.isfinite(stamp) or not 0 <= age <= self.config['max_frame_age_s']:
                    self.invalidate('WAIT: stale side frame')
                    await asyncio.sleep(.01)
                    continue
                if self.last_capture is not None and stamp <= self.last_capture:
                    await asyncio.sleep(.005)
                    continue
                self.last_capture = stamp
                self.timestamp = now-age
                self.frames += 1
                sources = {im.name: im for im in images}
                bgr = decode_color(sources[self.color_source])
                k = self.intrinsics
                if bgr.shape[:2] != (k.height, k.width):
                    raise ValueError('Side image resolution changed')
                reason, point = 'no ball', None
                candidates = self.detect(bgr, self.config, self.previous)
                if self.stereo is not None:
                    self.stereo.add_side(self.timestamp, candidates, self.origin,
                                         self.rotation, self.intrinsics)
                for centre, radius, enclosing in candidates:
                    point, _, _, why = self.locate(
                        centre, radius, enclosing, self.origin, self.rotation, k, self.config)
                    if point is not None:
                        if (self.previous is not None and np.linalg.norm(centre-self.previous)
                                > self.config.get('track_gate_px', 200.)):
                            self.fit.reset()
                            self.pixels.clear()
                        self.previous = centre
                        break
                    reason = why
                if point is None:
                    self.invalidate(reason)
                else:
                    self.seen += 1
                    self.sequence += 1
                    self.pixels.append((self.timestamp, self.previous.copy()))
                    self.flight = moving_fit(self.fit, self.timestamp, point,
                                             self.config['side_min_speed_mm_s'])
                    if self.flight is not None and not pixel_motion(
                            self.pixels, self.fit.samples[0][0],
                            self.config.get('side_min_speed_px_s', 80.)):
                        self.flight = None
                    self.status = f'BALL {point.round(0).tolist()} mm (size range)'
                    if self.flight is None:
                        self.status += ' | waiting for flight'
                    else:
                        self.status += (f' | fit n={self.flight.samples} '
                                        f'r={self.flight.residual_mm:.0f}mm')
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.invalidate(f'WAIT: {type(error).__name__}: {str(error)[:160]}')
                await asyncio.sleep(.1)
            await asyncio.sleep(.002)


async def prepare_side(robot, args, config, detect, locate):
    from viam.components.camera import Camera
    camera = Camera.from_robot(robot, args.side_camera)
    p = (await camera.get_properties(timeout=3)).intrinsic_parameters
    k = SimpleNamespace(fx=p.focal_x_px, fy=p.focal_y_px, cx=p.center_x_px,
                        cy=p.center_y_px, width=p.width_px, height=p.height_px)
    calibration = args.side_calibration or args.config.parent/config.get(
        'side_calibration', 'cam2_pnp.json')
    origin, rotation, trusted, report = load_calibration(calibration, args.side_camera, k)
    print(f'SIDE {args.side_camera}: calibration {report}; '
          f'{"quality check passed" if trusted else "PREVIEW ONLY: recalibration required"}', flush=True)
    if args.execute and args.trajectory_source in ('side', 'stereo') and not trusted:
        raise ValueError('Side calibration quality failed; re-record cam2 PnP samples '
                         'before executing side catches. --trajectory-source wrist '
                         '--no-side keeps the original wrist-only mode.')
    side_config = dict(config, range_source='size', hue_min=config.get('side_hue_min', 18),
                       val_min=config.get('side_val_min', config['val_min']),
                       side_min_speed_mm_s=config.get('side_min_speed_mm_s', 500.))
    return SideTracker(camera, side_config, k, origin, rotation, detect, locate,
                       args.color_source)
