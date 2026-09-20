"""Track red-ball pixels from the existing Viam camera on the part machine.

Optional --aligned-depth adds camera-frame XYZ when depth is aligned to color.
This entry point never commands motion or calls the remote object segmenter.
"""
import argparse
import asyncio
import math
import time
from pathlib import Path

import cv2
import numpy as np
from viam.components.camera import Camera
from viam.robot.client import RobotClient

from old_files.red_ball.ball_tracking import detect_red_ball
from motion.viam_runtime import credentials, decode_color, positive


def red_candidates(bgr):
    """Saturated red circular blobs; skin must not merge with the ball mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 180, 45), (10, 255, 255))
    mask |= cv2.inRange(hsv, (170, 180, 45), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 80 or perimeter <= 0 or 4 * np.pi * area / perimeter**2 < .60:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 0 or y <= 0 or x+w >= bgr.shape[1] or y+h >= bgr.shape[0]:
            continue
        moments = cv2.moments(contour)
        center = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
        radius = float(np.sqrt(area/np.pi))
        candidates.append((center,radius))
    return candidates


def red_center(bgr):
    candidates = red_candidates(bgr)
    return candidates[0][0] if len(candidates) == 1 else None


async def run(args):
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    options.dial_options.timeout = 10
    # The inspected part serves TLS on loopback port 8080.
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        camera = Camera.from_robot(robot, args.camera)
        intrinsics = None
        if args.aligned_depth:
            properties = await camera.get_properties(timeout=5)
            p = properties.intrinsic_parameters
            intrinsics = dict(width=p.width_px, height=p.height_px,
                              fx=p.focal_x_px, fy=p.focal_y_px,
                              cx=p.center_x_px, cy=p.center_y_px)
        print('Local Viam camera tracker; Ctrl+C to stop. No robot motion.', flush=True)
        print('XYZ: camera optical frame (X right, Y down, Z forward).'
              if args.aligned_depth else
              'Pixel tracking only; --aligned-depth enables XYZ after alignment is verified.',
              flush=True)
        last_timestamp = None
        previous = None
        last_print = 0.0
        started = report_at = time.monotonic()
        frames = detections = duplicates = stale = 0
        next_poll = started
        period = 1 / args.hz
        while not args.duration or time.monotonic() - started < args.duration:
            images, metadata = await camera.get_images(timeout=5)
            now = time.monotonic()
            timestamp = metadata.captured_at.seconds + metadata.captured_at.nanos / 1e9
            status = None
            if not math.isfinite(timestamp) or not 0 <= time.time() - timestamp <= .25:
                stale += 1
                previous = None
                status = 'Missing/stale frame timestamp'
            elif last_timestamp is not None and timestamp <= last_timestamp:
                duplicates += 1
            else:
                last_timestamp = timestamp
                frames += 1
                sources = {im.name: im for im in images}
                if args.color_source not in sources:
                    raise ValueError(f'Color source missing; available: {list(sources)}')
                bgr = decode_color(sources[args.color_source])
                point = None
                if args.aligned_depth:
                    if args.depth_source not in sources:
                        raise ValueError(f'Depth source missing; available: {list(sources)}')
                    depth = np.asarray(sources[args.depth_source].bytes_to_depth_array(), dtype=float)
                    obs = detect_red_ball(bgr, depth, intrinsics, args.radius_mm, timestamp)
                    if obs is not None:
                        point = obs.xyz / 1000
                else:
                    point = red_center(bgr)
                if point is None:
                    previous = None
                    status = 'No single red ball' + (' with valid depth' if args.aligned_depth else '')
                else:
                    detections += 1
                    velocity = None
                    if previous is not None:
                        dt = timestamp - previous[0]
                        if 1e-6 < dt <= .25:
                            velocity = (point - previous[1]) / dt
                    previous = (timestamp, point)
                    units = 'm' if args.aligned_depth else 'px'
                    axes = 'camera XYZ' if args.aligned_depth else 'pixel UV'
                    status = f'{axes}={np.round(point, 3).tolist()} {units}'
                    if velocity is not None:
                        status += f' | velocity={np.round(velocity, 3).tolist()} {units}/s'
            if status is not None and now - last_print >= 1 / args.print_hz:
                print(status, flush=True)
                last_print = now
            if now - report_at >= 2:
                elapsed = now - report_at
                print(f'Fresh frames: {frames/elapsed:.1f} Hz | ball observations: '
                      f'{detections/elapsed:.1f} Hz | duplicates: {duplicates} | stale: {stale}',
                      flush=True)
                frames = detections = duplicates = stale = 0
                report_at = now
            # Poll slightly faster than the stream by default to avoid missing
            # every other frame when client and camera clocks are out of phase.
            next_poll += period
            finished = time.monotonic()
            if next_poll < finished:
                next_poll = finished
            await asyncio.sleep(max(0, next_poll - finished))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine-config', type=Path,
                        help='Read API credentials from this local Viam config; otherwise use env vars')
    parser.add_argument('--camera', default='cam')
    parser.add_argument('--color-source', default='color')
    parser.add_argument('--depth-source', default='depth')
    parser.add_argument('--hz', type=positive, default=120,
                        help='Poll rate; default 120 to consume a 60 FPS stream without phase locking')
    parser.add_argument('--print-hz', type=positive, default=5)
    parser.add_argument('--duration', type=positive, default=None,
                        help='Stop after this many seconds; default runs until Ctrl+C')
    parser.add_argument('--aligned-depth', action='store_true',
                        help='Use XYZ only if returned depth is aligned to color and intrinsics describe color')
    parser.add_argument('--radius-mm', type=positive, default=30,
                        help='Measured ball radius for XYZ estimation (default 30 mm)')
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('\nTracking stopped.')
    except (ValueError, KeyError, OSError, RuntimeError, asyncio.TimeoutError) as error:
        parser.exit(2, f'Tracker failed: {error}\n')


if __name__ == '__main__':
    main()
