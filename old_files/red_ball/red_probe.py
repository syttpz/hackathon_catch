"""Diagnose red-ball detection on live frames; saves a frame and its mask.

Read-only: no arm, gripper or motion commands. Use it to check whether the
tracker's HSV thresholds actually cover the ball under the current lighting,
and to pick `min_radius_px` / `ball_radius_mm` from a real measurement.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from viam.components.camera import Camera
from viam.robot.client import RobotClient

from motion.viam_runtime import credentials, decode_color, positive
from old_files.red_ball.trajectory_local import red_candidates
from old_files.red_ball.track_red import locate_target, estimate_distance

# Progressively looser saturation/value floors; the follower defaults to 90/40.
SWEEP = [(180, 45), (140, 40), (90, 40), (60, 30)]


def candidate_colours(bgr, limit=6):
    """Median HSV and size of each reddish region, to pick a threshold from data.

    Skin is reddish but only moderately saturated; a red ball is much more so.
    Reading the actual numbers beats guessing a saturation floor -- and the
    largest reddish blob in a room is often a face, not the ball.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = (hsv[..., i].astype(int) for i in range(3))
    reddish = (((hue <= 12) | (hue >= 168)) & (saturation >= 60) & (value >= 30)).astype(np.uint8)
    reddish = cv2.morphologyEx(reddish, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(reddish)
    rows = []
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < 120:
            continue
        mask = labels == index
        left, width = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_WIDTH]
        top, height = stats[index, cv2.CC_STAT_TOP], stats[index, cv2.CC_STAT_HEIGHT]
        rows.append(dict(area=int(area), centre=centres[index],
                         radius=float(np.sqrt(area/np.pi)),
                         saturation=float(np.median(saturation[mask])),
                         value=float(np.median(value[mask])),
                         edge=bool(left <= 1 or top <= 1 or left+width >= bgr.shape[1]-1
                                   or top+height >= bgr.shape[0]-1)))
    rows.sort(key=lambda r: -r['area'])
    return rows[:limit]


def sweep_masks(bgr):
    """Blob count and largest-blob radius at each threshold, to expose a too-strict mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rows = []
    for saturation, value in SWEEP:
        mask = cv2.inRange(hsv, (0, saturation, value), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, saturation, value), (179, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
        areas = [a for a in areas if a >= 40]
        best = float(np.sqrt(areas[0]/np.pi)) if areas else 0.
        rows.append((saturation, value, int(mask.any(axis=None) and np.count_nonzero(mask)), len(areas), best))
    return rows


async def run(args):
    config = json.loads(args.config.read_text())
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        cam = Camera.from_robot(robot, args.camera or config['camera'])
        properties = await cam.get_properties(timeout=5)
        p = properties.intrinsic_parameters
        k = SimpleNamespace(fx=p.focal_x_px, fy=p.focal_y_px, cx=p.center_x_px, cy=p.center_y_px,
                            width=p.width_px, height=p.height_px)
        print(f'Intrinsics fx={k.fx:.1f} fy={k.fy:.1f} cx={k.cx:.1f} cy={k.cy:.1f} '
              f'{k.width}x{k.height}', flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        corners = [(.25, .25), (.75, .25), (.25, .75), (.75, .75)]
        for index in range(args.frames):
            images, _ = await cam.get_images(timeout=5)
            bgr = decode_color(next(im for im in images if im.name == args.color_source))
            print(f'--- frame {index+1}/{args.frames} ---', flush=True)
            for saturation, value, pixels, blobs, radius in sweep_masks(bgr):
                print(f'  S>={saturation:<4} V>={value:<4} red px {pixels:<7} blobs {blobs:<3} '
                      f'largest r={radius:.1f}px', flush=True)
            rows = candidate_colours(bgr)
            if rows:
                print(f'  {"centre":>16} {"r_px":>6} {"medS":>5} {"medV":>5}  note', flush=True)
                for row in rows:
                    note = 'TOUCHES IMAGE EDGE' if row['edge'] else ''
                    print(f'  ({row["centre"][0]:6.0f},{row["centre"][1]:6.0f}) '
                          f'{row["radius"]:6.1f} {row["saturation"]:5.0f} {row["value"]:5.0f}  {note}',
                          flush=True)
                best = max(rows, key=lambda r: r['saturation'])
                others = [r['saturation'] for r in rows if r is not best]
                if others and best['saturation'] > max(others):
                    print(f'  THRESHOLD: the most saturated blob is S={best["saturation"]:.0f}; '
                          f'next is S={max(others):.0f}. Use --min-saturation about '
                          f'{(best["saturation"]+max(others))/2:.0f} to keep only the ball.',
                          flush=True)
                    if best['radius'] < max(r['radius'] for r in rows):
                        print('  WARNING: the most saturated blob is NOT the largest. Picking the '
                              'largest red blob would select something else (often skin).',
                              flush=True)
            for shape in ('blob', 'ball', 'can'):
                center, size_px, physical_mm, reason = locate_target(bgr, {**config, 'detector': shape})
                if center is None:
                    print(f'  as {shape}: {reason}', flush=True)
                else:
                    if physical_mm is None:
                        print(f'  as {shape}: uv=({center[0]:.0f}, {center[1]:.0f}) {size_px:.1f}px '
                              f'-> on image border, range withheld', flush=True)
                    else:
                        distance = estimate_distance(size_px, k, physical_mm)
                        print(f'  as {shape}: uv=({center[0]:.0f}, {center[1]:.0f}) {size_px:.1f}px '
                              f'-> {distance:.0f} mm | image centre ({k.cx:.0f}, {k.cy:.0f})', flush=True)
            if center is not None:
                # Aiming aid: where the target sits in frame, and how much of the
                # image it fills. Both matter for calibration and for tracking.
                fill = 2*size_px/min(k.width, k.height)
                print(f'  aim: {(center[0]/k.width)*100:.0f}% across, '
                      f'{(center[1]/k.height)*100:.0f}% down, filling {fill*100:.1f}% of the '
                      f'short side', flush=True)
                if size_px < 12:
                    print('  AIM: target is small; move the camera closer or zoom the workspace in',
                          flush=True)
            if args.save:
                cv2.imwrite(str(args.out/f'frame{index+1}.jpg'), bgr)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (0, 180, 45), (10, 255, 255)) | cv2.inRange(hsv, (170, 180, 45), (179, 255, 255))
                cv2.imwrite(str(args.out/f'mask{index+1}.jpg'), mask)
            if index+1 < args.frames:
                await asyncio.sleep(args.interval)
        if args.save:
            print(f'Saved frames and masks under {args.out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('track_red.config.json'))
    parser.add_argument('--machine-config', type=Path)
    parser.add_argument('--camera', help='Override the configured camera, e.g. cam2')
    parser.add_argument('--color-source', default='color')
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--interval', type=positive, default=1.0)
    parser.add_argument('--out', type=Path, default=Path('/tmp/red_probe'))
    parser.add_argument('--save', action='store_true', help='Write frame and mask JPEGs for inspection')
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('\nProbe stopped.')
    except Exception as error:
        parser.exit(2, f'Probe failed: {error}\n')


if __name__ == '__main__':
    main()
