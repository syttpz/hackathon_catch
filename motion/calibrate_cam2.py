"""Measure a fixed camera's pose in the world from a ball held in the gripper.

Two steps, both driven by you moving the arm in manual mode -- this never
commands motion:

    ./calibrate-cam2.sh capture --out cam2_samples.json    # once per pose
    ./calibrate-cam2.sh solve   --samples cam2_samples.json

Each capture records the gripper pose from forward kinematics and the ball
centre in the camera frame, estimated two independent ways:

  depth  deproject the blob centroid, then step one ball radius further along
         the viewing ray, because depth reads the front surface not the centre
  size   range from the projected radius, f*R/r -- no depth involved

They are reported side by side every capture. Large disagreement means the
depth stream is not aligned to colour on this camera, and `solve --source size`
is then the trustworthy one. Neither is assumed correct.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from viam.components.camera import Camera
from viam.services.motion import MotionClient
from viam.robot.client import RobotClient

from motion.viam_runtime import credentials, decode_color, positive
from motion.live_camera_pose import pose_matrix
from motion.handeye import solve_eye_to_hand, orientation_spread_deg, viam_frame


def ray(pixel, intrinsics):
    """Unit viewing direction in the camera optical frame (X right, Y down, Z forward)."""
    direction = np.array([(pixel[0]-intrinsics.cx)/intrinsics.fx,
                          (pixel[1]-intrinsics.cy)/intrinsics.fy, 1.0])
    return direction/np.linalg.norm(direction)


def centre_from_size(pixel, radius_px, intrinsics, ball_radius_mm):
    """Ball centre from its projected radius alone. Independent of depth."""
    if radius_px <= 0:
        raise ValueError('Ball pixel radius must be positive')
    focal = (intrinsics.fx+intrinsics.fy)*.5
    return ray(pixel, intrinsics)*(focal*ball_radius_mm/radius_px)


def centre_from_depth(pixel, radius_px, depth, intrinsics, ball_radius_mm):
    """Ball centre from the depth map: front surface plus one radius along the ray.

    Samples a disc well inside the blob and takes the low quartile, so a few
    background pixels bleeding in cannot pull the range outwards.
    """
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    window = max(1, int(radius_px*.4))
    patch = depth[max(0, v-window):v+window+1, max(0, u-window):u+window+1]
    valid = patch[(patch > 0) & np.isfinite(patch)]
    if valid.size < 4:
        return None
    surface_z = float(np.percentile(valid, 25))
    direction = ray(pixel, intrinsics)
    # depth is measured along the optical axis, so scale the ray to reach it.
    surface_range = surface_z/direction[2]
    return direction*(surface_range+ball_radius_mm)


def ball_silhouette(bgr, config):
    """Red blobs sized by their minimum enclosing circle, not equivalent area.

    A lit sphere has a specular highlight and a darkening rim, so a saturation
    mask keeps only the core and `sqrt(area/pi)` understates the radius -- by a
    factor that changes with pose and lighting (measured here: 1.26 to 1.50
    across three frames). Range goes as f*R/r, so that error goes straight into
    the calibration, and because it is not constant a single fitted scale
    cannot absorb it. The silhouette of a sphere is a circle, so the smallest
    circle enclosing the blob tracks the true edge far more stably.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation, value = config['min_saturation'], config['min_value']
    mask = cv2.inRange(hsv, (0, saturation, value), (config['hue_low_max'], 255, 255))
    mask |= cv2.inRange(hsv, (config['hue_high_min'], saturation, value), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = bgr.shape[:2]
    found = []
    for contour in contours:
        if cv2.contourArea(contour) < config['min_area_px']:
            continue
        (u, v), radius = cv2.minEnclosingCircle(contour)
        x, y, w, h = cv2.boundingRect(contour)
        truncated = bool(x <= 1 or y <= 1 or x+w >= width-1 or y+h >= height-1)
        found.append((np.array([u, v], float), float(radius), truncated))
    found.sort(key=lambda c: -c[1])
    return found


def pick_blob(bgr, config, index):
    candidates = ball_silhouette(bgr, config)
    return candidates, (candidates[index] if index < len(candidates) else None)


async def observe(cam, motion, config, args, intrinsics):
    """One synchronised look: gripper pose from FK, ball centre from the camera."""
    images, _ = await cam.get_images(timeout=10)
    sources = {im.name: im for im in images}
    bgr = decode_color(sources[args.color_source])
    candidates, chosen = pick_blob(bgr, config, args.pick)
    pose = await asyncio.wait_for(motion.get_pose(args.gripper, args.world_frame, timeout=10), 10)
    if pose.reference_frame != args.world_frame:
        raise ValueError('Unexpected gripper reference frame')
    rotation, position = pose_matrix(pose.pose)[1], pose_matrix(pose.pose)[0]
    if chosen is None:
        return None, candidates, bgr, (rotation, position)
    pixel, radius_px, truncated = chosen
    if truncated:
        return None, candidates, bgr, (rotation, position)
    by_size = centre_from_size(pixel, radius_px, intrinsics, config['ball_radius_mm'])
    by_depth = None
    if args.depth_source in sources:
        depth = np.asarray(sources[args.depth_source].bytes_to_depth_array(), dtype=float)
        by_depth = centre_from_depth(pixel, radius_px, depth, intrinsics, config['ball_radius_mm'])
    sample = {'pixel': [float(pixel[0]), float(pixel[1])], 'radius_px': float(radius_px),
              'camera_size': by_size.tolist(),
              'camera_depth': None if by_depth is None else by_depth.tolist(),
              'gripper_rotation': rotation.tolist(), 'gripper_position': position.tolist()}
    return sample, candidates, bgr, (rotation, position)


async def capture(args):
    config = json.loads(args.config.read_text())
    # The follower's permissive mask (S>=90) also matches skin, which would
    # silently calibrate against a bare arm and produce a plausible but wrong
    # extrinsic. A saturated red ball sits well above S=170; skin does not.
    config = {**config, 'min_saturation': args.min_saturation}
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    samples = []
    if args.out.exists() and not args.overwrite:
        samples = json.loads(args.out.read_text())['samples']
        print(f'Appending to {len(samples)} existing samples in {args.out}', flush=True)
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        cam = Camera.from_robot(robot, args.camera)
        motion = MotionClient.from_robot(robot, args.motion)
        p = (await cam.get_properties(timeout=10)).intrinsic_parameters
        intrinsics = SimpleNamespace(fx=p.focal_x_px, fy=p.focal_y_px, cx=p.center_x_px,
                                     cy=p.center_y_px, width=p.width_px, height=p.height_px)
        print(f'{args.camera}: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} '
              f'cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f} {intrinsics.width}x{intrinsics.height}',
              flush=True)
        print(f'Ball radius {config["ball_radius_mm"]:.1f} mm. Move the arm in manual mode, then '
              'press Enter to capture. Type q then Enter to finish.', flush=True)
        print('Vary the WRIST ORIENTATION between poses, not just the position: with a fixed '
              'orientation the ball offset cannot be separated from the camera translation.',
              flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        while True:
            answer = sys.stdin.readline()
            if not answer or answer.strip().lower() == 'q':
                break
            try:
                sample, candidates, bgr, (rotation, position) = await observe(
                    cam, motion, config, args, intrinsics)
            except Exception as error:
                print(f'  capture failed: {error}', flush=True)
                continue
            print(f'  gripper ({position[0]:7.1f}, {position[1]:7.1f}, {position[2]:7.1f}) mm '
                  f'| {len(candidates)} red blobs', flush=True)
            for rank, (pixel, radius_px, truncated) in enumerate(candidates[:4]):
                mark = '<-' if rank == args.pick else '  '
                print(f'   {mark} [{rank}] uv=({pixel[0]:6.0f},{pixel[1]:6.0f}) r={radius_px:5.1f}px'
                      + (' TRUNCATED' if truncated else ''), flush=True)
            if sample is None:
                print('  no usable blob at --pick index; not recorded', flush=True)
                continue
            if len(candidates) > 1:
                print(f'  {len(candidates)} candidates survived S>={args.min_saturation:.0f}; '
                      'check the saved frame that [--pick] is the ball', flush=True)
            size = np.array(sample['camera_size'])
            line = f'  size  -> ({size[0]:7.1f}, {size[1]:7.1f}, {size[2]:7.1f}) mm'
            if sample['camera_depth'] is not None:
                depth_point = np.array(sample['camera_depth'])
                line += (f'\n  depth -> ({depth_point[0]:7.1f}, {depth_point[1]:7.1f}, '
                         f'{depth_point[2]:7.1f}) mm | disagreement '
                         f'{np.linalg.norm(depth_point-size):.1f} mm')
            print(line, flush=True)
            samples.append(sample)
            if args.save_frames:
                # Keep the raw frame too: the annotation draws over the ball's
                # edge, so a marked-up image cannot be re-measured later.
                cv2.imwrite(str(args.out.parent/f'raw{len(samples):02d}.jpg'), bgr)
                annotated = bgr.copy()
                pixel = candidates[args.pick][0]
                cv2.circle(annotated, (int(pixel[0]), int(pixel[1])),
                           int(candidates[args.pick][1]), (0, 255, 0), 1)
                cv2.drawMarker(annotated, (int(pixel[0]), int(pixel[1])), (0, 255, 0),
                               cv2.MARKER_CROSS, 12, 1)
                cv2.imwrite(str(args.out.parent/f'calib{len(samples):02d}.jpg'), annotated)
            args.out.write_text(json.dumps(
                {'camera': args.camera, 'ball_radius_mm': config['ball_radius_mm'],
                 'samples': samples}, indent=1))
            spread = orientation_spread_deg([s['gripper_rotation'] for s in samples])
            print(f'  recorded {len(samples)} samples | orientation spread {spread:.1f} deg',
                  flush=True)
    args.out.write_text(json.dumps({'camera': args.camera, 'ball_radius_mm': config['ball_radius_mm'],
                                    'samples': samples}, indent=1))
    print(f'Wrote {len(samples)} samples to {args.out}', flush=True)


def solve(args):
    data = json.loads(args.samples.read_text())
    samples = data['samples']
    key = 'camera_'+args.source
    usable = [s for s in samples if s.get(key) is not None]
    if len(usable) < len(samples):
        print(f'{len(samples)-len(usable)} samples have no {args.source} estimate; skipping them',
              flush=True)
    points = np.array([s[key] for s in usable], float)
    rotations = np.array([s['gripper_rotation'] for s in usable], float)
    positions = np.array([s['gripper_position'] for s in usable], float)
    rotation, translation, offset, residuals, scale = solve_eye_to_hand(
        points, rotations, positions, estimate_scale=args.estimate_scale)
    print(f'Solved from {len(usable)} poses using the {args.source} estimate.', flush=True)
    print(f'Orientation spread {orientation_spread_deg(rotations):.1f} deg', flush=True)
    print(f'Residual mm: mean {residuals.mean():.1f} | median {np.median(residuals):.1f} '
          f'| worst {residuals.max():.1f} (pose {int(np.argmax(residuals))})', flush=True)
    if args.estimate_scale:
        radius = data['ball_radius_mm']*scale
        print(f'Scale {scale:.4f}: the assumed {data["ball_radius_mm"]:.1f} mm ball radius implies a '
              f'real radius of {radius:.1f} mm ({2*radius:.0f} mm across).', flush=True)
        if abs(scale-1) > .05:
            print('  Measure the ball and set ball_radius_mm to it, then re-solve without '
                  '--estimate-scale to confirm the residuals stay small.', flush=True)
    print(f'Ball centre in the gripper frame: ({offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}) mm',
          flush=True)
    print(f'{data["camera"]} origin in world: ({translation[0]:.1f}, {translation[1]:.1f}, '
          f'{translation[2]:.1f}) mm', flush=True)
    print('\nPaste into the camera component\'s `frame` in the Viam config:', flush=True)
    print(json.dumps(viam_frame(rotation, translation), indent=2), flush=True)
    worst = residuals.max()
    if worst > args.tolerance_mm:
        print(f'\nWORST RESIDUAL {worst:.1f} mm EXCEEDS {args.tolerance_mm:.0f} mm. Drop that pose '
              'and re-solve, or re-record: a misdetected ball or a moved camera shows up here.',
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    grab = subparsers.add_parser('capture', help='Record one sample per Enter press')
    grab.add_argument('--config', type=Path, default=Path(__file__).parents[1]/'calibrate_cam2.config.json')
    grab.add_argument('--machine-config', type=Path)
    grab.add_argument('--camera', default='cam2')
    grab.add_argument('--gripper', default='gripper')
    grab.add_argument('--motion', default='builtin')
    grab.add_argument('--world-frame', default='world')
    grab.add_argument('--color-source', default='color')
    grab.add_argument('--depth-source', default='depth')
    grab.add_argument('--pick', type=int, default=0, help='Which ranked red blob is the ball')
    grab.add_argument('--min-saturation', type=float, default=140,
                      help='Red mask saturation floor for calibration. Skin measures around S=80-115 '
                           'and a lit red ball well above it, but the gap moves with the lighting: '
                           'read the actual numbers from red_probe and set this between them')
    grab.add_argument('--out', type=Path, default=Path('cam2_samples.json'))
    grab.add_argument('--overwrite', action='store_true')
    grab.add_argument('--save-frames', action='store_true', help='Write an annotated JPEG per capture')

    fit = subparsers.add_parser('solve', help='Fit the camera pose from recorded samples')
    fit.add_argument('--samples', type=Path, default=Path('cam2_samples.json'))
    fit.add_argument('--source', choices=('depth', 'size'), default='size',
                     help='Which per-sample centre estimate to fit; size needs no depth alignment')
    fit.add_argument('--tolerance-mm', type=positive, default=15.)
    fit.add_argument('--estimate-scale', action='store_true',
                     help='Also solve for a scale factor, which absorbs and measures a wrong '
                          'assumed ball radius; a rigid fit cannot and shows it as residual')

    args = parser.parse_args()
    try:
        if args.command == 'capture':
            asyncio.run(capture(args))
        else:
            solve(args)
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as error:
        parser.exit(2, f'{args.command} failed: {error}\n')


if __name__ == '__main__':
    main()
