"""Measure where a caught ball actually sits, instead of guessing the bowl's offset.

Put the ball in the bowl, hold still, run this. It locates the ball with the
wrist camera, reads the gripper frame from the machine's own configuration, and
reports the offset between them -- which is exactly the point a throw should be
aimed at, and the number `bowl_offset_gripper_mm` wants.

Guessing this from photographs cost an 89 mm error once already: the catch plane
ended up above every throw, and each one was correctly but uselessly refused.

Read-only: it never commands the arm or the gripper.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from viam.components.camera import Camera
from viam.services.motion import MotionClient
from viam.robot.client import RobotClient

from motion.viam_runtime import credentials, decode_color, positive
from motion.live_camera_pose import pose_matrix
from motion.camera_geometry import camera_transform
from motion.catch_plane import green_ball, ball_point


async def run(args):
    config = json.loads(args.config.read_text())
    # The ball rests a few centimetres from the lens; the flight-time floor would
    # throw every sample away.
    config = {**config, 'min_distance_mm': 80, 'max_distance_mm': 900,
              'max_range_disagreement': 1.0,
              'throw_volume_mm': [[-5000, -5000, -5000], [5000, 5000, 5000]]}
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        cam = Camera.from_robot(robot, config['camera'])
        motion = MotionClient.from_robot(robot, config['motion'])
        gripper = await asyncio.wait_for(
            motion.get_pose(config['gripper'], config['world_frame'], timeout=5), 5)
        if gripper.reference_frame != config['world_frame']:
            raise ValueError('Unexpected gripper reference frame')
        gripper_position, gripper_rotation = pose_matrix(gripper.pose)
        origin, rotation = await camera_transform(robot, config['camera'], config['world_frame'])
        p = (await cam.get_properties(timeout=5)).intrinsic_parameters
        intrinsics = SimpleNamespace(fx=p.focal_x_px, fy=p.focal_y_px, cx=p.center_x_px,
                                     cy=p.center_y_px, width=p.width_px, height=p.height_px)
        print(f'Gripper frame {gripper_position.round(1).tolist()} mm', flush=True)
        print(f'Hold the ball still ANYWHERE in view and keep the arm still. Sampling '
              f'{args.samples}...', flush=True)
        print('  In the bowl -> the offset below is the bowl calibration.', flush=True)
        print('  Held out at a known height -> compare the reported world Z against it to check '
              'the range estimate at that distance.', flush=True)

        points, sizes, depths = [], [], []
        deadline = time.monotonic()+args.timeout
        misses = 0
        while len(points) < args.samples:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f'only {len(points)} of {args.samples} samples in {args.timeout:.0f}s '
                    f'({misses} frames with no usable ball). Check the ball is in the bowl and '
                    'visible, or loosen hue_min/hue_max and sat_min in the config.')
            images, _ = await cam.get_images(timeout=5)
            sources = {im.name: im for im in images}
            bgr = decode_color(sources[args.color_source])
            depth = None
            if args.depth_source in sources:
                depth = np.asarray(sources[args.depth_source].bytes_to_depth_array(), dtype=float)
            for centre, radius_px, enclosing in green_ball(bgr, config):
                point, by_size, by_depth, _ = ball_point(
                    centre, radius_px, enclosing, origin, rotation, intrinsics, config, depth)
                if point is None:
                    continue
                points.append(point)
                sizes.append(by_size)
                depths.append(by_depth if by_depth is not None else np.nan)
                print(f'  {len(points):2d}: world {np.round(point, 1).tolist()} mm | '
                      f'size {by_size:.0f} mm' +
                      ('' if by_depth is None else f' | depth {by_depth:.0f} mm'), flush=True)
                break
            else:
                misses += 1
                if misses % 40 == 0:
                    print(f'  ...{misses} frames with no usable ball yet', flush=True)
            await asyncio.sleep(.05)

        points = np.array(points)
        spread = float(np.max(np.linalg.norm(points-points.mean(axis=0), axis=1)))
        centre = points.mean(axis=0)
        offset = gripper_rotation.T@(centre-gripper_position)
        print(f'\nBall centre in world: {centre.round(1).tolist()} mm (spread {spread:.0f} mm)',
              flush=True)
        print(f'Gripper origin      : {gripper_position.round(1).tolist()} mm', flush=True)
        print(f'\nWorld Z of the ball: {centre[2]:.0f} mm', flush=True)
        print(f'Range from the camera: {float(np.linalg.norm(centre-origin)):.0f} mm', flush=True)
        print(f'\nIf the ball was in the bowl, set bowl_offset_gripper_mm to '
              f'{np.round(offset, 0).tolist()} (catch plane z={centre[2]:.0f} mm).', flush=True)
        if spread > 30:
            print(f'\nWARNING: samples scatter by {spread:.0f} mm, so this offset is only good to '
                  'about that. Check the ball is still and clearly visible.', flush=True)
        if np.isfinite(depths).any():
            gap = float(np.nanmean(np.abs(np.array(sizes)-np.array(depths))))
            print(f'Size and depth ranges differed by {gap:.0f} mm on average here.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parents[1]/'catch_plane.config.json')
    parser.add_argument('--machine-config', type=Path)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--timeout', type=positive, default=25.)
    parser.add_argument('--color-source', default='color')
    parser.add_argument('--depth-source', default='depth')
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as error:
        parser.exit(2, f'Bowl calibration failed: {error}\n')


if __name__ == '__main__':
    main()
