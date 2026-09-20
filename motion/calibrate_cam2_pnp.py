"""Locate a fixed camera in the world by PnP, using the ball the arm already holds.

The previous attempt registered two 3D point sets and failed: the ball's position
in cam2's frame needs a RANGE, and range is the one quantity this rig measures
badly -- cam2's depth is not aligned to its colour (it read the wall behind the
ball) and the projected radius came out of a mask whose edge moved from frame to
frame. Residuals stayed above 100 mm.

PnP removes that error source entirely. The ball's world position is computed
from the machine's own kinematics -- the gripper frame is calibrated, and the
ball's offset inside the bowl was measured to a 2 mm spread -- so each sample is
a known 3D point paired with a pixel. Pixels are good to about one part in a
thousand of the image. Nothing here depends on knowing how far away anything is.

    ./calibrate-cam2-pnp.sh collect --out cam2_pnp.json   # press Enter per pose
    ./calibrate-cam2-pnp.sh solve   --samples cam2_pnp.json

Read-only: it never commands the arm.
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
from motion.handeye import rotation_to_quaternion, viam_frame
from motion.catch_plane import green_ball, ball_point
from motion.camera_geometry import camera_transform


def ball_world(gripper_position, gripper_rotation, config):
    """Where the ball would sit if it never shifted inside the bowl.

    Used only as a reference to compare against. The ball rests loose in a bowl
    90 mm across, so moving the arm rolls it: assuming a fixed offset put 7 of
    12 samples outside a 4 px reprojection tolerance. The wrist camera measures
    where it actually is instead.
    """
    return (np.asarray(gripper_position, float)
            + np.asarray(gripper_rotation, float)
            @ np.asarray(config['bowl_offset_gripper_mm'], float))


async def measured_ball_world(robot, wrist, motion, config, args, intrinsics):
    """The ball's world position as the wrist camera sees it, arm stationary.

    Range here is about 250 mm, where the projected-size estimate was checked
    against the machine's kinematics and agreed to the millimetre -- far better
    than it manages at the metre-plus distances cam2 works at.
    """
    origin, rotation = await camera_transform(robot, config['camera'], config['world_frame'])
    images, _ = await wrist.get_images(timeout=10)
    sources = {im.name: im for im in images}
    bgr = decode_color(sources[args.color_source])
    near = {**config, 'min_distance_mm': 80, 'max_distance_mm': 900, 'range_source': 'size',
            'max_range_disagreement': 10., 'throw_volume_mm': [[-9e3]*3, [9e3]*3]}
    for centre, radius_px, enclosing in green_ball(bgr, near):
        point, *_ = ball_point(centre, radius_px, enclosing, origin, rotation,
                               intrinsics, near, None)
        if point is not None:
            return point
    return None


# SOLVEPNP_ITERATIVE, not SQPNP or EPNP. On OpenCV 4.14.0 with these millimetre
# world coordinates both of the others returned poses that reprojected over a
# thousand pixels away from the input they were given, while reporting success.
# Every call below is checked by its own reprojection error, which is what
# caught it.
PNP_METHOD = cv2.SOLVEPNP_ITERATIVE


def reproject(rvec, tvec, points, intrinsics):
    matrix = np.array([[intrinsics.fx, 0, intrinsics.cx],
                       [0, intrinsics.fy, intrinsics.cy], [0, 0, 1.]])
    projected, _ = cv2.projectPoints(np.asarray(points, float).reshape(-1, 1, 3),
                                     rvec, tvec, matrix, None)
    return projected.reshape(-1, 2)


def solve_pose_and_offset(gripper_positions, gripper_rotations, pixels, intrinsics, *,
                          offset_guess=(0., 0., 0.), tolerance_px=4.0, rounds=25):
    """Camera pose AND where the ball sits in the gripper frame, from pixels alone.

    The offset cannot be assumed: it was measured for a ball resting in the bowl
    and the ball was later gripped directly instead, which moved it enough to
    throw most samples outside tolerance.

    Solved by alternating, not as one nine-parameter fit. Rotation in radians
    and translation in millimetres span too many orders of magnitude for a
    hand-rolled damped Newton step to handle -- that version settled 300 px from
    a solution that fits exactly. Here OpenCV's PnP does the six-parameter half
    and a three-parameter Gauss-Newton does the offset, each well conditioned.

    Requires the wrist orientation to VARY: with it fixed, `R_i @ b` is constant
    and the offset is indistinguishable from the camera translation.

    Returns (rotation, translation, offset, per-sample error, inlier mask).
    """
    positions = np.asarray(gripper_positions, float)
    rotations = np.asarray(gripper_rotations, float)
    pixels = np.asarray(pixels, float)
    if len(positions) < 6:
        raise ValueError('Solving pose and offset together needs at least six poses')
    spread = 0.
    for i in range(len(rotations)):
        for j in range(i+1, len(rotations)):
            cosine = (np.trace(rotations[i].T@rotations[j])-1)/2
            spread = max(spread, float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    if spread < 15:
        raise ValueError(f'Wrist orientation varies by only {spread:.1f} deg. Below 15 deg the '
                         'ball offset cannot be separated from the camera translation.')

    matrix = np.array([[intrinsics.fx, 0, intrinsics.cx],
                       [0, intrinsics.fy, intrinsics.cy], [0, 0, 1.]])

    def world_of(offset):
        return np.einsum('nij,j->ni', rotations, offset)+positions

    def errors_of(rvec, tvec, offset):
        return np.linalg.norm(reproject(rvec, tvec, world_of(offset), intrinsics)-pixels, axis=1)

    offset = np.asarray(offset_guess, float)
    keep = np.ones(len(pixels), bool)
    rvec = tvec = None
    for round_index in range(rounds):
        # Plain PnP while the offset is still wrong. RANSAC cannot help there:
        # a wrong offset displaces EVERY point, in a pose-dependent way, so no
        # subset agrees and it simply reports failure. Outliers are separated
        # afterwards, once the offset has settled.
        chosen = np.flatnonzero(keep)
        ok, rvec, tvec = cv2.solvePnP(world_of(offset)[chosen], pixels[chosen], matrix, None,
                                      flags=PNP_METHOD)
        if not ok:
            raise ValueError('PnP did not converge; check that the picked blob is the ball')
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            raise ValueError('PnP returned a non-finite pose')
        rvec, tvec = cv2.solvePnPRefineLM(world_of(offset)[chosen], pixels[chosen],
                                          matrix, None, rvec, tvec)
        # Three-parameter Gauss-Newton on the offset, camera held fixed.
        previous = offset.copy()
        for _ in range(12):
            base = (reproject(rvec, tvec, world_of(offset)[chosen], intrinsics)
                    - pixels[chosen]).ravel()
            jacobian = np.empty((base.size, 3))
            for axis in range(3):
                step = np.zeros(3)
                step[axis] = .5
                shifted = (reproject(rvec, tvec, world_of(offset+step)[chosen], intrinsics)
                           - pixels[chosen]).ravel()
                jacobian[:, axis] = (shifted-base)/step[axis]
            try:
                delta = np.linalg.lstsq(jacobian, -base, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            offset = offset+delta
            if np.linalg.norm(delta) < 1e-6:
                break
        settled = errors_of(rvec, tvec, offset)
        # Only start discarding samples once the offset has stopped moving, so a
        # bad early estimate cannot decide which observations are "wrong".
        if round_index >= 3:
            updated = settled <= max(tolerance_px, 3*np.median(settled))
            if updated.sum() >= 6:
                keep = updated
        if np.linalg.norm(offset-previous) < 1e-4 and round_index >= 3:
            break

    errors = errors_of(rvec, tvec, offset)
    world_to_camera, _ = cv2.Rodrigues(rvec)
    rotation = world_to_camera.T
    translation = (-rotation@np.asarray(tvec).reshape(3, 1)).ravel()
    return rotation, translation, offset, errors, keep


def solve_pose(world_points, pixels, intrinsics, *, reprojection_tolerance_px=3.0):
    """Camera-to-world transform from known 3D points and their projections.

    Returns (rotation, translation, per-sample reprojection error, inlier mask).
    """
    world_points = np.asarray(world_points, float)
    pixels = np.asarray(pixels, float)
    if len(world_points) < 4:
        raise ValueError('PnP needs at least four poses; use many more in practice')
    if world_points.shape[0] != pixels.shape[0]:
        raise ValueError('Each world point needs exactly one pixel')
    spread = np.linalg.svd(world_points-world_points.mean(axis=0), compute_uv=False)
    if spread[2] < 1e-6*max(spread[0], 1e-9):
        raise ValueError('The sampled points are coplanar; move the arm in depth as well')
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.cx],
                              [0, intrinsics.fy, intrinsics.cy],
                              [0, 0, 1.]])
    # The colour stream is assumed rectified; no distortion model is applied.
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        world_points, pixels, camera_matrix, None,
        reprojectionError=reprojection_tolerance_px, flags=PNP_METHOD)
    if not ok:
        raise ValueError('PnP did not converge; check that the picked blob is the ball')
    rvec, tvec = cv2.solvePnPRefineLM(
        world_points[inliers.ravel()], pixels[inliers.ravel()], camera_matrix, None, rvec, tvec)
    projected, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, None)
    errors = np.linalg.norm(projected.reshape(-1, 2)-pixels, axis=1)
    world_to_camera, _ = cv2.Rodrigues(rvec)
    # solvePnP gives world->camera; the frame system wants the camera in world.
    rotation = world_to_camera.T
    translation = (-rotation@tvec).ravel()
    mask = np.zeros(len(world_points), bool)
    mask[inliers.ravel()] = True
    return rotation, translation, errors, mask


async def collect(args):
    config = json.loads(args.config.read_text())
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    samples = []
    if args.out.exists() and not args.overwrite:
        samples = json.loads(args.out.read_text())['samples']
        print(f'Appending to {len(samples)} existing samples', flush=True)
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        cam = Camera.from_robot(robot, args.camera)
        wrist = Camera.from_robot(robot, config['camera'])
        motion = MotionClient.from_robot(robot, config['motion'])
        w = (await wrist.get_properties(timeout=10)).intrinsic_parameters
        wrist_intrinsics = SimpleNamespace(fx=w.focal_x_px, fy=w.focal_y_px, cx=w.center_x_px,
                                           cy=w.center_y_px, width=w.width_px, height=w.height_px)
        p = (await cam.get_properties(timeout=10)).intrinsic_parameters
        print(f'{args.camera}: fx={p.focal_x_px:.1f} fy={p.focal_y_px:.1f} '
              f'cx={p.center_x_px:.1f} cy={p.center_y_px:.1f} {p.width_px}x{p.height_px}', flush=True)
        print(f'Ball offset in the gripper frame: {config["bowl_offset_gripper_mm"]} mm', flush=True)
        print('Put the ball in the bowl. Move the arm in manual mode so the ball reaches a '
              'DIFFERENT part of cam2\'s view each time, including nearer and further, then '
              'press Enter. q then Enter to finish.', flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        while True:
            answer = sys.stdin.readline()
            if not answer or answer.strip().lower() == 'q':
                break
            try:
                pose = await asyncio.wait_for(
                    motion.get_pose(config['gripper'], config['world_frame'], timeout=10), 10)
                if pose.reference_frame != config['world_frame']:
                    raise ValueError('Unexpected gripper reference frame')
                gripper_position, gripper_rotation = pose_matrix(pose.pose)
                images, _ = await cam.get_images(timeout=10)
                bgr = decode_color(next(im for im in images if im.name == args.color_source))
                found = green_ball(bgr, config)
            except Exception as error:
                print(f'  capture failed: {error}', flush=True)
                continue
            if not found:
                print('  no green ball in cam2; not recorded', flush=True)
                continue
            if len(found) > 1:
                print(f'  {len(found)} green blobs; using the one at '
                      f'({found[args.pick][0][0]:.0f}, {found[args.pick][0][1]:.0f}) -- check the '
                      'saved frame', flush=True)
            centre, _, enclosing = found[args.pick]
            assumed = ball_world(gripper_position, gripper_rotation, config)
            point = await measured_ball_world(robot, wrist, motion, config, args, wrist_intrinsics)
            if point is None:
                print('  wrist camera cannot see the ball in the bowl; not recorded', flush=True)
                continue
            rolled = float(np.linalg.norm(point-assumed))
            samples.append({'pixel': [float(centre[0]), float(centre[1])],
                            'radius_px': float(enclosing),
                            'world': point.tolist(),
                            'assumed_world': assumed.tolist(),
                            'gripper_position': gripper_position.tolist(),
                            'gripper_rotation': gripper_rotation.tolist()})
            print(f'  {len(samples):2d}: ball world {point.round(1).tolist()} mm -> '
                  f'pixel ({centre[0]:.1f}, {centre[1]:.1f}) r={enclosing:.1f}px | '
                  f'rolled {rolled:.0f} mm from the assumed spot', flush=True)
            if args.save_frames:
                marked = bgr.copy()
                cv2.circle(marked, (int(centre[0]), int(centre[1])), int(enclosing), (0, 255, 0), 1)
                cv2.imwrite(str(args.out.parent/f'pnp{len(samples):02d}.jpg'), marked)
            coverage = np.array([s['pixel'] for s in samples])
            depth = np.array([s['world'] for s in samples])
            print(f'     image coverage {np.ptp(coverage[:, 0]):.0f}x{np.ptp(coverage[:, 1]):.0f} px '
                  f'of {p.width_px}x{p.height_px} | world spread '
                  f'{np.ptp(depth, axis=0).round(0).tolist()} mm', flush=True)
            args.out.write_text(json.dumps({'camera': args.camera, 'samples': samples,
                                            'intrinsics': {'fx': p.focal_x_px, 'fy': p.focal_y_px,
                                                           'cx': p.center_x_px, 'cy': p.center_y_px,
                                                           'width': p.width_px,
                                                           'height': p.height_px}}, indent=1))
    print(f'Wrote {len(samples)} samples to {args.out}', flush=True)


def solve(args):
    data = json.loads(args.samples.read_text())
    samples = data['samples']
    intrinsics = SimpleNamespace(**data['intrinsics'])
    pixels = np.array([s['pixel'] for s in samples], float)
    positions = np.array([s['gripper_position'] for s in samples], float)
    rotations = np.array([s['gripper_rotation'] for s in samples], float)
    assumed = np.array([s['world'] for s in samples], float)
    guess = rotations[0].T@(assumed[0]-positions[0])
    rotation, translation, offset, errors, inliers = solve_pose_and_offset(
        positions, rotations, pixels, intrinsics, offset_guess=guess,
        tolerance_px=args.tolerance_px)
    world = np.einsum('nij,j->ni', rotations, offset)+positions
    print(f'Ball in the gripper frame: {offset.round(1).tolist()} mm '
          f'(started from {np.round(guess, 1).tolist()})', flush=True)
    print(f'Solved from {int(inliers.sum())} of {len(samples)} samples '
          f'({len(samples)-int(inliers.sum())} rejected as outliers).', flush=True)
    kept = errors[inliers]
    print(f'Reprojection error px: mean {kept.mean():.2f} | worst {kept.max():.2f}', flush=True)
    print(f'Image coverage {np.ptp(pixels[:, 0]):.0f}x{np.ptp(pixels[:, 1]):.0f} px of '
          f'{intrinsics.width:.0f}x{intrinsics.height:.0f}', flush=True)
    print(f'World spread {np.ptp(world, axis=0).round(0).tolist()} mm', flush=True)
    print(f'\n{data["camera"]} origin in world: ({translation[0]:.1f}, {translation[1]:.1f}, '
          f'{translation[2]:.1f}) mm', flush=True)
    forward = rotation[:, 2]
    print(f'Optical axis {forward.round(3).tolist()} | elevation '
          f'{np.degrees(np.arcsin(np.clip(forward[2], -1, 1))):+.1f} deg', flush=True)
    for index, (error, keep) in enumerate(zip(errors, inliers)):
        if not keep or error > kept.mean()*3:
            print(f'  sample {index}: {error:.1f} px' + ('' if keep else '  REJECTED'), flush=True)
    print('\nPaste into cam2\'s `frame` in the Viam config:', flush=True)
    print(json.dumps(viam_frame(rotation, translation), indent=2), flush=True)
    if kept.mean() > args.warn_px:
        print(f'\nMean reprojection error {kept.mean():.2f} px exceeds {args.warn_px:.1f} px. '
              'At 1.5 m one pixel is about 2.5 mm, so this is roughly '
              f'{kept.mean()*2.5:.0f} mm of error there. Re-record with the ball spread more '
              'widely across the image and in depth.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    grab = subparsers.add_parser('collect')
    grab.add_argument('--config', type=Path,
                      default=Path(__file__).parents[1]/'catch_plane.config.json')
    grab.add_argument('--machine-config', type=Path)
    grab.add_argument('--camera', default='cam2')
    grab.add_argument('--color-source', default='color')
    grab.add_argument('--pick', type=int, default=0)
    grab.add_argument('--out', type=Path, default=Path('cam2_pnp.json'))
    grab.add_argument('--overwrite', action='store_true')
    grab.add_argument('--save-frames', action='store_true')
    fit = subparsers.add_parser('solve')
    fit.add_argument('--samples', type=Path, default=Path('cam2_pnp.json'))
    fit.add_argument('--tolerance-px', type=positive, default=4.0)
    fit.add_argument('--warn-px', type=positive, default=2.0)
    args = parser.parse_args()
    try:
        if args.command == 'collect':
            asyncio.run(collect(args))
        else:
            solve(args)
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as error:
        parser.exit(2, f'{args.command} failed: {error}\n')


if __name__ == '__main__':
    main()
