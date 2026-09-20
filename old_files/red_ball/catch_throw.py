"""One-shot interception of a thrown ball, then an optional wrist-camera nudge.

    ./catch-throw.sh                 # preview: predicts and reports, moves nothing
    ./catch-throw.sh --execute       # opens the gripper, commits once, closes

Deliberately uses the WRIST camera as the tracker. Its frame is already
calibrated, and while the arm is parked the camera pose comes straight from
forward kinematics -- no pose interpolation and no exposure-timestamp problem,
which is what makes wrist-camera tracking unreliable during motion. The arm is
stationary for the whole observation phase, so that objection does not apply.
The cost is that tracking stops the moment the arm commits: this is open-loop
interception plus one optional correction, not closed-loop catching.

Sequence per attempt:
  1. park, gripper open, watch the throw zone
  2. fit free flight; require the prediction to be stable before committing
  3. solve for a reachable interception the arm can still reach in time
  4. move there, verify arrival
  5. optionally re-observe and nudge (--adjust)
  6. close the gripper timed on the predicted arrival

A fixed second camera can replace step 1 once its extrinsics are known; nothing
below assumes the tracker is on the wrist except `observe`.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient
from viam.robot.client import RobotClient
from viam.proto.common import Pose, PoseInFrame

from motion.viam_runtime import credentials, decode_color, positive
from old_files.red_ball.track_red import locate_target, estimate_distance, camera_tilt_deg
from motion.arm_workspace import DirectArmMove
from motion.live_camera_pose import pose_matrix, matrix_pose
from motion.camera_geometry import camera_transform
from motion.ballistic import (BallisticFit, ArmTiming, intercept, reachable,
                              required_lead, catch_pose)
from motion.prediction_gate import PredictionGate


async def read_pose(motion, name, world, timeout):
    pose = await asyncio.wait_for(motion.get_pose(name, world, timeout=timeout), timeout)
    if pose.reference_frame != world:
        raise ValueError(f'Unexpected reference frame for {name}')
    return pose_matrix(pose.pose)


def ball_in_world(center, size_px, physical_mm, origin, rotation, intrinsics, config):
    """Camera-frame ray at the estimated range, lifted into world coordinates."""
    if physical_mm is None:
        return None, 'ball on the image border: range unusable'
    distance = estimate_distance(size_px, intrinsics, physical_mm)
    if not config['min_distance_mm'] <= distance <= config['max_distance_mm']:
        return None, f'ball at {distance:.0f} mm, outside the tracking range'
    ray = np.array([(center[0]-intrinsics.cx)/intrinsics.fx,
                    (center[1]-intrinsics.cy)/intrinsics.fy, 1.])
    ray = ray/np.linalg.norm(ray)
    return origin+rotation@(ray*distance), None


async def commit(arm, gripper, motion, mover, config, target, velocity, arrival, *,
                 adjust_fn=None):
    """Place the catcher's mouth on the interception point before the ball arrives.

    With a bowl there is nothing to time: it catches passively, so the arm
    simply has to be there first and hold still. With jaws the closure is timed
    on the predicted arrival instead.
    """
    bowl = config.get('catcher', 'gripper') == 'bowl'
    flange_target, rotation = catch_pose(target, velocity, config)
    pose = matrix_pose(flange_target, rotation)
    destination = PoseInFrame(reference_frame=config['world_frame'], pose=pose)

    async def halt():
        results = await asyncio.gather(asyncio.wait_for(arm.stop(timeout=3), 3),
                                       asyncio.wait_for(gripper.stop(timeout=3), 3),
                                       return_exceptions=True)
        for name, result in zip(('arm', 'gripper'), results):
            if isinstance(result, BaseException):
                print(f'STOP FAILED ({name}): {result}', flush=True)

    try:
        budget = arrival-time.monotonic()-(0. if bowl else config['close_lead_s'])
        if budget <= 0:
            raise TimeoutError('Missed the interception before commanding the move')
        print(f'COMMIT: catch point {np.round(target, 1).tolist()} mm -> flange '
              f'{np.round(flange_target, 1).tolist()} mm, {budget:.2f}s to arrival', flush=True)
        ok = await asyncio.wait_for(mover.move(component_name=config['arm'],
                                               destination=destination, timeout=budget), budget)
        if not ok:
            raise RuntimeError('Planner did not complete the interception move')
        actual = await read_pose(motion, config['arm'], config['world_frame'], config['rpc_timeout_s'])
        error = float(np.linalg.norm(actual[0]-flange_target))
        print(f'ARRIVED: {error:.1f} mm from the interception point', flush=True)
        if error > config['arrival_tolerance_mm']:
            raise ValueError(f'Arm stopped {error:.1f} mm away; not closing on a bad pose')
        if adjust_fn is not None:
            await adjust_fn(actual)
        if bowl:
            remaining = arrival-time.monotonic()
            print(f'HOLDING: bowl in place, ball due in {remaining:.2f}s', flush=True)
            if remaining > 0:
                await asyncio.sleep(remaining)
            # Nothing on this arm senses a ball landing in a bowl, so this
            # reports that the catcher was in position, not that it caught.
            print('IN POSITION at the predicted arrival. Whether the ball landed is for you to '
                  'see: no sensor here can tell.', flush=True)
            return None
        remaining = arrival-config['close_lead_s']-time.monotonic()
        if remaining < -config['close_slip_s']:
            raise TimeoutError(f'Closing deadline missed by {-remaining:.2f}s')
        if remaining > 0:
            print(f'WAIT: closing in {remaining:.2f}s', flush=True)
            await asyncio.sleep(remaining)
        grabbed = bool(await asyncio.wait_for(gripper.grab(timeout=config['rpc_timeout_s']),
                                              config['rpc_timeout_s']))
        print('CAUGHT: gripper reports an object' if grabbed else 'MISS: gripper reports nothing',
              flush=True)
        return grabbed
    except BaseException:
        await asyncio.shield(halt())
        raise


async def run(args):
    config = json.loads(args.config.read_text())
    timing = ArmTiming(latency_s=config['arm_latency_s'], speed_mm_s=config['arm_speed_mm_s'])
    if not config.get('timing_measured'):
        print('WARNING: arm_latency_s/arm_speed_mm_s are NOT measured. Run measure-timing.sh and '
              'set them, or every feasibility decision below is a guess.', flush=True)
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        arm = Arm.from_robot(robot, config['arm'])
        cam = Camera.from_robot(robot, config['camera'])
        motion = MotionClient.from_robot(robot, config['motion'])
        gripper = Gripper.from_robot(robot, config['gripper']) if args.execute else None
        if await arm.is_moving(timeout=config['rpc_timeout_s']):
            raise ValueError('Start with the arm parked and stationary')
        before = np.array((await arm.get_joint_positions(timeout=config['rpc_timeout_s'])).values)
        flange = await read_pose(motion, config['arm'], config['world_frame'], config['rpc_timeout_s'])
        if not reachable(flange[0], config):
            raise ValueError(f'Park pose {flange[0].round(1).tolist()} mm is outside catch_box_mm '
                             'or the reach limits; move the arm into the catch zone first')
        origin, rotation = await camera_transform(robot, config['camera'], config['world_frame'])
        p = (await cam.get_properties(timeout=config['rpc_timeout_s'])).intrinsic_parameters
        intrinsics = SimpleNamespace(fx=p.focal_x_px, fy=p.focal_y_px, cx=p.center_x_px,
                                     cy=p.center_y_px, width=p.width_px, height=p.height_px)
        after = np.array((await arm.get_joint_positions(timeout=config['rpc_timeout_s'])).values)
        if before.shape != after.shape or np.max(abs(after-before)) > .1:
            raise ValueError('Arm moved during startup; it must be parked')
        print(f'Parked flange {flange[0].round(1).tolist()} mm | camera {origin.round(1).tolist()} mm '
              f'| tilt {camera_tilt_deg(rotation):+.1f} deg', flush=True)
        print(f'Timing: {timing.latency_s:.3f}s + distance/{timing.speed_mm_s:.0f} mm/s | '
              f'minimum lead for a 200 mm move {required_lead(200, config, timing):.2f}s', flush=True)
        if args.execute and config.get('catcher', 'gripper') != 'bowl':
            print('PREPARING: opening gripper. Do not throw yet.', flush=True)
            await asyncio.wait_for(gripper.open(timeout=config['rpc_timeout_s']),
                                   config['rpc_timeout_s'])
        elif args.execute:
            print(f'Catcher: bowl, mouth {config["tool_offset_mm"]:.0f} mm beyond the flange, '
                  f'radius {config["bowl_radius_mm"]:.0f} mm. The gripper is never commanded.',
                  flush=True)
        print('READY: throw the ball. ' + ('ARM WILL CATCH.' if args.execute
                                           else 'PREVIEW: nothing will move.'), flush=True)

        fit = BallisticFit(max_gap_s=config['max_gap_s'], max_residual_mm=config['max_residual_mm'],
                           min_samples=config['min_samples'])
        gate = PredictionGate()
        mover = motion if args.planned else DirectArmMove(arm)
        last_stamp = None
        started = last_report = time.monotonic()
        frames = seen = 0
        last_print = 0.
        try:
            while not args.duration or time.monotonic()-started < args.duration:
                images, meta = await cam.get_images(timeout=.3)
                now = time.monotonic()
                stamp = meta.captured_at.seconds+meta.captured_at.nanos/1e9
                age = time.time()-stamp
                status = None
                if not 0 <= age <= config['max_frame_age_s']:
                    status = 'WAIT: stale camera frame'
                elif last_stamp is None or stamp > last_stamp:
                    last_stamp = stamp
                    frames += 1
                    bgr = decode_color(next(im for im in images if im.name == args.color_source))
                    center, size_px, physical_mm, reason = locate_target(bgr, config)
                    if center is None:
                        status = reason
                    else:
                        point, why = ball_in_world(center, size_px, physical_mm, origin, rotation,
                                                   intrinsics, config)
                        if point is None:
                            status = why
                        else:
                            seen += 1
                            # Observation clock is the camera's; convert once so
                            # every deadline below is on the monotonic clock.
                            flight = fit.add(now-age, point)
                            status = f'BALL {np.round(point, 0).tolist()} mm'
                            if flight is None:
                                gate.reset()
                            else:
                                found, refusal = intercept(flight, flange[0], config, timing, now,
                                                           prefer=args.prefer)
                                if found is None:
                                    status += f' | {refusal}'
                                    gate.reset()
                                else:
                                    status += (f' | v={np.linalg.norm(flight.velocity):.0f} mm/s '
                                               f'fit={flight.residual_mm:.0f} mm -> catch at '
                                               f'{np.round(found.point, 0).tolist()} in '
                                               f'{found.arrival-now:.2f}s (slack {found.slack_s:.2f}s)')
                                    if not gate.add(now, found.point, found.arrival):
                                        status += ' | settling'
                                    elif not args.execute:
                                        status += ' | WOULD CATCH'
                                        gate.reset()
                                    else:
                                        print(status, flush=True)
                                        _, ball_velocity = flight.at(found.arrival)
                                        return await commit(arm, gripper, motion, mover, config,
                                                            found.point, ball_velocity,
                                                            found.arrival)
                if status and now-last_print >= .15:
                    print(status, flush=True)
                    last_print = now
                if now-last_report >= 2:
                    elapsed = now-last_report
                    print(f'Frames {frames/elapsed:.1f} Hz | ball {seen/elapsed:.1f} Hz', flush=True)
                    frames = seen = 0
                    last_report = now
                await asyncio.sleep(.002)
        finally:
            if args.execute:
                try:
                    await asyncio.shield(asyncio.wait_for(arm.stop(timeout=3), 3))
                except Exception as error:
                    print(f'STOP FAILED (arm): {error}', flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('catch_throw.config.json'))
    parser.add_argument('--machine-config', type=Path)
    parser.add_argument('--execute', action='store_true', help='Open the gripper, commit and close')
    parser.add_argument('--planned', action='store_true',
                        help='Route the interception move through the builtin motion service')
    parser.add_argument('--prefer', choices=('slowest', 'earliest'), default='slowest',
                        help='slowest catches nearer the apex; earliest keeps the largest time buffer')
    parser.add_argument('--color-source', default='color')
    parser.add_argument('--duration', type=positive)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as error:
        parser.exit(2, f'Catch failed: {error}\n')


if __name__ == '__main__':
    main()
