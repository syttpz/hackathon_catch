"""Measure how long the arm actually takes to make a Cartesian move.

Every timing budget in this repo -- the 1.5 s move budget, the 1.8 s minimum
catch lead -- is an assumption that has never been measured. A thrown ball is
airborne for well under a second, so whether catching one is possible at all
turns on these numbers.

Commands a series of out-and-back moves of growing distance from the current
pose and times each one three ways:

    submit   -> the move RPC returns
    settle   -> the arm reports is_moving() false
    total    -> submit to settled

then fits `total = latency + distance / speed`. The intercept is the fixed cost
of planning and starting a move regardless of distance; that cost is what a
short corrective move is dominated by, and it is the number that decides
whether a late correction can be made at all.

Preview prints the plan and moves nothing. --execute moves the arm.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from viam.components.arm import Arm
from viam.services.motion import MotionClient
from viam.robot.client import RobotClient
from viam.proto.common import Pose, PoseInFrame

from motion.viam_runtime import credentials, positive
from motion.live_camera_pose import pose_matrix, matrix_pose
from motion.arm_workspace import clamp_target, in_workspace, DirectArmMove


async def read_flange(motion, config, name='arm'):
    pose = await asyncio.wait_for(motion.get_pose(name, config['world_frame'], timeout=5), 5)
    if pose.reference_frame != config['world_frame']:
        raise ValueError('Unexpected flange reference frame')
    return pose_matrix(pose.pose)


async def timed_move(arm, mover, config, pose, timeout):
    """Submit one move and time the RPC and the settle separately."""
    started = time.monotonic()
    destination = PoseInFrame(reference_frame=config['world_frame'], pose=pose)
    ok = await asyncio.wait_for(mover.move(component_name='arm', destination=destination,
                                           timeout=timeout), timeout)
    submitted = time.monotonic()
    while await asyncio.wait_for(arm.is_moving(timeout=1), 1):
        if time.monotonic()-started > timeout:
            raise TimeoutError('Arm never reported a stop')
        await asyncio.sleep(.005)
    settled = time.monotonic()
    return bool(ok), submitted-started, settled-submitted, settled-started


async def run(args):
    config = json.loads(args.config.read_text())
    # The catch config calls the same box catch_box_mm; accept either spelling.
    if 'workspace_mm' not in config and 'catch_box_mm' in config:
        config = {**config, 'workspace_mm': config['catch_box_mm']}
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        arm = Arm.from_robot(robot, config['arm'])
        motion = MotionClient.from_robot(robot, config['motion'])
        if await arm.is_moving(timeout=5):
            raise ValueError('Start with the arm stationary')
        home = await read_flange(motion, config)
        if not in_workspace(home[0], config):
            raise ValueError(f'Flange at {home[0].round(1).tolist()} mm is outside workspace_mm; '
                             'move it into range first')
        axis = np.array(args.axis, float)
        axis = axis/np.linalg.norm(axis)
        print(f'Home {home[0].round(1).tolist()} mm | axis {axis.round(2).tolist()} | '
              f'{"planned (builtin motion)" if args.planned else "direct (move_to_position)"}',
              flush=True)

        plan = []
        for distance in args.distances:
            goal = clamp_target(home[0]+axis*distance, config)
            actual = float(np.linalg.norm(goal-home[0]))
            if actual < distance-1:
                print(f'  {distance:.0f} mm clipped to {actual:.0f} mm by the workspace bounds',
                      flush=True)
            if actual < 5:
                print(f'  {distance:.0f} mm skipped: no room along this axis', flush=True)
                continue
            plan.append((actual, matrix_pose(goal, home[1])))
        if not plan:
            raise ValueError('No usable distances; try --axis pointing into free space')
        if not args.execute:
            print('PREVIEW: no motion commanded. Distances that would be tested (mm): '
                  + ', '.join(f'{d:.0f}' for d, _ in plan), flush=True)
            print('Each is commanded out and back, repeated --repeats times. Add --execute to run.',
                  flush=True)
            return

        mover = motion if args.planned else DirectArmMove(arm)
        home_pose = matrix_pose(*home)
        rows = []
        try:
            for distance, pose in plan:
                for _ in range(args.repeats):
                    ok, submit, settle, total = await timed_move(arm, mover, config, pose, args.timeout)
                    rows.append((distance, submit, settle, total, ok))
                    print(f'  {distance:6.0f} mm | rpc {submit:5.3f}s | settle {settle:5.3f}s | '
                          f'total {total:5.3f}s' + ('' if ok else ' | PLANNER RETURNED FALSE'),
                          flush=True)
                    await timed_move(arm, mover, config, home_pose, args.timeout)
                    await asyncio.sleep(args.rest)
        finally:
            await asyncio.shield(asyncio.wait_for(arm.stop(timeout=3), 3))

        good = np.array([(d, t) for d, _, _, t, ok in rows if ok], float)
        if len(good) < 2:
            print('Too few successful moves to fit a model.', flush=True)
            return
        design = np.column_stack((np.ones(len(good)), good[:, 0]))
        (latency, inverse_speed), *_ = np.linalg.lstsq(design, good[:, 1], rcond=None)
        print(f'\nFit over {len(good)} moves: total = {latency:.3f}s + distance / '
              f'{1/inverse_speed if inverse_speed > 0 else float("inf"):.0f} mm/s', flush=True)
        residual = good[:, 1]-design@np.array([latency, inverse_speed])
        print(f'Fit residual: rms {np.sqrt((residual**2).mean()):.3f}s | worst {abs(residual).max():.3f}s',
              flush=True)
        print(f'\nFixed cost per move is {latency:.3f}s regardless of distance. A late correction '
              f'cannot be faster than that.', flush=True)
        for flight in (0.6, 0.8, 1.2):
            budget = flight-latency-config.get('close_lead_s', .15)
            reach = budget/inverse_speed if inverse_speed > 0 and budget > 0 else 0
            print(f'  {flight:.1f}s flight: after latency and a 0.15s close lead, '
                  + (f'{budget:.2f}s of motion left -> about {reach:.0f} mm of travel'
                     if reach > 0 else 'NO time left to move'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[1]/'catch_plane.config.json')
    parser.add_argument('--machine-config', type=Path)
    parser.add_argument('--execute', action='store_true', help='Actually move the arm')
    parser.add_argument('--planned', action='store_true',
                        help='Time the builtin motion service instead of arm.move_to_position')
    parser.add_argument('--axis', type=float, nargs=3, default=[0, 1, 0],
                        help='Direction to move along, in world mm (default +Y)')
    parser.add_argument('--distances', type=positive, nargs='+',
                        default=[25, 50, 100, 200, 300],
                        help='Move lengths to time, in mm')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rest', type=positive, default=.4, help='Pause between moves')
    parser.add_argument('--timeout', type=positive, default=15.)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as error:
        parser.exit(2, f'Timing run failed: {error}\n')


if __name__ == '__main__':
    main()
