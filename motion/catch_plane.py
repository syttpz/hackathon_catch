"""Catch a lobbed ball in a bowl held at one fixed height.

    ./catch-plane.sh                # preview: tracks and predicts, moves nothing
    ./catch-plane.sh --execute      # commits once, then nudges while it waits

Fixing the catch height and the wrist orientation reduces the commit to a short
planar move. That matters because the arm's cost per move was measured at
0.159 s + distance/435 mm/s: the fixed part dominates, so the only way a
sub-second flight survives is to keep the correction short and never reorient.
Interception is then one quadratic -- when the arc descends through the plane --
instead of a search over the trajectory.

Tracking runs continuously until the moment of commit. After the arm arrives it
keeps watching and will issue one further small correction if the prediction has
drifted and there is still time, which is the only closed-loop part of this.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.services.motion import MotionClient
from viam.robot.client import RobotClient
from viam.proto.common import Pose, PoseInFrame

from motion.trajectory_local import credentials, decode_color, positive
from motion.live_camera_pose import pose_matrix, matrix_pose
from motion.rolling_preview import camera_transform
from motion.ballistic import BallisticFit, ArmTiming, plane_intercept, reachable
from motion.rolling_catch import PredictionGate
from motion.catch_side import prepare_side
from motion.rough_cycle import RoughCycle, RoughHandoff, approaching_bowl
from motion.stereo_tracking import StereoTracker


DEFAULT_POSE = dict(x=-392.5960289174601, y=-90.034134523461546,
                    z=294.29301759794976, o_x=-0.99527936854834231,
                    o_y=0.08567952441382401, o_z=0.045585059375129855,
                    theta=-178.77187102202018)


async def return_default(arm, motion, config):
    """Planned homing; errors/cancellation stop the arm and prevent rearming."""
    print('HOMING: moving to default pose; wait for READY.', flush=True)
    try:
        ok, why = await move_flange(motion, dict(config, move_timeout_s=15.),
                                   Pose(**DEFAULT_POSE))
        if not ok:
            raise RuntimeError(f'Default pose move failed: {why}')
        position, rotation = await read_flange(motion, config)
        expected, expected_rotation = pose_matrix(Pose(**DEFAULT_POSE))
        error = float(np.linalg.norm(position-expected))
        if error > 3. or np.linalg.norm(rotation-expected_rotation) > .03:
            raise RuntimeError(f'Default pose not reached: position error {error:.1f}mm')
        print(f'HOME OK: position error {error:.1f}mm', flush=True)
        return position, rotation
    except BaseException:
        await asyncio.shield(asyncio.wait_for(arm.stop(timeout=3), 3))
        raise


def green_ball(bgr, config, previous=None):
    """The pickleball's own hue, which is not what 'green' usually means.

    Measured on this ball: H=31 lit from above, H=26 backlit -- the hue moves
    with the light, and the conventional green band (35-85) misses it entirely.

    Warm background (wood ceiling, window glare) also passes the hue test, but
    only as small specks: at 2 m this 74 mm ball projects to ~11 px, so
    `min_radius_px` rejects them before they can open a gap in the fit. Where
    several survive, the one nearest last frame's ball wins, because a distant
    speck must not steal the track from the ball.

    Returns candidates as (centre, equivalent_radius_px, enclosing_radius_px),
    best first. Range is NOT checked here: a caller that picked one blob and
    only then rejected its range would keep re-picking the same background
    object every frame and never see the ball behind it.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (config['hue_min'], config['sat_min'], config['val_min']),
                       (config['hue_max'], 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config['min_area_px']:
            continue
        moments = cv2.moments(contour)
        if moments['m00'] <= 0:
            continue
        _, enclosing = cv2.minEnclosingCircle(contour)
        if enclosing < config['min_radius_px']:
            continue
        centre = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
        found.append((centre, float(np.sqrt(area/np.pi)), float(enclosing)))
    if previous is None:
        return sorted(found, key=lambda c: -c[2])
    # Order by continuity first, then by size. Membership is tested on indices:
    # these tuples hold numpy arrays, and `in` on them raises rather than compares.
    gate = float(config['track_gate_px'])
    near = sorted((i for i, c in enumerate(found)
                   if np.linalg.norm(c[0]-previous) <= gate),
                  key=lambda i: np.linalg.norm(found[i][0]-previous))
    seen = set(near)
    far = sorted((i for i in range(len(found)) if i not in seen), key=lambda i: -found[i][2])
    return [found[i] for i in near+far]


def ball_point(centre, radius_px, enclosing_px, origin, rotation, intrinsics, config, depth=None):
    """World position of the ball centre.

    Range comes from depth when the stream is aligned to colour, because a
    projected radius is only as good as the mask's edge -- measured here, the
    equivalent-area and enclosing radii of the same blob disagreed by 32%, which
    is 180 mm of range. `--range size` falls back to the radius when depth is
    unusable, and the two are compared every frame so the disagreement is visible.
    """
    ray = np.array([(centre[0]-intrinsics.cx)/intrinsics.fx,
                    (centre[1]-intrinsics.cy)/intrinsics.fy, 1.])
    ray = ray/np.linalg.norm(ray)
    # The saturation mask stops short of the ball's darker rim, so the equivalent
    # -area radius understates it; measured against aligned depth, the enclosing
    # circle was the closer of the two (25.7 px vs 23.8 px for a true 27.9 px).
    by_size = (intrinsics.fx+intrinsics.fy)*.5*config['ball_radius_mm']/max(enclosing_px, 1e-6)
    by_depth = None
    if depth is not None:
        u, v = int(round(centre[0])), int(round(centre[1]))
        window = max(2, int(radius_px*.4))
        patch = depth[max(0, v-window):v+window+1, max(0, u-window):u+window+1]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size >= 4:
            surface = float(np.percentile(valid, 25))
            by_depth = surface/ray[2]+config['ball_radius_mm']
    chosen = by_size if (by_depth is None or config['range_source'] == 'size') else by_depth
    if not config['min_distance_mm'] <= chosen <= config['max_distance_mm']:
        return None, by_size, by_depth, f'range {chosen:.0f} mm outside limits'
    # Two independent estimates of the same quantity. On the real ball they
    # tracked within ~15%; on the background objects that kept hijacking the
    # fit they differed by 2-10x. Requiring agreement rejects those outright,
    # which no single-measurement test could do.
    if by_depth is not None:
        disagreement = abs(by_size-by_depth)/max(by_depth, 1.)
        if disagreement > config['max_range_disagreement']:
            return (None, by_size, by_depth,
                    f'size {by_size:.0f} vs depth {by_depth:.0f} mm disagree by '
                    f'{disagreement*100:.0f}%')
    point = origin+rotation@(ray*chosen)
    lower, upper = np.asarray(config['throw_volume_mm'], float)
    if not ((point >= lower).all() and (point <= upper).all()):
        return (None, by_size, by_depth,
                f'{np.round(point, 0).tolist()} mm is outside the throw volume')
    return point, by_size, by_depth, None


async def read_pose_of(motion, component, config):
    timeout = config['rpc_timeout_s']
    pose = await asyncio.wait_for(motion.get_pose(component, config['world_frame'],
                                                  timeout=timeout), timeout)
    if pose.reference_frame != config['world_frame']:
        raise ValueError(f'Unexpected reference frame for {component}')
    return pose_matrix(pose.pose)


async def read_flange(motion, config):
    return await read_pose_of(motion, config['arm'], config)


def bowl_offset(flange_position, gripper_position, gripper_rotation, config):
    """Flange-to-bowl-mouth vector in world coordinates.

    Anchored on the machine's configured gripper frame, not on a guess measured
    off the flange. The bowl sits on the gripper, so only the short residual
    from the gripper origin to the mouth centre is estimated here; the 105 mm
    flange-to-gripper part comes from the frame system and is already calibrated.
    """
    mouth = (np.asarray(gripper_position, float)
             + np.asarray(gripper_rotation, float)
             @ np.asarray(config['bowl_offset_gripper_mm'], float))
    return mouth-np.asarray(flange_position, float)


def planar_target(point, park_rotation, offset, config):
    """Flange pose that puts the bowl mouth on `point` in the catch plane."""
    flange = np.array([point[0]-offset[0], point[1]-offset[1],
                       float(config['catch_plane_z_mm'])-offset[2]])
    return flange, matrix_pose(flange, park_rotation)


async def move_flange(mover, config, pose):
    """Command the move and report whether it completed, never raising.

    The deadline must not be the time left before the ball arrives. A move that
    runs long is a missed catch, not a failure of the program; using the
    remaining flight time as the RPC deadline turned one late move into
    'Deadline exceeded' and ended the whole session. Lateness is judged
    separately, against the predicted arrival.
    """
    timeout = float(config['move_timeout_s'])
    destination = PoseInFrame(reference_frame=config['world_frame'], pose=pose)
    try:
        done = await asyncio.wait_for(mover.move(component_name=config['arm'],
                                                 destination=destination, timeout=timeout), timeout)
        return bool(done), None
    except asyncio.CancelledError:
        raise
    except Exception as error:
        return False, str(error)[:120]


class ServoMove:
    """arm.move_to_position, which is what the measured timing was taken with.

    The driver refuses long moves ("Linear speed exceeds limit in ServoJ mode"),
    so this is only safe for the short corrections this program makes; parking
    is done separately through the motion service.
    """
    def __init__(self, arm):
        self.arm = arm

    async def move(self, *, component_name, destination, timeout):
        await asyncio.wait_for(self.arm.move_to_position(pose=destination.pose, timeout=timeout),
                               timeout)
        return True


async def run(args):
    config = json.loads(args.config.read_text())
    config['rough_attempt'] = bool(getattr(args, 'rough', False))
    if config['rough_attempt']:
        if args.trajectory_source != 'wrist':
            raise ValueError('--rough currently requires --trajectory-source wrist')
        config.update(stability_samples=1, stability_span_s=0., commit_slack_s=0.,
                      catch_plane_z_mm='parked')
        print('ROUGH: continuous; cam2 approach gate; 150mm XY from session start; may arrive late.', flush=True)
    if args.no_side and args.trajectory_source != 'wrist':
        raise ValueError('--no-side requires --trajectory-source wrist')
    if args.execute:
        for proc in Path('/proc').glob('[0-9]*'):
            if proc.name == str(os.getpid()):
                continue
            try:
                command = (proc/'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            controllers = (b'motion.capture_can', b'motion.catch_plane', b'motion.catch_throw')
            if b'--execute' in command and any(name in command for name in controllers):
                raise ValueError(f'Another motion controller is executing (PID {proc.name}); '
                                 'stop it before starting a catch controller')
    if args.range_source:
        config['range_source'] = args.range_source
    timing = ArmTiming(config['arm_latency_s'], config['arm_speed_mm_s'],
                       config.get('arm_acceleration_mm_s2'))
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        arm = Arm.from_robot(robot, config['arm'])
        cam = Camera.from_robot(robot, config['camera'])
        motion = MotionClient.from_robot(robot, config['motion'])
        side = None if args.no_side else await prepare_side(
            robot, args, config, green_ball, ball_point)
        stereo = StereoTracker(config) if args.trajectory_source == 'stereo' else None
        if stereo is not None:
            side.stereo = stereo
        if await arm.is_moving(timeout=config['rpc_timeout_s']):
            raise ValueError('Start with the arm parked and stationary')
        if config['rough_attempt'] and args.execute:
            await return_default(arm, motion, config)
        before = np.array((await arm.get_joint_positions(timeout=config['rpc_timeout_s'])).values)
        park_position, park_rotation = await read_flange(motion, config)
        origin, rotation = await camera_transform(robot, config['camera'], config['world_frame'])
        p = (await cam.get_properties(timeout=config['rpc_timeout_s'])).intrinsic_parameters
        intrinsics = SimpleNamespace(fx=p.focal_x_px, fy=p.focal_y_px, cx=p.center_x_px,
                                     cy=p.center_y_px, width=p.width_px, height=p.height_px)
        after = np.array((await arm.get_joint_positions(timeout=config['rpc_timeout_s'])).values)
        if before.shape != after.shape or np.max(abs(after-before)) > .1:
            raise ValueError('Arm moved during startup; it must be parked')

        gripper_position, gripper_rotation = await read_pose_of(
            motion, config['gripper'], config)
        offset = bowl_offset(park_position, gripper_position, gripper_rotation, config)
        bowl_axis = park_rotation@np.asarray(config['bowl_axis_flange'], float)
        bowl_axis = bowl_axis/np.linalg.norm(bowl_axis)
        mouth = park_position+offset
        print(f'Gripper frame {gripper_position.round(1).tolist()} mm (from the machine config); '
              f'bowl mouth taken as gripper + {config["bowl_offset_gripper_mm"]} mm', flush=True)
        tilt = float(np.degrees(np.arccos(np.clip(bowl_axis@[0, 0, 1], -1, 1))))
        camera_elevation = float(np.degrees(np.arcsin(np.clip(rotation[:, 2][2], -1, 1))))
        print(f'Parked flange {park_position.round(1).tolist()} mm | bowl mouth {mouth.round(1).tolist()} mm',
              flush=True)
        print(f'Bowl tilted {tilt:.0f} deg from vertical | camera elevation {camera_elevation:+.0f} deg',
              flush=True)
        if tilt > config['max_bowl_tilt_deg']:
            print(f'WARNING: the bowl is {tilt:.0f} deg off vertical; a ball may not stay in it. '
                  'Re-aim the wrist if it spills.', flush=True)
        if camera_elevation > 80:
            print('WARNING: the camera is pointed nearly straight up and will not see the throw. '
                  'The bowl and camera axes are only 1.8 deg apart, so tilting one tilts the other.',
                  flush=True)
        if config['catch_plane_z_mm'] == 'parked':
            # Catch at exactly the height the bowl already sits at, so the flange
            # never changes Z and any error in the measured bowl offset cancels
            # instead of biasing the crossing time.
            config['catch_plane_z_mm'] = float(mouth[2])
            print(f'Catch plane taken from the parked bowl: z={mouth[2]:.0f} mm', flush=True)
        offset_from_plane = float(mouth[2])-float(config['catch_plane_z_mm'])
        print(f'Catch plane z={config["catch_plane_z_mm"]:.0f} mm | parked mouth height '
              f'{mouth[2]:.0f} mm (flange must rise {-offset_from_plane:+.0f} mm) | '
              + (f'timing {timing.latency_s:.3f}s + 2*sqrt(d/{timing.acceleration_mm_s2:.0f})'
                 if timing.acceleration_mm_s2 else
                 f'timing {timing.latency_s:.3f}s + d/{timing.speed_mm_s:.0f}'), flush=True)
        print('READY: lob the ball. ' + ('ARM ENABLED; waits for an eligible target.' if args.execute else
                                         'PREVIEW: nothing will move.'), flush=True)

        # A long window silently blends the pre-throw hold into the fit: the ball
        # sits still for tenths of a second, which no parabola can reconcile with
        # the flight that follows, so every fit is thrown out on its residual.
        fit = BallisticFit(min_samples=config['min_samples'], max_gap_s=config['max_gap_s'],
                           max_samples=config['max_samples'],
                           max_residual_mm=config['max_residual_mm'],
                           release_speed_mm_s=config['release_speed_mm_s'],
                           min_span_s=config['min_span_s'])
        gate = PredictionGate(tolerance_mm=config['stability_mm'],
                              arrival_spread_s=config['stability_arrival_s'],
                              min_samples=config['stability_samples'],
                              min_span_s=config['stability_span_s'])
        mover = ServoMove(arm) if not args.planned else motion
        committed = None
        rough_cycle = RoughCycle()
        rough_handoff = RoughHandoff()
        config['rough_anchor_mm'] = park_position.tolist()
        previous_pixel = None
        last_stamp = None
        started = last_report = time.monotonic()
        last_print = 0.
        frames = seen = 0
        side_task = asyncio.create_task(side.run()) if side is not None else None
        side_sequence = -1
        wrist_status = 'waiting for first frame'
        wrist_flight = None
        wrist_timestamp = None
        preview_ready = False
        print(f'Catch trajectory source: {args.trajectory_source.upper()} | '
              'SIDE and WRIST fits remain independent.', flush=True)
        try:
            while not args.duration or time.monotonic()-started < args.duration:
                ready_now = False
                try:
                    images, meta = await cam.get_images(timeout=.3)
                except Exception as error:
                    images, meta = [], None
                    wrist_status = f'WAIT: {type(error).__name__}: {str(error)[:120]}'
                now = time.monotonic()
                stamp = (meta.captured_at.seconds+meta.captured_at.nanos/1e9
                         if meta is not None else 0.)
                age = time.time()-stamp
                wrist_updated = False
                if not 0 <= age <= config['max_frame_age_s']:
                    wrist_status = 'WAIT: stale/missing wrist frame'
                    fit.reset()
                    previous_pixel = None
                    wrist_flight = None
                elif last_stamp is None or stamp > last_stamp:
                    wrist_updated = True
                    wrist_timestamp = now-age
                    last_stamp = stamp
                    frames += 1
                    sources = {im.name: im for im in images}
                    bgr = decode_color(sources[args.color_source])
                    depth = None
                    if config['range_source'] != 'size' and args.depth_source in sources:
                        depth = np.asarray(sources[args.depth_source].bytes_to_depth_array(),
                                           dtype=float)
                    # Walk the candidates in preference order and take the first
                    # whose range is usable, so a well-placed background blob
                    # cannot mask the ball.
                    # The camera rides on the wrist. Orientation is held for the whole
                    # attempt, so it only ever translates with the flange -- correcting
                    # the startup origin by that displacement keeps post-commit frames
                    # usable, where the stale transform would have been off by the
                    # whole move.
                    live_origin = origin
                    if committed is not None:
                        moved_flange, _ = await read_flange(motion, config)
                        live_origin = origin+(moved_flange-park_position)
                    point = None
                    rejected = None
                    wrist_candidates = green_ball(bgr, config, previous_pixel)
                    if stereo is not None:
                        stereo.add_wrist(wrist_timestamp, wrist_candidates, live_origin,
                                         rotation, intrinsics)
                    for centre, radius_px, enclosing in wrist_candidates:
                        point, by_size, by_depth, why = ball_point(
                            centre, radius_px, enclosing, live_origin, rotation, intrinsics,
                            config, depth)
                        if point is not None:
                            previous_pixel = centre
                            break
                        rejected = rejected or why
                    if point is None:
                        previous_pixel = None
                        wrist_status = rejected or 'no ball'
                        fit.reset()
                        wrist_flight = None
                    else:
                        seen += 1
                        disagree = ('' if by_depth is None
                                    else f' | size {by_size:.0f} vs depth {by_depth:.0f} mm')
                        wrist_status = f'BALL {np.round(point, 0).tolist()}{disagree}'
                        wrist_flight = fit.add(now-age, point)
                        if wrist_flight is not None:
                            wrist_status += (f' | fit v={np.linalg.norm(wrist_flight.velocity):.0f} '
                                             f'mm/s n={wrist_flight.samples} '
                                             f'r={wrist_flight.residual_mm:.0f}')
                            wt, why = plane_intercept(
                                wrist_flight, park_position, offset, config, timing, now)
                            wrist_status += (f' | cross {wt.point.round(0).tolist()} '
                                             f'in {wt.arrival-now:.2f}s' if wt else f' | {why}')
                # Side runs independently; a missed/rejected wrist observation must
                # neither erase its fit nor prevent a new side prediction.
                now = time.monotonic()
                if side_task is not None and side_task.done():
                    await side_task
                if stereo is not None:
                    stereo.update(now)
                    flight = stereo.current(now)
                    updated = stereo.sequence != side_sequence
                    side_sequence = stereo.sequence
                elif args.trajectory_source == 'side':
                    flight = side.current(now)
                    updated = side.sequence != side_sequence
                    side_sequence = side.sequence
                else:
                    flight, updated = wrist_flight, wrist_updated
                    if wrist_timestamp is None or now-wrist_timestamp > config['max_frame_age_s']:
                        flight = None
                status = f'WRIST: {wrist_status}'
                if side is not None:
                    side_status = side.status
                    if side.current(now) is None and side.flight is not None:
                        side_status = 'WAIT: side prediction expired'
                    status += f' || SIDE: {side_status}'
                if stereo is not None:
                    status += f' || STEREO: {stereo.status}'
                status += f' || CATCH[{args.trajectory_source}]'
                if config['rough_attempt']:
                    approach_flight = side.current(now) if side is not None else flight
                    approaching = approaching_bowl(approach_flight, now, park_position+offset)
                    if not rough_cycle.ready:
                        rough_handoff.clear()
                    else:
                        if wrist_updated and wrist_flight is not None:
                            candidate, _ = plane_intercept(
                                wrist_flight, park_position, offset, config, timing, now)
                            rough_handoff.observe_wrist(
                                wrist_timestamp, wrist_flight if candidate is not None else None)
                        if approaching:
                            rough_handoff.observe_approach(
                                side.timestamp if side is not None else wrist_timestamp)
                    allowed = rough_cycle.update(
                        now, rough_handoff.approach_recent(now) if rough_cycle.ready else approaching)
                    # Both exposure times remain bounded even when neither camera
                    # supplies a new frame on this iteration. Intercept is rechecked below.
                    flight = rough_handoff.candidate(now) if allowed else None
                    if flight is not None:
                        updated = True
                        status += ' | ROUGH: recent wrist + cam2 evidence matched (<=0.25s)'
                    elif not rough_cycle.ready:
                        status += ' | ROUGH: cooldown / waiting for previous flight to clear'
                    elif rough_handoff.approach_recent(now):
                        status += ' | ROUGH: cam2 accepted; waiting for recent valid wrist prediction'
                    else:
                        status += ' | ROUGH: armed; waiting for recent cam2 approach'
                if flight is None:
                    gate.reset()
                else:
                    target, refusal = plane_intercept(
                        flight, park_position, offset, config, timing, now)
                    if target is None:
                        status += f' | {refusal}'
                        gate.reset()
                    else:
                        status += (f' | cross ({target.point[0]:.0f}, '
                                   f'{target.point[1]:.0f}) in {target.arrival-now:.2f}s '
                                   f'slack {target.slack_s:.2f}s')
                        stable = gate.add(now, target.point, target.arrival) if updated else False
                        # Hold off while there is time to spare: every extra
                        # frame shortens the extrapolation, and committing at
                        # the first stable prediction throws that away.
                        if (stable and config['commit_slack_s'] > 0
                            and target.slack_s > config['commit_slack_s']):
                            stable = False
                            status += f' | holding, {target.slack_s:.2f}s spare'
                        if not stable:
                            have, need = gate.progress()
                            status += f' | settling {have}/{need}'
                        elif not args.execute:
                            status += ' | WOULD CATCH'
                            ready_now = True
                        if not args.execute:
                            pass
                        elif committed is None and stable:
                            # Aim at the consensus of recent predictions, not at
                            # whichever frame happened to be last.
                            aim = gate.consensus()
                            flange, pose = planar_target(aim, park_rotation,
                                                         offset, config)
                            if not reachable(flange, config):
                                status += ' | target flange unreachable'
                            else:
                                print(status, flush=True)
                                print(f'COMMIT: bowl -> ({aim[0]:.0f}, '
                                      f'{aim[1]:.0f}) mm, ball in '
                                      f'{target.arrival-now:.2f}s', flush=True)
                                print(f'MOVE REQUEST: flange -> ({flange[0]:.1f}, '
                                      f'{flange[1]:.1f}, {flange[2]:.1f}) mm; '
                                      f'XY travel {np.linalg.norm((flange-park_position)[:2]):.1f}mm; '
                                      f'estimated slack {target.slack_s:+.3f}s', flush=True)
                                ok, why = await move_flange(mover, config, pose)
                                committed = (target.arrival, np.array(aim))
                                late = time.monotonic()-target.arrival
                                print(f'ARRIVED {late:+.2f}s relative to the ball' if ok
                                      else f'MOVE FAILED: {why}', flush=True)
                                if config['rough_attempt']:
                                    actual, _ = await read_flange(motion, config)
                                    print(f'MOVE RESULT: success={ok}; measured displacement '
                                          f'{np.linalg.norm(actual-park_position):.1f}mm; '
                                          f'target error {np.linalg.norm(actual-flange):.1f}mm', flush=True)
                                    if not ok:
                                        return
                                    rough_cycle.fired(time.monotonic())
                                    rough_handoff.clear()
                                    # Stay under the predicted landing until the ball has passed.
                                    await asyncio.sleep(max(.5, target.arrival-time.monotonic()+.4))
                                    park_position, park_rotation = await return_default(arm, motion, config)
                                    origin, rotation = await camera_transform(
                                        robot, config['camera'], config['world_frame'])
                                    gripper_position, gripper_rotation = await read_pose_of(
                                        motion, config['gripper'], config)
                                    offset = bowl_offset(park_position, gripper_position, gripper_rotation, config)
                                    if side is not None:
                                        side.invalidate('reset after return to default')
                                    committed = None
                                    fit.reset()
                                    wrist_flight = None
                                    previous_pixel = None
                                    gate.reset()
                                    print('READY: back at default; waiting for next throw / cooldown.', flush=True)
                        elif committed is not None and updated:
                            drift = float(np.linalg.norm(
                                (target.point-committed[1])[:2]))
                            spare = (target.arrival-now)-timing.reach_seconds(drift) \
                                - config['arrival_margin_s']
                            if drift > config['nudge_threshold_mm'] and spare > 0:
                                flange, pose = planar_target(target.point,
                                                             park_rotation, offset, config)
                                if reachable(flange, config):
                                    print(f'NUDGE: {drift:.0f} mm, {spare:.2f}s spare',
                                          flush=True)
                                    await move_flange(mover, config, pose)
                                    committed = (target.arrival, np.array(target.point))
                if committed is not None and time.monotonic() > committed[0]+.3:
                    print('Ball should have arrived. No sensor here can confirm a catch; look in '
                          'the bowl.', flush=True)
                    return
                if status and (now-last_print >= .15 or (ready_now and not preview_ready)):
                    print(status, flush=True)
                    last_print = now
                preview_ready = ready_now
                if now-last_report >= 2:
                    elapsed = now-last_report
                    rates = f'WRIST frames {frames/elapsed:.1f} Hz | ball {seen/elapsed:.1f} Hz'
                    if side is not None:
                        rates += (f' || SIDE frames {side.frames/elapsed:.1f} Hz | '
                                  f'ball {side.seen/elapsed:.1f} Hz')
                        side.frames = side.seen = 0
                    print(rates, flush=True)
                    frames = seen = 0
                    last_report = now
                await asyncio.sleep(.002)
        finally:
            if side_task is not None:
                side_task.cancel()
                await asyncio.gather(side_task, return_exceptions=True)
            if args.execute:
                try:
                    await asyncio.shield(asyncio.wait_for(arm.stop(timeout=3), 3))
                except Exception as error:
                    print(f'STOP FAILED: {error}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parents[1]/'catch_plane.config.json')
    parser.add_argument('--machine-config', type=Path)
    parser.add_argument('--rough', action='store_true',
                        help='Continuous wrist catches within 150mm of session start; cam2 approach gate')
    parser.add_argument('--execute', action='store_true', help='Command the arm')
    parser.add_argument('--trajectory-source', choices=('side', 'wrist', 'stereo'), default='side',
                        help='Which independent world trajectory supplies the catch target')
    parser.add_argument('--side-camera', default='cam2')
    parser.add_argument('--side-calibration', type=Path,
                        help='Override side_calibration in the catch config')
    parser.add_argument('--no-side', action='store_true', help='Disable side camera (use wrist source)')
    parser.add_argument('--planned', action='store_true',
                        help='Route moves through the builtin motion service instead of ServoJ')
    parser.add_argument('--range-source', choices=('depth', 'size'),
                        help='Override how the ball range is estimated')
    parser.add_argument('--color-source', default='color')
    parser.add_argument('--depth-source', default='depth')
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
