"""Visual-servo the wrist camera onto a handheld red ball; --execute moves the arm.

Image-space control. No table plane and no depth-to-color alignment assumption,
so the ball may be held anywhere in view rather than resting on the table.
Distance comes from the ball's projected radius, so the measured `ball_radius_mm`
sets the loop gain. Flange orientation is held fixed; only the position changes.

This is a proportional follower, not a catcher: it never closes the gripper.
"""
import argparse
import asyncio
import json
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

from motion.trajectory_local import credentials, decode_color, positive, red_candidates
from motion.can_tracking import red_can_candidates
from motion.rolling_preview import camera_transform
from motion.live_camera_pose import PoseBuffer, sample_camera, pose_matrix, matrix_pose


def nonnegative(value):
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError('must be finite and not negative')
    return result


def camera_tilt_deg(camera_rotation):
    """Angle of the camera's forward axis above/below horizontal, signed upward."""
    forward = camera_rotation[:, 2]
    return float(np.degrees(np.arcsin(np.clip(forward[2]/np.linalg.norm(forward), -1, 1))))


def red_blob_candidates(bgr, config):
    """Permissive red-blob detector for following: colour and size only.

    A follower needs the direction to the object, not its shape, so this keeps
    blobs the ball/can detectors reject: motion-blurred (lower saturation),
    partly occluded by a hand (not circular), and touching the image border.
    Returns (centre_px, equivalent_radius_px, truncated). `truncated` marks a
    blob on the border, whose projected size understates its true range.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation, value = config['min_saturation'], config['min_value']
    mask = cv2.inRange(hsv, (0, saturation, value), (config['hue_low_max'], 255, 255))
    mask |= cv2.inRange(hsv, (config['hue_high_min'], saturation, value), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Close the white lettering and specular highlights that split one object.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = bgr.shape[:2]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if area < config['min_area_px'] or moments['m00'] <= 0:
            continue
        center = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
        x, y, w, h = cv2.boundingRect(contour)
        truncated = bool(x <= 1 or y <= 1 or x+w >= width-1 or y+h >= height-1)
        candidates.append((center, float(np.sqrt(area/np.pi)), truncated))
    return candidates


def locate_target(bgr, config, previous=None):
    """Largest clearly dominant red target, so a second red object cannot flip the loop.

    Returns (centre_px, size_px, physical_mm, None) or (None, None, None, reason).
    `size_px` and `physical_mm` are the same kind of measurement for the configured
    shape: ball radius against ball radius, can width against can diameter.
    """
    detector = config.get('detector', config.get('object_type', 'ball'))
    if detector == 'blob':
        found = red_blob_candidates(bgr, config)
        if not found:
            return None, None, None, 'NO TARGET: no red blob of usable size'
        # A permissive mask finds several reds, so ambiguity is resolved by
        # continuity rather than rejected: stay on the blob nearest the one
        # followed last frame, and fall back to the largest without history.
        near = ([c for c in found if np.linalg.norm(np.asarray(c[0])-previous) <= config['track_gate_px']]
                if previous is not None else [])
        if near:
            center, radius, truncated = min(near, key=lambda c: np.linalg.norm(np.asarray(c[0])-previous))
        else:
            center, radius, truncated = max(found, key=lambda c: c[1])
        # A truncated blob measures smaller than it is, so its range is unusable.
        return np.asarray(center, float), float(radius), (None if truncated else float(config['ball_radius_mm'])), None
    if detector == 'can':
        # A can is not circular in the image; its width is the diameter at any roll angle.
        found = [(c.center_px, c.short_side_px) for c in red_can_candidates(bgr)]
        physical, label = config['can_diameter_mm'], 'CAN'
        floor = config['min_width_px']
    else:
        found = red_candidates(bgr)
        physical, label = config['ball_radius_mm'], 'BALL'
        floor = config['min_radius_px']
    candidates = [c for c in found if c[1] >= floor]
    if not candidates:
        return None, None, None, f'NO {label}: no saturated red shape of usable size'
    candidates.sort(key=lambda c: -c[1])
    if len(candidates) > 1 and candidates[0][1] < config['dominance_ratio']*candidates[1][1]:
        return None, None, None, f'AMBIGUOUS: {len(candidates)} similar red shapes'
    return np.asarray(candidates[0][0], float), float(candidates[0][1]), float(physical), None


def estimate_distance(size_px, intrinsics, physical_mm):
    """Range from projected size. Accurate only for a fully visible, measured object."""
    if size_px <= 0:
        raise ValueError('Projected size must be positive')
    return (intrinsics.fx+intrinsics.fy)*.5*physical_mm/size_px


def clamp_target(goal, config):
    """Clip to the free-space box, then to the arm's radial reach about its base.

    The box keeps commands clear of the configured table top and front wall; the
    radial limits keep them inside the xArm850's envelope and away from the base
    singularity. Neither replaces the planner's own reachability and collision
    checks, and `--direct` bypasses those entirely.
    """
    bounds = np.array(config['workspace_mm'], float)
    goal = np.clip(goal, bounds[0], bounds[1])
    reach = float(np.linalg.norm(goal))
    if reach > config['max_reach_mm']:
        goal = goal*config['max_reach_mm']/reach
        goal = np.clip(goal, bounds[0], bounds[1])
    elif reach < config['min_reach_mm']:
        if reach < 1e-6:
            raise ValueError('Target collapsed onto the arm base')
        goal = goal*config['min_reach_mm']/reach
        goal = np.clip(goal, bounds[0], bounds[1])
    return goal


def in_workspace(position, config):
    bounds = np.array(config['workspace_mm'], float)
    reach = float(np.linalg.norm(position))
    return bool((position >= bounds[0]).all() and (position <= bounds[1]).all()
                and config['min_reach_mm'] <= reach <= config['max_reach_mm'])


def servo_step(center, distance, flange, camera_rotation, intrinsics, config):
    """Flange translation that drives the ball toward the image centre and standoff.

    `flange` is (position, rotation) in world. Returns (Pose, telemetry) or
    (None, reason). The flange rotation is passed through unchanged, so the
    camera keeps looking the way the operator aimed it.
    """
    position, rotation = flange
    if not config['min_distance_mm'] <= distance <= config['max_distance_mm']:
        return None, f'OUT OF RANGE: target at {distance:.0f} mm'
    error = np.array([center[0]-intrinsics.cx, center[1]-intrinsics.cy], float)
    # Camera optical axes are X right, Y down, Z forward: translate the camera
    # toward the ball's off-centre offset to null it.
    lateral = error*distance/np.array([intrinsics.fx, intrinsics.fy])
    forward = distance-config['standoff_mm']
    if abs(forward) < config['standoff_deadband_mm']:
        forward = 0.
    delta = camera_rotation@np.array([
        lateral[0]*config['lateral_gain'],
        lateral[1]*config['lateral_gain'],
        forward*config['forward_gain']])
    length = float(np.linalg.norm(delta))
    if length > config['max_step_mm']:
        delta = delta*config['max_step_mm']/length
    goal = clamp_target(position+delta, config)
    moved = float(np.linalg.norm(goal-position))
    telemetry = SimpleNamespace(distance=distance, error=error, step=moved)
    if moved < config['min_step_mm']:
        return None, f'HOLD: centred within {np.abs(error).max():.0f} px, step {moved:.1f} mm'
    return matrix_pose(goal, rotation), telemetry


class SmoothMove:
    """Back-to-back moves with no mid-flight cancellation.

    Replacing an in-flight move means stopping the arm first, and a hard stop
    every command is what makes following visibly jerky. This instead lets each
    move finish and immediately commands the newest target, so the arm only
    decelerates at waypoints. `preempt_mm` (0 disables) re-aims mid-flight when
    the target has moved so far that finishing the current move is pointless.

    Planner failures are counted, not raised: an unreachable target is normal
    while following a hand held anywhere in view.
    """
    def __init__(self, arm, motion, arm_name, world, *, preempt_mm=0., timeout=5.):
        self.arm = arm
        self.motion = motion
        self.arm_name = arm_name
        self.world = world
        self.preempt_mm = preempt_mm
        self.timeout = timeout
        self.task = None
        self.target = None
        self.pending = None
        self.commanded = False
        self.moves = 0
        self.failures = 0
        self.last_error = None

    async def settle(self):
        """Consume a finished move without disturbing one still in flight."""
        if self.task is None or not self.task.done():
            return
        task, self.task = self.task, None
        try:
            if not bool(await task):
                raise RuntimeError('Planner failed the follow move')
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.failures += 1
            self.last_error = error

    async def update(self, pose):
        """Record the newest target and command it as soon as the arm is free."""
        await self.settle()
        self.pending = pose
        if self.task is not None:
            if not self.preempt_mm:
                return False
            moved = float(np.linalg.norm(np.array([pose.x, pose.y, pose.z])-self.target))
            if moved < self.preempt_mm:
                return False
            await self.abort()
        return await self.submit()

    async def submit(self):
        if self.pending is None:
            return False
        pose, self.pending = self.pending, None
        self.target = np.array([pose.x, pose.y, pose.z], float)
        self.commanded = True
        self.moves += 1
        destination = PoseInFrame(reference_frame=self.world, pose=pose)
        self.task = asyncio.create_task(asyncio.wait_for(self.motion.move(
            component_name=self.arm_name, destination=destination, timeout=self.timeout), self.timeout))
        return True

    async def abort(self):
        task, self.task = self.task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(self.arm.stop(timeout=3), 3)

    async def stop(self):
        if self.task is not None:
            await self.abort()
        elif self.commanded:
            await asyncio.wait_for(self.arm.stop(timeout=3), 3)
        self.commanded = False
        self.pending = None
        self.target = None


class DirectArmMove:
    """`arm.move_to_position` adapter with the MotionClient.move signature.

    Skips the builtin motion service's obstacle-aware planning, so configured
    table/wall geometry is NOT enforced. Lower latency, less protection.
    """
    def __init__(self, arm):
        self.arm = arm

    async def move(self, *, component_name, destination, timeout):
        await asyncio.wait_for(self.arm.move_to_position(pose=destination.pose, timeout=timeout), timeout)
        return True


async def read_flange(motion, config):
    timeout = config['rpc_timeout_s']
    pose = await asyncio.wait_for(motion.get_pose(config['arm'], config['world_frame'], timeout=timeout), timeout)
    if pose.reference_frame != config['world_frame']:
        raise ValueError('Unexpected flange reference frame')
    return pose_matrix(pose.pose)


async def run(args):
    config = json.loads(args.config.read_text())
    for name in ('lateral_gain', 'forward_gain', 'max_step_mm', 'standoff_mm',
                 'preempt_mm', 'min_saturation', 'min_area_px'):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    print('TRACKER track-red-v1: image-space follower; no gripper commands.', flush=True)
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    async with await RobotClient.at_address('127.0.0.1:8080', options) as robot:
        arm = Arm.from_robot(robot, config['arm'])
        cam = Camera.from_robot(robot, config['camera'])
        motion = MotionClient.from_robot(robot, config['motion'])
        if await arm.is_moving(timeout=config['rpc_timeout_s']):
            raise ValueError('Start with the arm stationary')
        before = np.array((await arm.get_joint_positions(timeout=config['rpc_timeout_s'])).values)
        flange = await read_flange(motion, config)
        if not in_workspace(flange[0], config):
            raise ValueError(f'Flange at {flange[0].round(1).tolist()} mm (reach '
                             f'{np.linalg.norm(flange[0]):.0f} mm) is outside workspace_mm '
                             f'{config["workspace_mm"]} / reach {config["min_reach_mm"]}-'
                             f'{config["max_reach_mm"]} mm; move it into range first')
        # Verify the frame-system camera transform while stationary, before it is
        # trusted on moving-camera feedback.
        origin, rotation = await camera_transform(robot, config['camera'], config['world_frame'])
        properties = await cam.get_properties(timeout=config['rpc_timeout_s'])
        p = properties.intrinsic_parameters
        intrinsics = SimpleNamespace(fx=p.focal_x_px, fy=p.focal_y_px, cx=p.center_x_px,
                                     cy=p.center_y_px, width=p.width_px, height=p.height_px)
        if min(intrinsics.fx, intrinsics.fy, intrinsics.width, intrinsics.height) <= 0:
            raise ValueError('Invalid camera intrinsics')
        after = np.array((await arm.get_joint_positions(timeout=config['rpc_timeout_s'])).values)
        if before.shape != after.shape or not before.size or np.max(abs(after-before)) > .1:
            raise ValueError('Arm moved during startup calibration')
        tilt = camera_tilt_deg(rotation)
        print(f'Camera at {origin.round(1).tolist()} mm | flange at {flange[0].round(1).tolist()} mm', flush=True)
        print(f'Camera forward {rotation[:, 2].round(3).tolist()} | tilt {tilt:+.1f} deg from horizontal '
              f'({"level" if abs(tilt) <= 5 else "NOT LEVEL: re-aim in manual mode if you want it level"})', flush=True)
        print(f'Orientation is held fixed: the camera keeps this exact aim while following.', flush=True)
        print(f'Gain reference: {config.get("object_type","ball")} size {config["ball_radius_mm"] if config.get("object_type","ball")=="ball" else config["can_diameter_mm"]:.0f} mm '
              f'- measure yours and set it, it scales every step.', flush=True)
        if not config['forward_gain']:
            print('Standoff control OFF: the arm tracks up/down and side to side only, never toward you.', flush=True)
        print(f'READY: hold the red {config.get("object_type","ball")} in view. '
              + ('ARM WILL FOLLOW.' if args.execute else 'PREVIEW: no arm commands.'), flush=True)
        if args.execute and not args.planned:
            print('Planner SKIPPED (default): arm.move_to_position, lower latency. Targets are held '
                  'inside workspace_mm and the reach limits; nothing else is checked. '
                  'Use --planned for obstacle-aware planning.', flush=True)
        print(f'Free space: X {config["workspace_mm"][0][0]}..{config["workspace_mm"][1][0]}, '
              f'Y {config["workspace_mm"][0][1]}..{config["workspace_mm"][1][1]}, '
              f'Z {config["workspace_mm"][0][2]}..{config["workspace_mm"][1][2]} mm | reach '
              f'{config["min_reach_mm"]}..{config["max_reach_mm"]} mm', flush=True)
        preempt = config['preempt_mm']
        print(f'Smoothness: step <= {config["max_step_mm"]:.0f} mm, moves run back to back, '
              + (f're-aiming only past {preempt:.0f} mm.' if preempt else 'never cancelled mid-flight.'),
              flush=True)

        buffer = PoseBuffer()
        sampler = asyncio.create_task(sample_camera(robot, config['camera'], config['world_frame'], buffer))
        previous = None
        mover = SmoothMove(arm, motion if args.planned else DirectArmMove(arm),
                           config['arm'], config['world_frame'],
                           preempt_mm=config['preempt_mm'], timeout=config['move_timeout_s'])
        distance = None
        last_timestamp = None
        last_seen = None
        started = last_report = time.monotonic()
        last_print = 0.
        frames = detections = commands = 0
        try:
            while not args.duration or time.monotonic()-started < args.duration:
                if sampler.done():
                    await sampler
                if args.execute:
                    await mover.settle()
                images, meta = await cam.get_images(timeout=.3)
                now = time.monotonic()
                stamp = meta.captured_at.seconds+meta.captured_at.nanos/1e9
                status = None
                if not 0 <= time.time()-stamp <= .12:
                    status = 'WAIT: camera frame stale'
                elif last_timestamp is None or stamp > last_timestamp:
                    last_timestamp = stamp
                    frames += 1
                    bgr = decode_color(next(im for im in images if im.name == args.color_source))
                    if bgr.shape[:2] != (intrinsics.height, intrinsics.width):
                        raise ValueError('Camera resolution changed')
                    center, size_px, physical_mm, reason = locate_target(bgr, config, previous)
                    previous = center
                    if center is None:
                        status = reason
                        distance = None
                    else:
                        detections += 1
                        last_seen = now
                        camera_rotation = None
                        for _ in range(5):
                            try:
                                _, camera_rotation = buffer.at(stamp)
                                break
                            except ValueError:
                                await asyncio.sleep(.005)
                        if camera_rotation is None:
                            status = 'WAIT: no fresh camera-pose bracket'
                        else:
                            flange = await read_flange(motion, config)
                            if physical_mm is None:
                                # Border-truncated blob: keep the last good range
                                # rather than letting an understated size inflate the gain.
                                measured = distance
                            else:
                                measured = estimate_distance(size_px, intrinsics, physical_mm)
                            if measured is None:
                                status = 'WAIT: target on the image border, no range yet'
                                target = info = None
                            else:
                                blend = config['distance_smoothing']
                                distance = measured if distance is None else (1-blend)*distance+blend*measured
                                target, info = servo_step(center, distance, flange,
                                                          camera_rotation, intrinsics, config)
                            if target is None and info is not None:
                                status = info
                            elif target is None:
                                pass
                            else:
                                status = (f'TARGET uv=({center[0]:.0f}, {center[1]:.0f}) {size_px:.1f}px '
                                          f'| {info.distance:.0f} mm | err=({info.error[0]:+.0f}, {info.error[1]:+.0f}) px')
                                if args.execute:
                                    if await mover.update(target):
                                        commands += 1
                                        status += f' | MOVE -> ({target.x:.0f}, {target.y:.0f}, {target.z:.0f}) mm'
                                else:
                                    status += f' | WOULD MOVE -> ({target.x:.0f}, {target.y:.0f}, {target.z:.0f}) mm'
                if last_seen is not None and now-last_seen > config['lost_timeout_s']:
                    if args.execute and mover.commanded:
                        await mover.stop()
                        print('STOP: ball lost; holding position.', flush=True)
                    last_seen = None
                if status and now-last_print >= .3:
                    print(status, flush=True)
                    last_print = now
                if now-last_report >= 2:
                    elapsed = now-last_report
                    note = f' | move failures {mover.failures}' if mover.failures else ''
                    print(f'Fresh frames {frames/elapsed:.1f} Hz | ball {detections/elapsed:.1f} Hz '
                          f'| commands {commands/elapsed:.1f} Hz{note}', flush=True)
                    frames = detections = commands = 0
                    last_report = now
                await asyncio.sleep(.002)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            if args.execute:
                await asyncio.shield(mover.stop())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[1]/'track_red.config.json')
    parser.add_argument('--machine-config', type=Path)
    parser.add_argument('--execute', action='store_true', help='Command the physical arm to follow the ball')
    parser.add_argument('--planned', action='store_true',
                        help='Route moves through the builtin motion service (obstacle-aware but slower); '
                             'the default uses arm.move_to_position for lower latency')
    parser.add_argument('--color-source', default='color')
    parser.add_argument('--lateral-gain', type=positive)
    parser.add_argument('--forward-gain', type=nonnegative,
                        help='Standoff control; 0 (default) keeps the arm from driving toward the operator')
    parser.add_argument('--max-step-mm', type=positive,
                        help='Longest single commanded step; larger means fewer, smoother sweeps')
    parser.add_argument('--preempt-mm', type=nonnegative,
                        help='Cancel an in-flight move once the target shifts this far; 0 never cancels')
    parser.add_argument('--min-saturation', type=nonnegative,
                        help='Red mask saturation floor; lower accepts duller and blurrier reds')
    parser.add_argument('--min-area-px', type=positive, help='Smallest accepted red blob area')
    parser.add_argument('--standoff-mm', type=positive)
    parser.add_argument('--duration', type=positive)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('\nTracker stopped.')
    except Exception as error:
        parser.exit(2, f'Tracker stopped: {error}\n')


if __name__ == '__main__':
    main()
