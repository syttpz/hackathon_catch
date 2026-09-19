"""Catch a ball that leaves rest using the same Viam 3D localization as stationary grab.

Watch while the wrist is still. When the ball leaves rest, fit velocity from world
samples, predict its position at catch_delay_s, then start the arm so the TCP
arrives at that pose at the same time as the ball.

Preview (default) never moves. --execute opens the gripper and intercepts.

  python -m motion.viam_ball_catch
  python -m motion.viam_ball_catch --execute
"""
import argparse
import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from connection import connect
from vision.viam_ball import BallDetection, BallPose, ball_pose_in_world

DEFAULT_CONFIG = Path(__file__).parents[1] / "ball_catch.config.json"
logger = logging.getLogger(__name__)


def _fmt(xyz):
    return f"[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm"


@dataclass(frozen=True)
class Sample:
    t: float
    xyz: tuple[float, float, float]


def _vector(values, size, name):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a nonfinite number")
    return result


def validate_config(config):
    for key in ("camera", "detector", "segmenter", "arm", "gripper", "motion", "world_frame", "ball_label"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing resource/frame name: {key}")
    for key in ("rest_speed_mm_s", "leave_rest_speed_mm_s", "leave_rest_shift_mm", "sample_period_s",
                "track_timeout_s", "catch_delay_s", "min_catch_delay_s", "max_catch_delay_s",
                "max_tcp_speed_mm_s", "arrival_tolerance_mm", "settle_s", "move_timeout_s",
                "rpc_timeout_s", "max_track_jump_mm"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("min_rest_samples", "min_motion_samples"):
        if int(config[key]) < 2:
            raise ValueError(f"{key} must be at least 2")
    if not (config["min_catch_delay_s"] <= config["catch_delay_s"] <= config["max_catch_delay_s"]):
        raise ValueError("catch_delay_s must lie between min_catch_delay_s and max_catch_delay_s")
    for key in ("grasp_z_world_mm", "expected_ball_z_world_mm"):
        if not math.isfinite(float(config[key])):
            raise ValueError(f"{key} must be finite")
    if config["grasp_z_world_mm"] <= config["expected_ball_z_world_mm"]:
        raise ValueError("Taught TCP grasp height must be above expected ball center")
    _vector(config["grasp_xy_offset_mm"], 2, "grasp_xy_offset_mm")
    _vector(config["acceleration_mm_s2"], 3, "acceleration_mm_s2")
    orientation = _vector(config["grasp_orientation"], 4, "grasp_orientation")
    if not math.isclose(math.sqrt(sum(v*v for v in orientation[:3])), 1, abs_tol=0.01):
        raise ValueError("Grasp orientation direction must be normalized")
    bounds = config["workspace_mm"]
    if len(bounds) != 2:
        raise ValueError("workspace_mm needs lower and upper XYZ bounds")
    lower, upper = [_vector(row, 3, "workspace_mm") for row in bounds]
    if any(a >= b for a, b in zip(lower, upper)):
        raise ValueError("Workspace lower bounds must be less than upper bounds")


def _xyz(pose):
    return [float(pose.x), float(pose.y), float(pose.z)]


def _in_workspace(point, config):
    point = _vector(list(point), 3, "target")
    return all(low <= value <= high for value, low, high in zip(point, *config["workspace_mm"]))


def sample_speed(samples):
    """Instantaneous speed between the last two world samples, mm/s."""
    if len(samples) < 2:
        return 0.0
    dt = samples[-1].t - samples[-2].t
    if dt <= 1e-6:
        return 0.0
    return math.dist(samples[-1].xyz, samples[-2].xyz) / dt


def at_rest(samples, config):
    needed = int(config["min_rest_samples"])
    if len(samples) < needed:
        return False
    window = samples[-needed:]
    duration = window[-1].t - window[0].t
    if duration <= 1e-6:
        return False
    travel = math.dist(window[0].xyz, window[-1].xyz)
    return travel / duration <= float(config["rest_speed_mm_s"])


def left_rest(samples, rest_xyz, config):
    if rest_xyz is None or len(samples) < 2:
        return False
    shift = math.dist(samples[-1].xyz, rest_xyz)
    return shift >= float(config["leave_rest_shift_mm"])


def fit_velocity(samples, acceleration):
    """Least-squares p,v at the last sample: p(t)=p + v t + 0.5 a t^2."""
    if len(samples) < 2:
        raise ValueError("Need at least two samples to estimate velocity")
    acceleration = _vector(acceleration, 3, "acceleration_mm_s2")
    times = [sample.t - samples[-1].t for sample in samples]
    if max(times) - min(times) < 0.04:
        raise ValueError("Need at least 40 ms of motion samples")
    count = len(times)
    sum_t = sum(times)
    sum_t2 = sum(t * t for t in times)
    det = count * sum_t2 - sum_t * sum_t
    if abs(det) < 1e-12:
        raise ValueError("Degenerate velocity fit")
    position, velocity, sq_error = [], [], 0.0
    for axis, accel in enumerate(acceleration):
        values = [sample.xyz[axis] - 0.5 * accel * t * t for sample, t in zip(samples, times)]
        sum_y = sum(values)
        sum_ty = sum(t * value for t, value in zip(times, values))
        origin = (sum_t2 * sum_y - sum_t * sum_ty) / det
        speed = (count * sum_ty - sum_t * sum_y) / det
        if not math.isfinite(origin) or not math.isfinite(speed):
            raise ValueError("Nonfinite velocity fit")
        position.append(origin)
        velocity.append(speed)
        sq_error += sum((origin + speed * t - value) ** 2 for t, value in zip(times, values))
    residual = math.sqrt(sq_error / (count * 3))
    return position, velocity, residual


def predict_position(position, velocity, acceleration, delay_s):
    position = _vector(position, 3, "position")
    velocity = _vector(velocity, 3, "velocity")
    acceleration = _vector(acceleration, 3, "acceleration_mm_s2")
    return [p + v * delay_s + 0.5 * a * delay_s * delay_s
            for p, v, a in zip(position, velocity, acceleration)]


def catch_target(predicted_ball, config):
    dx, dy = config["grasp_xy_offset_mm"]
    tcp_to_ball_z = float(config["grasp_z_world_mm"]) - float(config["expected_ball_z_world_mm"])
    target = [predicted_ball[0] + dx, predicted_ball[1] + dy, predicted_ball[2] + tcp_to_ball_z]
    if not _in_workspace(target, config):
        raise ValueError(f"Catch target {target} is outside workspace_mm")
    return target


def choose_delay(position, velocity, acceleration, config):
    """Use catch_delay_s if that intercept is reachable; otherwise the earliest valid delay."""
    preferred = float(config["catch_delay_s"])
    candidates = [preferred]
    start = float(config["min_catch_delay_s"])
    stop = float(config["max_catch_delay_s"])
    candidates.extend(start + i * (stop - start) / 10 for i in range(11))
    seen = set()
    for delay in candidates:
        delay = round(delay, 4)
        if delay in seen:
            continue
        seen.add(delay)
        predicted = predict_position(position, velocity, acceleration, delay)
        try:
            target = catch_target(predicted, config)
        except ValueError:
            continue
        return delay, predicted, target
    raise ValueError("No intercept in workspace_mm within the configured catch delay window")


def move_duration(current, target, config):
    speed = float(config["max_tcp_speed_mm_s"])
    return max(math.dist(current, target) / speed, 0.05)


def _in_xy_workspace(pose, config, pad=50):
    lo, hi = config["workspace_mm"]
    return lo[0]-pad <= pose.x <= hi[0]+pad and lo[1]-pad <= pose.y <= hi[1]+pad


def pick_tracked_ball(worlds, previous_xyz, config):
    """Prefer the cloud nearest the last ball; do not follow a larger hand/arm blob."""
    if not worlds:
        raise ValueError("Expected a ball segment; found 0")
    if previous_xyz is not None:
        chosen = min(worlds, key=lambda pose: math.dist((pose.x, pose.y, pose.z), previous_xyz))
        jump = math.dist((chosen.x, chosen.y, chosen.z), previous_xyz)
        limit = float(config["max_track_jump_mm"])
        if jump > limit:
            if len(worlds) == 1:
                logger.warning("single segment jumped %.0f mm; treating as ball motion", jump)
                return chosen
            raise ValueError(
                f"Ball jumped {jump:.0f} mm (limit {limit:.0f} mm); "
                f"from {_fmt(previous_xyz)} to {_fmt((chosen.x, chosen.y, chosen.z))}"
            )
        if len(worlds) > 1:
            logger.info(
                "found %d ball segments %s; kept nearest (jump=%.0f mm)",
                len(worlds), [_fmt((pose.x, pose.y, pose.z)) for pose in worlds], jump)
        return chosen
    in_xy = [pose for pose in worlds if _in_xy_workspace(pose, config)]
    pool = in_xy or worlds
    if len(worlds) > 1:
        logger.warning("found %d ball segments; using largest%s",
                       len(worlds), " in workspace XY" if in_xy else "")
    return max(pool, key=lambda pose: pose.point_count)


def _fatal_locate_error(error):
    if isinstance(error, FileNotFoundError):
        return True
    errno = getattr(error, "errno", None)
    text = str(error).casefold()
    return errno == 2 or "connection lost" in text or "no such file or directory" in text


async def locate_world(machine, arm, config, segmenter, *, settle=False, previous_xyz=None):
    """Segmenter → world. Skip the extra 2D detect; detections-to-segments already used it."""
    timeout = config["rpc_timeout_s"]
    if previous_xyz is None and await arm.is_moving(timeout=timeout):
        raise ValueError("Arm must be stopped while localizing with the wrist camera")
    if settle:
        logger.info("settling %.2fs before first localization", config["settle_s"])
        await asyncio.sleep(config["settle_s"])
    before = list((await arm.get_joint_positions(timeout=timeout)).values)
    objects = await segmenter.get_object_point_clouds(config["camera"], timeout=timeout)
    candidates = [(obj, geometry) for obj in objects for geometry in obj.geometries.geometries
                  if geometry.label.strip().casefold() == config["ball_label"].strip().casefold()
                  and obj.point_cloud and geometry.HasField("center")]

    async def to_world(obj, geometry):
        source_frame = obj.geometries.reference_frame or config["camera"]
        center = geometry.center
        camera_pose = BallPose(center.x, center.y, center.z, source_frame, geometry.label, len(obj.point_cloud))
        return await asyncio.wait_for(ball_pose_in_world(machine, camera_pose, config["world_frame"]), timeout)

    worlds = list(await asyncio.gather(*[to_world(obj, geometry) for obj, geometry in candidates]))
    world = pick_tracked_ball(worlds, previous_xyz, config)
    after = list((await arm.get_joint_positions(timeout=timeout)).values)
    if (len(after) != len(before) or max(abs(a - b) for a, b in zip(before, after)) > 0.1):
        raise ValueError("Arm moved while localizing; discard camera-to-world result")
    logger.debug("localized world=%s points=%d", _fmt((world.x, world.y, world.z)), world.point_count)
    return world, BallDetection(world.label, 1.0, 0, 0, 0, 0)


async def run_ball_catch(machine, config, *, execute=False, locate=None, now=None, sleep=None):
    """Track rest → motion, predict intercept, optionally move to meet the ball."""
    validate_config(config)
    locate = locate or locate_world
    now = now or time.monotonic
    sleep = sleep or asyncio.sleep
    arm = Arm.from_robot(machine, config["arm"])
    gripper = Gripper.from_robot(machine, config["gripper"])
    motion = MotionClient.from_robot(machine, config["motion"])
    segmenter = VisionClient.from_robot(machine, config["segmenter"])
    state, commanded, grabbed = "watch", False, False
    result = {"success": False, "executed": False, "grabbed": False}
    samples = []
    rest_xyz = None
    saw_rest = False
    motion_samples = []
    acceleration = config["acceleration_mm_s2"]

    async def current_pose():
        value = await motion.get_pose(config["gripper"], config["world_frame"], timeout=config["rpc_timeout_s"])
        if value.reference_frame != config["world_frame"]:
            raise ValueError("Gripper pose returned an unexpected frame")
        return _xyz(value.pose)

    async def stop():
        responses = await asyncio.gather(arm.stop(timeout=3), gripper.stop(timeout=3), return_exceptions=True)
        return [str(value) for value in responses if isinstance(value, BaseException)]

    needed = int(config["min_motion_samples"])
    leave_speed = float(config["leave_rest_speed_mm_s"])
    leave_shift = float(config["leave_rest_shift_mm"])
    if execute:
        logger.info("execute=True: will open gripper and intercept after leave-rest")
    else:
        logger.warning("execute=False: preview only; the arm will not move. Re-run with --execute")
    logger.info(
        "watching up to %.1fs; leave-rest needs shift>=%.0f mm from the latched rest pose; "
        "%d motion samples, target period %.3fs",
        config["track_timeout_s"], leave_shift, needed, config["sample_period_s"])

    async def pace(started):
        remaining = float(config["sample_period_s"]) - (now() - started)
        if remaining > 0:
            await sleep(remaining)

    try:
        deadline = now() + float(config["track_timeout_s"])
        first = True
        misses = 0
        while now() < deadline:
            started = now()
            try:
                previous = samples[-1].xyz if samples else None
                ball, detection = await locate(
                    machine, arm, config, segmenter, settle=first, previous_xyz=previous)
                first = False
                misses = 0
            except asyncio.CancelledError:
                raise
            except Exception as error:
                first = False
                if _fatal_locate_error(error):
                    raise ValueError(f"Lost connection to the machine while localizing: {error}") from error
                misses += 1
                logger.warning("localize miss %d: %s", misses, error)
                if saw_rest and motion_samples and misses >= 3:
                    if len(motion_samples) >= 2:
                        logger.warning("lost ball; predicting from %d motion samples", len(motion_samples))
                        break
                    raise ValueError(f"Lost ball after it left rest: {error}") from error
                await pace(started)
                continue
            sample = Sample(now(), (ball.x, ball.y, ball.z))
            samples.append(sample)
            speed = sample_speed(samples)
            shift = math.dist(sample.xyz, rest_xyz) if rest_xyz is not None else 0.0
            dt = sample.t - samples[-2].t if len(samples) > 1 else 0.0
            result.update(ball_world_mm=asdict(ball), midpoint_px=[detection.center_x, detection.center_y],
                          samples=[[round(item.t, 4), *[round(v, 2) for v in item.xyz]] for item in samples[-12:]])
            logger.info(
                "n=%d dt=%.2fs world=%s speed=%.1f mm/s shift=%.1f mm (need >= %.0f) "
                "saw_rest=%s tracking=%d/%d",
                len(samples), dt, _fmt(sample.xyz), speed, shift, leave_shift,
                saw_rest, len(motion_samples), needed)
            if state == "tracking":
                motion_samples.append(sample)
                logger.info("tracking %d/%d at %s", len(motion_samples), needed, _fmt(sample.xyz))
                if len(motion_samples) >= needed:
                    break
            elif state != "tracking" and at_rest(samples, config) and (
                    rest_xyz is None or math.dist(sample.xyz, rest_xyz) < leave_shift):
                if rest_xyz is None:
                    logger.info("ball at rest at %s", _fmt(sample.xyz))
                    rest_xyz = sample.xyz
                    saw_rest = True
                state = "rest"
            elif saw_rest and left_rest(samples, rest_xyz, config):
                motion_samples.append(sample)
                state = "tracking"
                logger.info("left rest: speed=%.1f mm/s shift=%.1f mm tracking=%d/%d",
                            speed, shift, len(motion_samples), needed)
                if len(motion_samples) >= needed:
                    break
            elif (not saw_rest and len(samples) >= needed and speed >= leave_speed):
                motion_samples = samples[-needed:]
                state = "tracking"
                logger.info("ball already moving, skipping rest wait: speed=%.1f mm/s samples=%d",
                            speed, len(motion_samples))
                break
            await pace(started)
        else:
            shift = math.dist(samples[-1].xyz, rest_xyz) if samples and rest_xyz is not None else 0.0
            raise ValueError(
                "Timed out waiting for the ball to leave rest with a usable velocity: "
                f"saw_rest={saw_rest}, samples={len(samples)}, "
                f"last_speed={sample_speed(samples):.1f} mm/s, "
                f"shift_from_rest={shift:.1f} mm (need >= {leave_shift:.0f})"
            )

        state = "predict"
        track = motion_samples if len(motion_samples) >= 2 else samples[-needed:]
        position, velocity, residual = fit_velocity(track, acceleration)
        delay, predicted, target = choose_delay(position, velocity, acceleration, config)
        current = await current_pose()
        travel_s = move_duration(current, target, config)
        wait_s = max(0.0, delay - travel_s)
        intercept_t = track[-1].t + delay
        result.update(state=state, rest_world_mm=list(rest_xyz) if rest_xyz else None,
                      position_world_mm=position, velocity_mm_s=velocity, fit_residual_mm=residual,
                      catch_delay_s=delay, predicted_ball_world_mm=predicted,
                      catch_target_world_mm=target, arm_travel_s=travel_s, wait_s=wait_s,
                      intercept_monotonic_s=intercept_t, grasp_orientation=config["grasp_orientation"])
        logger.info(
            "predict v=%s mm/s residual=%.2f mm delay=%.2fs ball@%s tcp@%s travel=%.2fs wait=%.2fs",
            [round(v, 1) for v in velocity], residual, delay, _fmt(predicted), _fmt(target),
            travel_s, wait_s)
        if not execute:
            logger.warning("preview complete; not commanding motion")
            return {**result, "success": True, "state": "preview"}

        state = "open"
        logger.info("opening gripper")
        await gripper.open(timeout=config["rpc_timeout_s"])
        remaining = intercept_t - now()
        wait_s = max(0.0, remaining - travel_s)
        result["wait_s"] = wait_s
        if wait_s > 0:
            state = "wait"
            logger.info("waiting %.2fs before intercept", wait_s)
            await sleep(wait_s)
        state = "intercept"
        ox, oy, oz, theta = config["grasp_orientation"]
        destination = PoseInFrame(reference_frame=config["world_frame"],
                                  pose=Pose(x=target[0], y=target[1], z=target[2],
                                            o_x=ox, o_y=oy, o_z=oz, theta=theta))
        commanded = True
        timeout = max(float(config["move_timeout_s"]), travel_s + 2)
        logger.info("moving gripper to %s (timeout=%.1fs)", _fmt(target), timeout)
        success = await asyncio.wait_for(motion.move(component_name=config["gripper"], destination=destination,
                                                     timeout=timeout), timeout)
        if not success:
            raise RuntimeError("Motion failed during intercept")
        state = "grab"
        grabbed = bool(await gripper.grab(timeout=config["rpc_timeout_s"]))
        arrival = await current_pose()
        result.update(final_gripper_world_mm=arrival,
                      arrival_error_mm=math.dist(arrival, target))
        logger.info("grab=%s arrival_error=%.1f mm at %s", grabbed, result["arrival_error_mm"], _fmt(arrival))
        return {**result, "success": True, "executed": True, "grabbed": grabbed, "state": "caught"}
    except asyncio.CancelledError:
        logger.warning("cancelled during %s; stopping hardware=%s", state, commanded)
        if commanded:
            await asyncio.shield(stop())
        raise
    except Exception as error:
        logger.error("failed during %s: %s", state, error)
        result.update(state=state, executed=commanded, grabbed=grabbed,
                      reason=f"{type(error).__name__}: {error}")
        if commanded:
            result["stop_errors"] = await stop()
        return result


async def _main(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.catch_delay is not None:
        config["catch_delay_s"] = args.catch_delay
    validate_config(config)
    async with await connect() as machine:
        return await run_ball_catch(machine, config, execute=args.execute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true", help="Move the physical robot to the intercept")
    parser.add_argument("--catch-delay", type=float, default=None,
                        help="Seconds from the last motion sample to intercept (overrides config)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    try:
        result = asyncio.run(_main(args))
        print(json.dumps(result, indent=2, allow_nan=False))
        if not result["success"]:
            raise SystemExit(2)
    except (ValueError, RuntimeError, TimeoutError) as error:
        parser.exit(2, f"Ball catch failed: {error}\n")


if __name__ == "__main__":
    main()
