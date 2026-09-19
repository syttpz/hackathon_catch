"""Viam-native stationary pickup. Preview by default; --execute moves hardware."""
import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from connection import connect
from vision.viam_ball import BallPose, ball_pose_in_world, detect_ball


_cycle_lock = asyncio.Lock()
DEFAULT_CONFIG = Path(__file__).parents[1] / "stationary_grab.config.json"


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
    for key in ("approach_height_mm", "lift_height_mm", "max_reacquire_shift_mm",
                "stationary_tolerance_mm", "move_timeout_s",
                "rpc_timeout_s", "max_observation_s", "arrival_tolerance_mm", "settle_s"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("grasp_z_world_mm", "expected_ball_z_world_mm"):
        if not math.isfinite(float(config[key])):
            raise ValueError(f"{key} must be finite")
    if config["grasp_z_world_mm"] <= config["expected_ball_z_world_mm"]:
        raise ValueError("Taught TCP grasp height must be above expected ball center")
    _vector(config["grasp_xy_offset_mm"], 2, "grasp_xy_offset_mm")
    orientation = _vector(config["grasp_orientation"], 4, "grasp_orientation")
    if not math.isclose(math.sqrt(sum(v*v for v in orientation[:3])), 1, abs_tol=0.01):
        raise ValueError("Grasp orientation direction must be normalized")
    if orientation[2] > -0.9:
        raise ValueError("This table pickup requires a downward grasp orientation")
    bounds = config["workspace_mm"]
    if len(bounds) != 2:
        raise ValueError("workspace_mm needs lower and upper XYZ bounds")
    lower, upper = [_vector(row, 3, "workspace_mm") for row in bounds]
    if any(a >= b for a, b in zip(lower, upper)):
        raise ValueError("Workspace lower bounds must be less than upper bounds")


def _xyz(pose):
    return [pose.x, pose.y, pose.z]


def _check_bounds(point, config, name="Target"):
    point = _vector(list(point), 3, name)
    if any(not low <= value <= high for value, low, high in zip(point, *config["workspace_mm"])):
        raise ValueError(f"{name} {point} is outside workspace_mm")


def make_targets(ball, config):
    """Use detected world XYZ. Grasp Z is the segment center plus the taught TCP offset.

    grasp_z_world_mm and expected_ball_z_world_mm are a calibration pair: TCP height
    at the same moment as a measured ball-center height. Their difference is applied
    to the live point-cloud Z so a tossed or raised ball is followed instead of a table.
    """
    if ball.reference_frame != config["world_frame"]:
        raise ValueError("Ball must be transformed into the configured world frame")
    _vector(_xyz(ball), 3, "ball XYZ")
    dx, dy = config["grasp_xy_offset_mm"]
    tcp_to_ball_z = config["grasp_z_world_mm"] - config["expected_ball_z_world_mm"]
    grasp = [ball.x+dx, ball.y+dy, ball.z + tcp_to_ball_z]
    targets = {"grasp": grasp,
               "approach": [grasp[0], grasp[1], grasp[2]+config["approach_height_mm"]],
               "lift": [grasp[0], grasp[1], grasp[2]+config["lift_height_mm"]]}
    for name, target in targets.items():
        _check_bounds(target, config, f"{name} target")
    return targets


async def _observe(machine, arm, config):
    """One stopped-arm sample. Later frames may lose the ball; callers keep this pose."""
    timeout = config["rpc_timeout_s"]
    if await arm.is_moving(timeout=timeout):
        raise ValueError("Arm must be stopped before wrist-camera localization")
    await asyncio.sleep(config["settle_s"])
    before = list((await arm.get_joint_positions(timeout=timeout)).values)
    if not before or not all(math.isfinite(v) for v in before):
        raise ValueError("Invalid arm joint positions")
    segmenter = VisionClient.from_robot(machine, config["segmenter"])
    start = time.monotonic()
    detection = await detect_ball(machine, config["detector"], config["camera"], config["ball_label"])
    if not math.isfinite(detection.confidence) or detection.confidence < 0.5:
        raise ValueError("Ball detection confidence is below 0.5")
    objects = await segmenter.get_object_point_clouds(config["camera"], timeout=timeout)
    candidates = [(obj, geometry) for obj in objects for geometry in obj.geometries.geometries
                  if geometry.label.strip().casefold() == config["ball_label"].strip().casefold()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one ball segment; found {len(candidates)}")
    obj, geometry = candidates[0]
    if not obj.point_cloud or not geometry.HasField("center"):
        raise ValueError("Ball segment has no point cloud or center")
    # detections-to-segments returns camera coordinates when its frame is unset.
    source_frame = obj.geometries.reference_frame or config["camera"]
    center = geometry.center
    ball = BallPose(center.x, center.y, center.z, source_frame, geometry.label, len(obj.point_cloud))
    _vector(_xyz(ball), 3, "segment XYZ")
    world = await asyncio.wait_for(ball_pose_in_world(machine, ball, config["world_frame"]), timeout)
    _vector(_xyz(world), 3, "world XYZ")
    after = list((await arm.get_joint_positions(timeout=timeout)).values)
    if (await arm.is_moving(timeout=timeout) or len(after) != len(before)
            or not all(math.isfinite(v) for v in after)
            or max(abs(a-b) for a, b in zip(before, after)) > 0.1):
        raise ValueError("Arm moved while localizing; discard camera-to-world result")
    elapsed = time.monotonic()-start
    if elapsed > config["max_observation_s"]:
        raise ValueError(
            "Localization took too long; target is stale: "
            f"elapsed={elapsed:.2f} s, limit={config['max_observation_s']:.2f} s"
        )
    return world, detection


async def run_stationary_grab(machine, config, *, execute=False, approach_only=False):
    """Detect once, then approach and optionally grab using that world pose.

    Preview (default) reads sensors and returns targets. --execute runs one cycle.
    Approach and grab both use the first localization; later camera loss is ignored.
    Requires exclusive arm control; concurrent calls in this process are rejected.
    """
    validate_config(config)
    if _cycle_lock.locked():
        return {"success": False, "state": "busy", "executed": False}
    async with _cycle_lock:
        arm = Arm.from_robot(machine, config["arm"])
        gripper = Gripper.from_robot(machine, config["gripper"])
        motion = MotionClient.from_robot(machine, config["motion"])
        state, commanded, grabbed = "detect", False, False
        result = {"success": False, "executed": False, "grabbed": False}

        async def current_pose():
            value = await motion.get_pose(config["gripper"], config["world_frame"], timeout=config["rpc_timeout_s"])
            if value.reference_frame != config["world_frame"]:
                raise ValueError("Gripper pose returned an unexpected frame")
            _vector(_xyz(value.pose), 3, "gripper XYZ")
            return value.pose

        async def move(point, *, linear=False, check_workspace=True):
            nonlocal commanded
            if check_workspace:
                _check_bounds(point, config)
            ox, oy, oz, theta = config["grasp_orientation"]
            target = PoseInFrame(reference_frame=config["world_frame"],
                                 pose=Pose(x=point[0], y=point[1], z=point[2],
                                           o_x=ox, o_y=oy, o_z=oz, theta=theta))
            constraints = Constraints(linear_constraint=[LinearConstraint(
                line_tolerance_mm=2, orientation_tolerance_degs=2)]) if linear else None
            commanded = True
            success = await asyncio.wait_for(motion.move(
                component_name=config["gripper"], destination=target, constraints=constraints,
                timeout=config["move_timeout_s"]), config["move_timeout_s"])
            if not success:
                raise RuntimeError(f"Motion failed during {state}")
            if await arm.is_moving(timeout=config["rpc_timeout_s"]):
                raise RuntimeError("Arm still moving after motion completion")
            if math.dist(_xyz(await current_pose()), point) > config["arrival_tolerance_mm"]:
                raise RuntimeError("Gripper did not reach the requested position")

        async def stop():
            responses = await asyncio.gather(arm.stop(timeout=3), gripper.stop(timeout=3), return_exceptions=True)
            return [str(value) for value in responses if isinstance(value, BaseException)]

        try:
            ball, detection = await _observe(machine, arm, config)
            result.update(ball_world_mm=asdict(ball),
                          midpoint_px=[detection.center_x, detection.center_y])
            targets = make_targets(ball, config)
            current = await current_pose()
            # workspace_mm bounds pickup targets only. The observe/home TCP may be outside
            # that table box; raising in place is allowed so the arm can then enter it.
            clearance = [current.x, current.y, max(current.z, targets["approach"][2])]
            result.update(targets_world_mm=targets, initial_clearance_world_mm=clearance,
                          initial_gripper_world_mm=_xyz(current),
                          grasp_orientation=config["grasp_orientation"])
            if not execute:
                return {**result, "success": True, "state": "preview"}

            state = "clearance"
            if current.z < clearance[2]-config["arrival_tolerance_mm"]:
                await move(clearance, check_workspace=False)
            state = "approach"
            await move(targets["approach"])
            if approach_only:
                return {**result, "success": True, "executed": True, "state": "approached"}
            state = "open"
            await gripper.open(timeout=config["rpc_timeout_s"])
            state = "descend"
            await move(targets["grasp"], linear=True)
            state = "grab"
            grabbed = bool(await gripper.grab(timeout=config["rpc_timeout_s"]))
            if not grabbed:
                raise RuntimeError("Gripper reported an empty/failed grasp; lift cancelled")
            state = "lift"
            await move(targets["lift"], linear=True)
            return {**result, "success": True, "executed": True, "grabbed": True, "state": "ball_grabbed"}
        except asyncio.CancelledError:
            if commanded:
                await asyncio.shield(stop())
            raise
        except Exception as error:
            result.update(state=state, executed=commanded, grabbed=grabbed,
                          reason=f"{type(error).__name__}: {error}")
            if commanded:
                result["stop_errors"] = await stop()
            return result


async def _main(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    async with await connect() as machine:
        return await run_stationary_grab(machine, config, execute=args.execute, approach_only=args.approach_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true", help="Move the physical robot for one cycle")
    parser.add_argument("--approach-only", action="store_true", help="Stop above the ball; do not open or grip")
    args = parser.parse_args()
    try:
        result = asyncio.run(_main(args))
        print(json.dumps(result, indent=2, allow_nan=False))
        if not result["success"]:
            raise SystemExit(2)
    except (ValueError, RuntimeError, TimeoutError) as error:
        parser.exit(2, f"Stationary grab failed: {error}\n")


if __name__ == "__main__":
    main()
