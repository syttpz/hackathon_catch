"""Run: python -m motion.ball_pickup --config ball_config.json [--execute]."""
import argparse
import asyncio
import json
import time

import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from ball_tracking import BallCamera, Trajectory
from connection import connect


def check_workspace(point, bounds):
    point, bounds = np.asarray(point, float), np.asarray(bounds, float)
    if (point.shape != (3,) or bounds.shape != (2, 3)
            or not np.isfinite(point).all() or not np.isfinite(bounds).all()
            or not np.all(bounds[0] < bounds[1])
            or np.any(point < bounds[0]) or np.any(point > bounds[1])):
        raise ValueError("Target is outside the configured world workspace")


async def collect_track(camera, config):
    track = Trajectory(config.get("acceleration_mm_s2", [0, 0, 0]),
                       max_age=config.get("max_age_s", 0.25))
    # Flush images that could predate the last arm move.
    await asyncio.sleep(config.get("settle_s", 0.4))
    not_before = time.time()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        observation = await camera.observe()
        if observation is None:
            track.samples.clear()
        elif observation.timestamp >= not_before:
            track.add(observation)
            if len(track.samples) >= 6:
                return track
        await asyncio.sleep(0.025)
    raise ValueError("No unambiguous red ball track within five seconds")


async def move_to_ball_and_grip(machine, config, execute=False):
    """Observe, approach, reobserve, descend, verify alignment, then grip.

    Conservative stop-and-look pickup for stationary/very slow balls. Prediction
    is available for moving balls, but this is not a real-time catching servo.
    Gripper's configured frame origin must be the actual grasp center.
    """
    arm = Arm.from_robot(machine, config["arm"])
    camera = BallCamera(Camera.from_robot(machine, config["camera"]), config, machine, arm)
    track = await collect_track(camera, config)
    lead = float(config["prediction_lead_s"])
    target, velocity, residual = track.predict(time.time() + lead)
    result = {"xyz_world_mm": target.tolist(), "velocity_mm_s": velocity.tolist(),
              "fit_error_mm": residual, "prediction_lead_s": lead, "executed": False}
    if not execute:
        return result
    if not config.get("calibration_verified"):
        raise ValueError("Verify calibration, gripper frame, orientation and workspace before execution")
    for key in ("max_pickup_speed_mm_s", "max_target_shift_mm", "grasp_tolerance_mm", "move_timeout_s"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if np.linalg.norm(velocity) > config["max_pickup_speed_mm_s"]:
        raise ValueError("Ball is too fast for stop-and-look pickup")
    bounds = config["workspace_mm"]
    orientation = np.asarray(config["grasp_orientation"], float)
    if orientation.shape != (4,) or not np.isfinite(orientation).all() or np.linalg.norm(orientation[:3]) < 1e-6:
        raise ValueError("Set grasp orientation [o_x, o_y, o_z, theta_degrees]")
    approach_mm = float(config["approach_mm"])
    if not np.isfinite(approach_mm) or approach_mm <= 0:
        raise ValueError("approach_mm must be positive")
    gripper = Gripper.from_robot(machine, config["gripper"])
    motion = MotionClient.from_robot(machine, config.get("motion", "builtin"))

    async def move(point):
        check_workspace(point, bounds)
        pose = Pose(x=float(point[0]), y=float(point[1]), z=float(point[2]),
                    o_x=float(orientation[0]), o_y=float(orientation[1]),
                    o_z=float(orientation[2]), theta=float(orientation[3]))
        success = await motion.move(
            component_name=config["gripper"],
            destination=PoseInFrame(reference_frame=config["world_frame"], pose=pose),
            timeout=config.get("move_timeout_s", 15))
        if not success:
            raise RuntimeError("Viam motion planner did not complete the move")

    async def fresh_target():
        fresh = await collect_track(camera, config)
        point, speed, _ = fresh.predict(time.time() + lead)
        if np.linalg.norm(speed) > config["max_pickup_speed_mm_s"]:
            raise ValueError("Ball accelerated; aborting pickup")
        return point

    try:
        approach = target + [0, 0, approach_mm]
        check_workspace(target, bounds)
        check_workspace(approach, bounds)
        await gripper.open(timeout=3)
        await move(approach)
        # Refresh after the approach; never descend using its old observation.
        updated = await fresh_target()
        if np.linalg.norm(updated-target) > config["max_target_shift_mm"]:
            raise ValueError("Ball moved too far during approach")
        await move(updated)
        actual = await motion.get_pose(config["gripper"], config["world_frame"], timeout=2)
        grasp_xyz = np.array([actual.pose.x, actual.pose.y, actual.pose.z])
        final_target = await fresh_target()
        if np.linalg.norm(final_target-grasp_xyz) > config["grasp_tolerance_mm"]:
            raise ValueError("Ball is no longer inside the grasp tolerance")
        grabbed = await gripper.grab(timeout=3)
        if not grabbed:
            raise RuntimeError("Gripper did not report a successful grasp")
        result.update(executed=True, grabbed=True, xyz_world_mm=final_target.tolist())
        return result
    except BaseException:
        # Cancellation/timeouts must also stop physical motion.
        await asyncio.gather(arm.stop(timeout=2), gripper.stop(timeout=2), return_exceptions=True)
        raise


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)
    async with await connect() as machine:
        print(json.dumps(await move_to_ball_and_grip(machine, config, args.execute), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
