"""Read-only Viam ball detection and 3D localization helpers.

Run ``python -m vision.viam_ball`` to verify the current machine configuration.
This module never moves the arm, opens/closes the gripper, or edits Viam config.
"""
import argparse
import asyncio
import json
from dataclasses import asdict, dataclass

from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from connection import connect


class BallLocalizationError(RuntimeError):
    """The Viam detector/segmenter could not provide an unambiguous ball pose."""


@dataclass(frozen=True)
class BallDetection:
    label: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2


@dataclass(frozen=True)
class BallPose:
    x: float
    y: float
    z: float
    reference_frame: str
    label: str
    point_count: int


def _matches_ball(label: str, ball_label: str) -> bool:
    return label.strip().casefold() == ball_label.strip().casefold()


async def detect_ball(machine, detector="color_detect", camera="cam", ball_label="ball") -> BallDetection:
    """Return the highest-confidence Viam detection with the requested label."""
    service = VisionClient.from_robot(machine, detector)
    detections = await service.get_detections_from_camera(camera, timeout=10)
    matches = [d for d in detections if _matches_ball(d.class_name, ball_label)]
    if not matches:
        raise BallLocalizationError(f"No {ball_label!r} detection from {detector!r} on {camera!r}")
    selected = max(matches, key=lambda d: d.confidence)
    return BallDetection(
        label=selected.class_name,
        confidence=float(selected.confidence),
        x_min=int(selected.x_min),
        y_min=int(selected.y_min),
        x_max=int(selected.x_max),
        y_max=int(selected.y_max),
    )


async def locate_ball_3d(machine, segmenter="object-segmenter", camera="cam", ball_label="ball") -> BallPose:
    """Return a matching detections-to-segments geometry center.

    The segmenter must be configured in Viam with ``detector_name: color_detect``
    and ``camera_name: cam``. This function does not estimate depth from a 2D box.
    """
    service = VisionClient.from_robot(machine, segmenter)
    try:
        objects = await service.get_object_point_clouds(camera, timeout=15)
    except Exception as error:
        raise BallLocalizationError(
            f"{segmenter!r} could not localize from {camera!r}: {error}. "
            "Verify its detector_name, camera_name, and that cam supports point clouds."
        ) from error

    candidates = []
    for obj in objects:
        frame = obj.geometries.reference_frame or camera
        for geometry in obj.geometries.geometries:
            if _matches_ball(geometry.label, ball_label):
                candidates.append((len(obj.point_cloud), geometry, frame))
    if not candidates:
        raise BallLocalizationError(
            f"No 3D segment labeled {ball_label!r} from {segmenter!r}. "
            "The 2D detector may not be connected to the segmenter, or depth is invalid."
        )
    point_count, geometry, reference_frame = max(candidates, key=lambda item: item[0])
    return BallPose(
        x=float(geometry.center.x),
        y=float(geometry.center.y),
        z=float(geometry.center.z),
        reference_frame=reference_frame,
        label=geometry.label,
        point_count=point_count,
    )


async def ball_pose_in_world(machine, pose: BallPose, world_frame="world") -> BallPose:
    """Transform a Viam segment center to the requested planning frame."""
    transformed = await machine.transform_pose(
        PoseInFrame(
            reference_frame=pose.reference_frame,
            pose=Pose(x=pose.x, y=pose.y, z=pose.z, o_z=1.0),
        ),
        world_frame,
    )
    return BallPose(
        x=float(transformed.pose.x),
        y=float(transformed.pose.y),
        z=float(transformed.pose.z),
        reference_frame=world_frame,
        label=pose.label,
        point_count=pose.point_count,
    )


async def camera_capabilities(machine, camera="cam") -> dict:
    """Return only the camera details needed to diagnose 3D localization."""
    properties = await Camera.from_robot(machine, camera).get_properties(timeout=10)
    intrinsics = properties.intrinsic_parameters
    return {
        "camera": camera,
        "supports_pcd": bool(properties.supports_pcd),
        "intrinsics": {
            "width": intrinsics.width_px,
            "height": intrinsics.height_px,
            "fx": intrinsics.focal_x_px,
            "fy": intrinsics.focal_y_px,
            "cx": intrinsics.center_x_px,
            "cy": intrinsics.center_y_px,
        },
    }


async def gripper_pose_in_world(machine, gripper="gripper", motion="builtin", world_frame="world") -> dict:
    """Return the configured gripper TCP pose without commanding any motion."""
    pose_in_world = await MotionClient.from_robot(machine, motion).get_pose(
        component_name=gripper,
        destination_frame=world_frame,
        timeout=10,
    )
    pose = pose_in_world.pose
    return {
        "frame": pose_in_world.reference_frame,
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "orientation": [float(pose.o_x), float(pose.o_y), float(pose.o_z), float(pose.theta)],
    }


async def verify_ball(camera="cam", detector="color_detect", segmenter="object-segmenter", ball_label="ball",
                      include_gripper_pose=False) -> dict:
    """Read and return detection, camera-frame pose, and world-frame pose."""
    async with await connect() as machine:
        result = {"camera": await camera_capabilities(machine, camera)}
        detection = await detect_ball(machine, detector, camera, ball_label)
        result["detection"] = {**asdict(detection), "midpoint_px": [detection.center_x, detection.center_y]}
        camera_pose = await locate_ball_3d(machine, segmenter, camera, ball_label)
        result["camera_pose_mm"] = asdict(camera_pose)
        result["world_pose_mm"] = asdict(await ball_pose_in_world(machine, camera_pose))
        if include_gripper_pose:
            gripper = await gripper_pose_in_world(machine)
            result["gripper_pose_mm"] = gripper
            result["ball_minus_gripper_mm"] = {
                "x": result["world_pose_mm"]["x"] - gripper["x"],
                "y": result["world_pose_mm"]["y"] - gripper["y"],
                "z": result["world_pose_mm"]["z"] - gripper["z"],
            }
        return result


async def verify_gripper_pose(gripper="gripper", motion="builtin", world_frame="world") -> dict:
    """Read the current gripper TCP pose independently of camera availability."""
    async with await connect() as machine:
        return {"gripper_pose_mm": await gripper_pose_in_world(machine, gripper, motion, world_frame)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="cam")
    parser.add_argument("--detector", default="color_detect")
    parser.add_argument("--segmenter", default="object-segmenter")
    parser.add_argument("--label", default="ball")
    parser.add_argument("--include-gripper-pose", action="store_true")
    parser.add_argument("--gripper-pose-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.gripper_pose_only:
            result = verify_gripper_pose()
        else:
            result = verify_ball(
                args.camera, args.detector, args.segmenter, args.label, args.include_gripper_pose
            )
        print(json.dumps(asyncio.run(result), indent=2))
    except BallLocalizationError as error:
        parser.exit(2, f"Ball localization failed: {error}\n")


if __name__ == "__main__":
    main()
