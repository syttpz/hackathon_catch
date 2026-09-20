"""Camera-frame geometry shared by catching and calibration workflows."""

import asyncio

import numpy as np
from viam.proto.common import Pose, PoseInFrame


async def camera_transform(robot, camera_name, world):
    """Return a camera origin and rotation in the requested world frame."""
    points = []
    for xyz in ((0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, 100)):
        transformed = await asyncio.wait_for(
            robot.transform_pose(
                PoseInFrame(
                    reference_frame=camera_name,
                    pose=Pose(x=xyz[0], y=xyz[1], z=xyz[2], o_z=1),
                ),
                world,
            ),
            5,
        )
        points.append([transformed.pose.x, transformed.pose.y, transformed.pose.z])

    origin = np.array(points[0])
    rotation = (np.array(points[1:]) - origin).T / 100
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.001):
        raise ValueError("Camera transform changed during calibration")
    return origin, rotation
