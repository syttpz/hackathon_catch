import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from viam.proto.common import Pose, PoseInFrame

from vision.viam_ball import (
    BallLocalizationError,
    BallPose,
    ball_pose_in_world,
    detect_ball,
    locate_ball_3d,
    gripper_pose_in_world,
)


def detection(label, confidence, xmin=10, ymin=20, xmax=30, ymax=40):
    return SimpleNamespace(
        class_name=label,
        confidence=confidence,
        x_min=xmin,
        y_min=ymin,
        x_max=xmax,
        y_max=ymax,
    )


def point_cloud_object(label, point_count, center=(1, 2, 3), frame="cam"):
    geometry = SimpleNamespace(label=label, center=Pose(x=center[0], y=center[1], z=center[2]))
    return SimpleNamespace(
        point_cloud=b"x" * point_count,
        geometries=SimpleNamespace(reference_frame=frame, geometries=[geometry]),
    )


class ViamBallTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_gripper_tcp_in_world_without_motion(self):
        motion = SimpleNamespace(get_pose=AsyncMock(return_value=PoseInFrame(
            reference_frame="world", pose=Pose(x=10, y=20, z=30, o_z=-1, theta=180)
        )))
        with patch("vision.viam_ball.MotionClient.from_robot", return_value=motion):
            result = await gripper_pose_in_world(object())
        self.assertEqual(result["z"], 30)
        self.assertEqual(result["orientation"], [0, 0, -1, 180])
        motion.get_pose.assert_awaited_once_with(
            component_name="gripper", destination_frame="world", timeout=10
        )

    async def test_detect_selects_highest_confidence_ball_and_midpoint(self):
        service = SimpleNamespace(get_detections_from_camera=AsyncMock(return_value=[
            detection("other", 1.0), detection("BALL", 0.6), detection("ball", 0.9, 11, 21, 31, 41)
        ]))
        with patch("vision.viam_ball.VisionClient.from_robot", return_value=service):
            result = await detect_ball(object())
        self.assertEqual(result.label, "ball")
        self.assertEqual((result.center_x, result.center_y), (21, 31))
        service.get_detections_from_camera.assert_awaited_once_with("cam", timeout=10)

    async def test_detect_rejects_missing_ball_label(self):
        service = SimpleNamespace(get_detections_from_camera=AsyncMock(return_value=[detection("cube", 1.0)]))
        with patch("vision.viam_ball.VisionClient.from_robot", return_value=service):
            with self.assertRaises(BallLocalizationError):
                await detect_ball(object())

    async def test_localize_selects_largest_matching_labeled_segment(self):
        service = SimpleNamespace(get_object_point_clouds=AsyncMock(return_value=[
            point_cloud_object("cube", 20),
            point_cloud_object("BALL", 10, center=(10, 20, 30)),
            point_cloud_object("ball", 30, center=(40, 50, 60)),
        ]))
        with patch("vision.viam_ball.VisionClient.from_robot", return_value=service):
            result = await locate_ball_3d(object())
        self.assertEqual((result.x, result.y, result.z, result.reference_frame), (40, 50, 60, "cam"))
        self.assertEqual(result.point_count, 30)

    async def test_localize_reports_segmenter_failure_without_fallback_depth(self):
        service = SimpleNamespace(get_object_point_clouds=AsyncMock(side_effect=RuntimeError("camera unavailable")))
        with patch("vision.viam_ball.VisionClient.from_robot", return_value=service):
            with self.assertRaisesRegex(BallLocalizationError, "camera_name"):
                await locate_ball_3d(object())

    async def test_transform_to_world_uses_source_reference_frame(self):
        machine = SimpleNamespace(transform_pose=AsyncMock(return_value=PoseInFrame(
            reference_frame="world", pose=Pose(x=100, y=200, z=300)
        )))
        result = await ball_pose_in_world(machine, BallPose(1, 2, 3, "cam", "ball", 99))
        self.assertEqual((result.x, result.y, result.z, result.reference_frame), (100, 200, 300, "world"))
        source, destination = machine.transform_pose.call_args.args
        self.assertEqual((source.reference_frame, source.pose.x, source.pose.y, source.pose.z, destination),
                         ("cam", 1, 2, 3, "world"))


if __name__ == "__main__":
    unittest.main()
