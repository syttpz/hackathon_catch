import json
import time
import queue
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from viam.proto.common import Pose, PoseInFrame

from ball_tracking import BallCamera, Observation, Trajectory, detect_red_ball
from motion.ball_pickup import check_workspace, move_to_ball_and_grip


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.color = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(self.color, (80, 60), 10, (0, 0, 255), -1)
        self.depth = np.full((120, 160), 480.0)
        self.intrinsics = dict(width=160, height=120, fx=250, fy=250, cx=80, cy=60)

    def detect(self):
        return detect_red_ball(self.color, self.depth, self.intrinsics, 20, 1)

    def test_center_depth_and_sphere_radius(self):
        np.testing.assert_allclose(self.detect().xyz, [0, 0, 500], atol=1)

    def test_box_midpoint_is_deprojection_pixel(self):
        observation = self.detect()
        left, top, right, bottom = observation.bbox
        self.assertEqual(observation.pixel, ((left+right)/2, (top+bottom)/2))
        self.assertEqual(observation.depth_mm, 480)

    def test_live_overlay_reports_alignment(self):
        from vision.live_ball import annotate
        image, status = annotate(self.color, self.depth, self.intrinsics, 20, 1, [75, 55])
        self.assertEqual(image.shape, (120, 320, 3))
        self.assertIn("du=+5.0, dv=+5.0", status)
        self.assertIn("500.0", status)

    def test_depth_holes(self):
        self.depth[:] = 0
        self.assertIsNone(self.detect())

    def test_two_red_balls_are_ambiguous(self):
        cv2.circle(self.color, (35, 60), 10, (0, 0, 255), -1)
        self.assertIsNone(self.detect())

    def test_misaligned_dimensions(self):
        self.depth = self.depth[:50]
        with self.assertRaises(ValueError):
            self.detect()

    def test_rolling_and_ballistic_prediction(self):
        for acceleration in ([0, 0, 0], [0, 0, -9810]):
            a = np.array(acceleration)
            track = Trajectory(a)
            start = np.array([100, 50, 1000])
            velocity = np.array([200, 0, 1000])
            for t in np.linspace(0, 0.2, 7):
                track.add(Observation(10+t, start+velocity*t+0.5*a*t*t, (0, 0)))
            point, _, error = track.predict(10.3, now=10.21)
            np.testing.assert_allclose(point, start+velocity*0.3+0.5*a*0.3**2, atol=1e-7)
            self.assertLess(error, 1e-7)
            with self.assertRaises(ValueError):
                track.predict(11, now=11)
            with self.assertRaises(ValueError):
                track.add(track.samples[-1])

    def test_workspace_rejects_nan_and_outside(self):
        for point in ([np.nan, 0, 0], [2, 0, 0]):
            with self.assertRaises(ValueError):
                check_workspace(point, [[-1, -1, -1], [1, 1, 1]])


class CameraTests(unittest.IsolatedAsyncioTestCase):
    async def test_viewer_reconnects_after_stream_termination(self):
        from grpclib.exceptions import StreamTerminatedError
        from vision.live_ball import stream
        stop = threading.Event()
        frames = queue.Queue(maxsize=1)
        first = SimpleNamespace(close=AsyncMock())
        second = SimpleNamespace(close=AsyncMock())
        broken = SimpleNamespace(get_images=AsyncMock(side_effect=StreamTerminatedError("Connection lost")))

        async def finish(**kwargs):
            stop.set()
            raise ValueError("test completed")

        good = SimpleNamespace(get_images=AsyncMock(side_effect=finish))
        with patch("vision.live_ball.connect", AsyncMock(side_effect=[first, second])) as connect_mock, \
             patch("vision.live_ball.Camera.from_robot", side_effect=[broken, good]), \
             patch("vision.live_ball.asyncio.sleep", AsyncMock()):
            await stream({"intrinsics": {}}, frames, stop)
        self.assertEqual(connect_mock.await_count, 2)
        first.close.assert_awaited_once()
        second.close.assert_awaited_once()
        good.get_images.assert_awaited_once()

    async def test_wrist_transform_and_motion_rejection(self):
        color = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(color, (80, 60), 10, (0, 0, 255), -1)
        _, png = cv2.imencode(".png", color)
        stamp = time.time()
        metadata = SimpleNamespace(captured_at=SimpleNamespace(seconds=int(stamp), nanos=int((stamp % 1)*1e9)))
        images = [SimpleNamespace(name="color", data=png.tobytes()),
                  SimpleNamespace(name="depth", bytes_to_depth_array=lambda: np.full((120, 160), 480))]
        camera = SimpleNamespace(get_images=AsyncMock(return_value=(images, metadata)))
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              get_joint_positions=AsyncMock(return_value=SimpleNamespace(values=[0]*6)))
        machine = SimpleNamespace(transform_pose=AsyncMock(return_value=PoseInFrame(pose=Pose(x=200, y=50, z=100))))
        config = {"color_source": "color", "depth_source": "depth", "ball_radius_mm": 20,
                  "camera_optical_frame": "cam", "world_frame": "world",
                  "intrinsics": dict(width=160, height=120, fx=250, fy=250, cx=80, cy=60)}
        adapter = BallCamera(camera, config, machine, arm)
        observation = await adapter.observe()
        np.testing.assert_allclose(observation.xyz, [200, 50, 100])
        query, destination = machine.transform_pose.call_args.args
        self.assertEqual(query.reference_frame, "cam")
        self.assertEqual(destination, "world")
        self.assertAlmostEqual(query.pose.z, 500)
        with self.assertRaises(ValueError):
            await adapter.observe()  # Duplicate timestamp.
        arm.is_moving.return_value = True
        with self.assertRaises(ValueError):
            await adapter.observe()


class PickupTests(unittest.IsolatedAsyncioTestCase):
    async def run_pickup(self, execute, move_success=True, grabbed=True, shift=False):
        config = json.loads((Path(__file__).parents[1] / "ball_config.example.json").read_text())
        config["calibration_verified"] = True
        arm = SimpleNamespace(stop=AsyncMock())
        gripper = SimpleNamespace(open=AsyncMock(), grab=AsyncMock(return_value=grabbed), stop=AsyncMock())
        motion = SimpleNamespace(move=AsyncMock(return_value=move_success),
                                 get_pose=AsyncMock(return_value=PoseInFrame(pose=Pose(x=200, y=0, z=100))))
        track = SimpleNamespace(predict=lambda *args: (np.array([200, 0, 100]), np.zeros(3), 0))
        changed = SimpleNamespace(predict=lambda *args: (np.array([250, 0, 100]), np.zeros(3), 0))
        with patch("motion.ball_pickup.Arm.from_robot", return_value=arm), \
             patch("motion.ball_pickup.Camera.from_robot"), \
             patch("motion.ball_pickup.Gripper.from_robot", return_value=gripper), \
             patch("motion.ball_pickup.MotionClient.from_robot", return_value=motion), \
             patch("motion.ball_pickup.collect_track", AsyncMock(side_effect=[track, changed if shift else track, track])):
            try:
                result = await move_to_ball_and_grip(object(), config, execute)
            except (RuntimeError, ValueError):
                result = None
        return result, arm, gripper, motion

    async def test_dry_run_never_moves_or_grips(self):
        result, _, gripper, motion = await self.run_pickup(False)
        self.assertFalse(result["executed"])
        motion.move.assert_not_awaited()
        gripper.open.assert_not_awaited()
        gripper.grab.assert_not_awaited()

    async def test_successful_pickup(self):
        result, _, gripper, motion = await self.run_pickup(True)
        self.assertTrue(result["grabbed"])
        self.assertEqual(motion.move.await_count, 2)
        self.assertEqual(motion.move.call_args.kwargs["component_name"], "gripper")
        gripper.grab.assert_awaited_once()

    async def test_failed_motion_stops_and_does_not_grip(self):
        result, arm, gripper, _ = await self.run_pickup(True, move_success=False)
        self.assertIsNone(result)
        arm.stop.assert_awaited_once()
        gripper.grab.assert_not_awaited()

    async def test_target_shift_aborts_before_descent(self):
        result, arm, gripper, motion = await self.run_pickup(True, shift=True)
        self.assertIsNone(result)
        self.assertEqual(motion.move.await_count, 1)
        gripper.grab.assert_not_awaited()
        arm.stop.assert_awaited_once()

    async def test_empty_grasp_is_failure(self):
        result, arm, _, _ = await self.run_pickup(True, grabbed=False)
        self.assertIsNone(result)
        arm.stop.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
