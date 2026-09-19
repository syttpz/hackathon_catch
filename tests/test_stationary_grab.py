import asyncio
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from viam.proto.common import Geometry, GeometriesInFrame, PointCloudObject, Pose, PoseInFrame

from motion.viam_stationary_grab import DEFAULT_CONFIG, _observe, make_targets, run_stationary_grab, validate_config
from vision.viam_ball import BallDetection, BallPose


def config():
    return json.loads(DEFAULT_CONFIG.read_text())


BALL = BallPose(197.6, -818.8, -5.958684503177324, "world", "ball", 100)
DETECTION = BallDetection("ball", 1, 720, 426, 801, 511)


def tcp_to_ball_z(cfg):
    return cfg["grasp_z_world_mm"] - cfg["expected_ball_z_world_mm"]


class TargetTests(unittest.TestCase):
    def test_taught_offset_keeps_table_grasp_at_taught_tcp(self):
        cfg = config()
        targets = make_targets(BALL, cfg)
        self.assertEqual(targets["grasp"][:2], [197.6, -818.8])
        self.assertAlmostEqual(targets["grasp"][2], cfg["grasp_z_world_mm"])
        self.assertEqual(targets["approach"][2], targets["grasp"][2]+100)
        self.assertEqual(targets["lift"][2], targets["grasp"][2]+125)

    def test_wrong_frame_outside_workspace_and_invalid_pose_fail(self):
        for ball in (BallPose(0, 0, -6, "cam", "ball", 1),
                     BallPose(999, -818, -6, "world", "ball", 1),
                     BallPose(198, -818, 100, "world", "ball", 1),
                     BallPose(float("nan"), -818, -6, "world", "ball", 1)):
            with self.assertRaises(ValueError):
                make_targets(ball, config())

    def test_grasp_z_follows_detected_ball_height(self):
        cfg = config()
        offset = tcp_to_ball_z(cfg)
        for z in (-5.96, 9.99, 40.0):
            targets = make_targets(BallPose(218.8, -651.2, z, "world", "ball", 1), cfg)
            self.assertAlmostEqual(targets["grasp"][2], z + offset)
            self.assertEqual(targets["grasp"][:2], [218.8, -651.2])
            self.assertEqual(targets["approach"][2], targets["grasp"][2]+100)

    def test_invalid_configuration_rejected(self):
        for key, value in (("move_timeout_s", float("nan")), ("settle_s", -1),
                           ("grasp_z_world_mm", float("inf")), ("grasp_orientation", [0, 0, 1, 0])):
            cfg = config()
            cfg[key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)


class CycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = config()
        self.events = []
        self.pose = Pose(x=228, y=-822, z=self.cfg["grasp_z_world_mm"], o_z=-1)
        self.arm = SimpleNamespace(is_moving=AsyncMock(return_value=False), stop=AsyncMock())

        async def move(**kwargs):
            self.events.append(("move", kwargs))
            self.pose.CopyFrom(kwargs["destination"].pose)
            return True

        async def get_pose(*args, **kwargs):
            return PoseInFrame(reference_frame="world", pose=self.pose)

        async def open_gripper(**kwargs):
            self.events.append(("open", None))

        async def grab(**kwargs):
            self.events.append(("grab", None))
            return True

        self.motion = SimpleNamespace(move=AsyncMock(side_effect=move), get_pose=AsyncMock(side_effect=get_pose))
        self.gripper = SimpleNamespace(open=AsyncMock(side_effect=open_gripper),
                                       grab=AsyncMock(side_effect=grab), stop=AsyncMock())
        self.observe = AsyncMock(return_value=(BALL, DETECTION))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("Arm.from_robot", self.arm), ("Gripper.from_robot", self.gripper),
                            ("MotionClient.from_robot", self.motion)):
            self.stack.enter_context(patch("motion.viam_stationary_grab."+name, return_value=value))
        self.stack.enter_context(patch("motion.viam_stationary_grab._observe", self.observe))

    async def test_preview_has_no_actuator_calls(self):
        result = await run_stationary_grab(object(), self.cfg)
        self.assertTrue(result["success"])
        self.assertFalse(result["executed"])
        self.assertEqual(result["state"], "preview")
        self.assertEqual(self.events, [])
        self.arm.stop.assert_not_awaited()
        self.gripper.stop.assert_not_awaited()

    async def test_preview_allows_gripper_outside_table_workspace(self):
        self.pose = Pose(x=105.6, y=-178.9, z=525.4, o_z=-1)
        result = await run_stationary_grab(object(), self.cfg)
        self.assertTrue(result["success"])
        self.assertEqual(result["state"], "preview")
        self.assertEqual(result["initial_gripper_world_mm"], [105.6, -178.9, 525.4])
        self.assertEqual(result["targets_world_mm"]["grasp"][:2], [BALL.x, BALL.y])
        self.assertEqual(self.events, [])

    async def test_cycle_moves_tcp_in_world_and_only_lifts_after_grab(self):
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertEqual(result["state"], "ball_grabbed")
        self.assertTrue(result["grabbed"])
        events = [name for name, _ in self.events]
        self.assertEqual(events, ["move", "move", "open", "move", "grab", "move"])
        self.observe.assert_awaited_once()
        moves = [kwargs for name, kwargs in self.events if name == "move"]
        for move in moves:
            self.assertEqual(move["component_name"], "gripper")
            self.assertEqual(move["destination"].reference_frame, "world")
        self.assertAlmostEqual(moves[-2]["destination"].pose.z, self.cfg["grasp_z_world_mm"])
        self.assertTrue(moves[-2]["constraints"].linear_constraint)
        self.assertTrue(moves[-1]["constraints"].linear_constraint)

    async def test_approach_only_never_opens_or_grabs(self):
        result = await run_stationary_grab(object(), self.cfg, execute=True, approach_only=True)
        self.assertEqual(result["state"], "approached")
        self.assertEqual(self.motion.move.await_count, 2)
        self.observe.assert_awaited_once()
        self.assertEqual(result["targets_world_mm"]["approach"][:2], [BALL.x, BALL.y])
        self.gripper.open.assert_not_awaited()
        self.gripper.grab.assert_not_awaited()

    async def test_approach_only_uses_initial_pose_if_ball_leaves(self):
        self.observe.side_effect = [(BALL, DETECTION), ValueError("No ball")]
        result = await run_stationary_grab(object(), self.cfg, execute=True, approach_only=True)
        self.assertEqual(result["state"], "approached")
        self.observe.assert_awaited_once()
        self.assertEqual(result["targets_world_mm"]["approach"][:2], [BALL.x, BALL.y])

    async def test_no_ball_is_read_only_failure(self):
        self.observe.side_effect = ValueError("No ball")
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertFalse(result["success"])
        self.assertFalse(result["executed"])
        self.assertEqual(self.events, [])

    async def test_lost_ball_after_approach_still_grabs_initial_target(self):
        self.observe.side_effect = [(BALL, DETECTION), ValueError("No ball")]
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertEqual(result["state"], "ball_grabbed")
        self.assertTrue(result["grabbed"])
        self.observe.assert_awaited_once()
        self.assertEqual(result["targets_world_mm"]["grasp"][:2], [BALL.x, BALL.y])
        self.gripper.grab.assert_awaited_once()

    async def test_later_ball_motion_is_ignored_after_initial_fix(self):
        moved = BallPose(BALL.x+30, BALL.y, BALL.z, "world", "ball", 100)
        self.observe.side_effect = [(BALL, DETECTION), (moved, DETECTION)]
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertEqual(result["state"], "ball_grabbed")
        self.observe.assert_awaited_once()
        self.assertEqual(result["targets_world_mm"]["grasp"][:2], [BALL.x, BALL.y])

    async def test_failed_grasp_never_lifts(self):
        self.gripper.grab.side_effect = None
        self.gripper.grab.return_value = False
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["state"], "grab")
        self.assertEqual(self.motion.move.await_count, 3)
        self.arm.stop.assert_awaited_once()

    async def test_motion_failure_and_timeout_stop_hardware(self):
        self.motion.move.side_effect = TimeoutError("motion timeout")
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertFalse(result["success"])
        self.arm.stop.assert_awaited_once()
        self.gripper.stop.assert_awaited_once()
        self.gripper.grab.assert_not_awaited()

    async def test_cancelled_motion_stops_and_propagates_cancellation(self):
        self.motion.move.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await run_stationary_grab(object(), self.cfg, execute=True)
        self.arm.stop.assert_awaited_once()
        self.gripper.stop.assert_awaited_once()


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    async def observe(self, positions, joints=None, count=1):
        cfg = config()
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False), get_joint_positions=AsyncMock(
            side_effect=[SimpleNamespace(values=j) for j in (joints or [[0]*6, [0]*6])]))
        objects = [PointCloudObject(point_cloud=b"cloud", geometries=GeometriesInFrame(
            reference_frame="cam", geometries=[Geometry(label="ball", center=Pose(x=10, y=20, z=700))]))]*count
        service = SimpleNamespace(get_object_point_clouds=AsyncMock(return_value=objects))
        machine = SimpleNamespace(transform_pose=AsyncMock(side_effect=[PoseInFrame(
            reference_frame="world", pose=Pose(x=x, y=-818, z=-6)) for x in positions]))
        with patch("motion.viam_stationary_grab.VisionClient.from_robot", return_value=service), \
             patch("motion.viam_stationary_grab.detect_ball", AsyncMock(return_value=DETECTION)), \
             patch("motion.viam_stationary_grab.asyncio.sleep", AsyncMock()):
            return await _observe(machine, arm, cfg)

    async def test_stationary_observation_uses_world_transform(self):
        ball, _ = await self.observe([198])
        self.assertEqual((ball.x, ball.reference_frame), (198, "world"))

    async def test_moving_wrist_rejected(self):
        with self.assertRaisesRegex(ValueError, "Arm moved"):
            await self.observe([198], joints=[[0]*6, [1]*6])

    async def test_multiple_balls_not_arbitrarily_selected(self):
        with self.assertRaisesRegex(ValueError, "found 2"):
            await self.observe([198], count=2)

    async def test_slow_localization_reports_elapsed_time(self):
        calls = {"n": 0}

        def monotonic():
            calls["n"] += 1
            return 0.0 if calls["n"] == 1 else 26.0

        with patch("motion.viam_stationary_grab.time.monotonic", side_effect=monotonic):
            with self.assertRaisesRegex(ValueError, r"elapsed=26.00 s, limit=25.00 s"):
                await self.observe([198])


if __name__ == "__main__":
    unittest.main()
