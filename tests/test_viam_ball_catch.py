import asyncio
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from viam.proto.common import Geometry, GeometriesInFrame, PointCloudObject, Pose, PoseInFrame

from motion.viam_ball_catch import (
    DEFAULT_CONFIG, Sample, _fatal_locate_error, at_rest, catch_target, choose_delay,
    fit_velocity, left_rest, locate_world, move_duration, pick_tracked_ball, predict_position,
    run_ball_catch, sample_speed, validate_config,
)
from vision.viam_ball import BallDetection, BallPose


def config():
    return json.loads(DEFAULT_CONFIG.read_text())


DETECTION = BallDetection("ball", 1, 720, 426, 801, 511)


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def now(self):
        return self.t

    async def sleep(self, seconds):
        self.t += float(seconds)


class ModelTests(unittest.TestCase):
    def test_rest_then_leave_rest(self):
        cfg = config()
        rest = [Sample(i * 0.12, (200, -800, -6)) for i in range(3)]
        self.assertTrue(at_rest(rest, cfg))
        self.assertFalse(left_rest(rest, rest[-1].xyz, cfg))
        moving = rest + [Sample(0.36, (200, -780, -6))]
        self.assertTrue(left_rest(moving, rest[-1].xyz, cfg))
        slow = rest + [Sample(1.36, (200, -782, -6))]
        self.assertTrue(left_rest(slow, rest[-1].xyz, cfg))
        nudge = rest + [Sample(1.36, (200, -794, -6))]
        self.assertFalse(left_rest(nudge, rest[-1].xyz, cfg))

    def test_velocity_and_intercept_match_constant_motion(self):
        samples = [Sample(0.0, (200, -820, -6)), Sample(0.12, (200, -796, -6)),
                   Sample(0.24, (200, -772, -6))]
        position, velocity, residual = fit_velocity(samples, [0, 0, 0])
        self.assertAlmostEqual(position[1], -772, places=5)
        self.assertAlmostEqual(velocity[1], 200, places=5)
        self.assertLess(residual, 1e-6)
        predicted = predict_position(position, velocity, [0, 0, 0], 0.5)
        self.assertAlmostEqual(predicted[1], -672, places=5)
        target = catch_target(predicted, config())
        self.assertAlmostEqual(target[0], 200)
        self.assertAlmostEqual(target[1], -672)
        self.assertAlmostEqual(target[2], predicted[2] + config()["grasp_z_world_mm"]
                               - config()["expected_ball_z_world_mm"])

    def test_tossed_ball_follows_predicted_z(self):
        cfg = config()
        predicted = [200, -700, 40]
        target = catch_target(predicted, cfg)
        self.assertAlmostEqual(target[2], 40 + cfg["grasp_z_world_mm"] - cfg["expected_ball_z_world_mm"])
        self.assertEqual(target[:2], [200, -700])

    def test_choose_delay_falls_back_when_default_leaves_workspace(self):
        cfg = config()
        # 0.5 s would put Y at -820+400= -420, outside the table box.
        delay, predicted, target = choose_delay([200, -820, -6], [0, 800, 0], [0, 0, 0], cfg)
        self.assertLess(delay, cfg["catch_delay_s"])
        self.assertTrue(cfg["workspace_mm"][0][1] <= target[1] <= cfg["workspace_mm"][1][1])
        self.assertAlmostEqual(predicted[1], -820 + 800 * delay, places=4)

    def test_move_duration_and_invalid_config(self):
        self.assertAlmostEqual(move_duration([0, 0, 0], [250, 0, 0], config()), 1.0)
        cfg = config()
        cfg["catch_delay_s"] = 2
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_pick_tracked_ball_prefers_nearest_not_largest(self):
        cfg = config()
        ball = BallPose(107, -881, -5, "world", "ball", 40)
        hand = BallPose(-5, -999, 4, "world", "ball", 200)
        chosen = pick_tracked_ball([hand, ball], (107, -881, -5), cfg)
        self.assertEqual((chosen.x, chosen.y), (107, -881))
        first = pick_tracked_ball([hand, ball], None, cfg)
        self.assertEqual((first.x, first.y), (107, -881))

    def test_pick_tracked_ball_rejects_teleport(self):
        cfg = config()
        with self.assertRaisesRegex(ValueError, "jumped"):
            pick_tracked_ball(
                [BallPose(-228, -575, -222, "world", "ball", 40),
                 BallPose(-200, -500, -200, "world", "ball", 10)],
                (107, -881, -5), cfg)

    def test_single_far_segment_is_treated_as_motion(self):
        cfg = config()
        far = BallPose(75, -709, 4, "world", "ball", 40)
        chosen = pick_tracked_ball([far], (112, -960, -9), cfg)
        self.assertEqual((round(chosen.x), round(chosen.y)), (75, -709))

    def test_connection_lost_is_fatal(self):
        self.assertTrue(_fatal_locate_error(FileNotFoundError("No such file or directory")))
        self.assertTrue(_fatal_locate_error(RuntimeError("Connection lost")))
        self.assertFalse(_fatal_locate_error(ValueError("Expected a ball segment; found 0")))


class CycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = config()
        self.clock = Clock()
        self.pose = Pose(x=228, y=-822, z=self.cfg["grasp_z_world_mm"], o_z=-1)
        self.events = []
        self.arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                                   get_joint_positions=AsyncMock(return_value=SimpleNamespace(values=[0]*6)),
                                   stop=AsyncMock())
        self.gripper = SimpleNamespace(open=AsyncMock(), grab=AsyncMock(return_value=True), stop=AsyncMock())

        async def move(**kwargs):
            self.events.append(("move", kwargs))
            self.pose.CopyFrom(kwargs["destination"].pose)
            return True

        self.motion = SimpleNamespace(move=AsyncMock(side_effect=move),
                                      get_pose=AsyncMock(side_effect=lambda *a, **k: PoseInFrame(
                                          reference_frame="world", pose=self.pose)))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("Arm.from_robot", self.arm), ("Gripper.from_robot", self.gripper),
                            ("MotionClient.from_robot", self.motion),
                            ("VisionClient.from_robot", SimpleNamespace())):
            self.stack.enter_context(patch("motion.viam_ball_catch."+name, return_value=value))

    def locator(self, xyzs):
        calls = {"n": 0}

        async def locate(machine, arm, config, segmenter, settle=False, **kwargs):
            xyz = xyzs[min(calls["n"], len(xyzs) - 1)]
            calls["n"] += 1
            return BallPose(*xyz, "world", "ball", 10), DETECTION

        locate.calls = calls
        return locate

    async def run_catch(self, xyzs, execute=False):
        return await run_ball_catch(object(), self.cfg, execute=execute, locate=self.locator(xyzs),
                                    now=self.clock.now, sleep=self.clock.sleep)

    async def test_preview_waits_for_leave_rest_and_does_not_move(self):
        rest = [(200, -800, -6)] * 3
        moving = [(200, -780, -6), (200, -756, -6), (200, -732, -6)]
        result = await self.run_catch(rest + moving)
        self.assertTrue(result["success"])
        self.assertEqual(result["state"], "preview")
        self.assertFalse(result["executed"])
        self.assertEqual(self.events, [])
        self.gripper.open.assert_not_awaited()
        self.assertAlmostEqual(result["velocity_mm_s"][1], 24 / self.cfg["sample_period_s"], delta=1)
        self.assertEqual(result["rest_world_mm"], [200.0, -800.0, -6.0])

    async def test_rest_stays_latched_after_slow_follow_through(self):
        rest = [(200, -800, -6)] * 3
        follow = [(200, -780, -6), (200, -778, -6), (200, -777, -6)]
        result = await self.run_catch(rest + follow)
        self.assertTrue(result["success"])
        self.assertEqual(result["rest_world_mm"], [200.0, -800.0, -6.0])
        self.assertEqual(len(result["samples"]), 6)
        self.assertAlmostEqual(
            result["catch_target_world_mm"][2],
            result["predicted_ball_world_mm"][2] + self.cfg["grasp_z_world_mm"]
            - self.cfg["expected_ball_z_world_mm"])

    async def test_execute_opens_then_intercepts_initial_fit(self):
        rest = [(200, -800, -6)] * 3
        moving = [(200, -780, -6), (200, -756, -6), (200, -732, -6)]
        result = await self.run_catch(rest + moving, execute=True)
        self.assertEqual(result["state"], "caught")
        self.assertTrue(result["grabbed"])
        self.gripper.open.assert_awaited_once()
        self.assertEqual(len(self.events), 1)
        pose = self.events[0][1]["destination"].pose
        self.assertAlmostEqual(pose.x, result["catch_target_world_mm"][0], places=4)
        self.assertAlmostEqual(pose.y, result["catch_target_world_mm"][1], places=4)
        self.assertAlmostEqual(pose.z, result["catch_target_world_mm"][2], places=4)

    async def test_already_moving_ball_still_gets_an_intercept(self):
        moving = [(200, -800, -6), (200, -790, -6), (200, -780, -6)]
        result = await self.run_catch(moving)
        self.assertTrue(result["success"])
        self.assertIsNone(result["rest_world_mm"])
        self.assertAlmostEqual(result["velocity_mm_s"][1], 10 / self.cfg["sample_period_s"], delta=1)

    async def test_timeout_without_motion(self):
        self.cfg["track_timeout_s"] = 0.2
        result = await self.run_catch([(200, -800, -6)] * 20)
        self.assertFalse(result["success"])
        self.assertIn("Timed out", result["reason"])

    async def test_connection_lost_stops_instead_of_spinning(self):
        async def locate(*args, **kwargs):
            raise RuntimeError("Connection lost")

        result = await run_ball_catch(object(), self.cfg, locate=locate,
                                      now=self.clock.now, sleep=self.clock.sleep)
        self.assertFalse(result["success"])
        self.assertIn("Lost connection", result["reason"])
        self.assertLess(self.clock.t, 1.0)

    async def test_failed_first_locate_does_not_resettle(self):
        settles = []

        async def locate(machine, arm, config, segmenter, settle=False, **kwargs):
            settles.append(settle)
            if len(settles) == 1:
                raise ValueError("transient")
            return BallPose(200, -800, -6, "world", "ball", 10), DETECTION

        self.cfg["track_timeout_s"] = 0.2
        await run_ball_catch(object(), self.cfg, locate=locate,
                             now=self.clock.now, sleep=self.clock.sleep)
        self.assertEqual(settles[0], True)
        self.assertGreater(len(settles), 1)
        self.assertFalse(any(settles[1:]))

    async def test_lost_ball_after_two_motion_samples_still_predicts(self):
        calls = {"n": 0}
        rest = [(200, -800, -6)] * 3
        moving = [(200, -780, -6), (200, -756, -6)]

        async def locate(machine, arm, config, segmenter, settle=False, **kwargs):
            n = calls["n"]
            calls["n"] += 1
            if n >= 5:
                raise ValueError("No 'ball' detection")
            xyz = (rest + moving)[n]
            return BallPose(*xyz, "world", "ball", 10), DETECTION

        result = await run_ball_catch(object(), self.cfg, locate=locate,
                                      now=self.clock.now, sleep=self.clock.sleep)
        self.assertTrue(result["success"])
        self.assertEqual(result["state"], "preview")
        self.assertAlmostEqual(result["velocity_mm_s"][1], 24 / self.cfg["sample_period_s"], delta=5)


class LocateTests(unittest.IsolatedAsyncioTestCase):
    async def test_locate_world_keeps_largest_of_multiple_balls(self):
        cfg = config()
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              get_joint_positions=AsyncMock(return_value=SimpleNamespace(values=[0]*6)))
        objects = [
            PointCloudObject(point_cloud=b"small", geometries=GeometriesInFrame(
                reference_frame="cam", geometries=[Geometry(label="ball", center=Pose(x=1, y=2, z=3))])),
            PointCloudObject(point_cloud=b"largest-cloud", geometries=GeometriesInFrame(
                reference_frame="cam", geometries=[Geometry(label="ball", center=Pose(x=10, y=20, z=30))])),
        ]
        segmenter = SimpleNamespace(get_object_point_clouds=AsyncMock(return_value=objects))
        worlds = {
            1: Pose(x=999, y=-999, z=0),
            10: Pose(x=200, y=-800, z=-6),
        }

        async def transform(pose_in_frame, frame):
            pose = worlds[round(pose_in_frame.pose.x)]
            return PoseInFrame(reference_frame="world", pose=pose)

        machine = SimpleNamespace(transform_pose=AsyncMock(side_effect=transform))
        ball, _ = await locate_world(machine, arm, cfg, segmenter)
        self.assertEqual((ball.x, ball.y, ball.z), (200, -800, -6))
        self.assertEqual(machine.transform_pose.await_count, 2)

    async def test_locate_world_keeps_nearest_to_previous(self):
        cfg = config()
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              get_joint_positions=AsyncMock(return_value=SimpleNamespace(values=[0]*6)))
        objects = [
            PointCloudObject(point_cloud=b"hand-is-huge"*20, geometries=GeometriesInFrame(
                reference_frame="cam", geometries=[Geometry(label="ball", center=Pose(x=1, y=2, z=3))])),
            PointCloudObject(point_cloud=b"ball", geometries=GeometriesInFrame(
                reference_frame="cam", geometries=[Geometry(label="ball", center=Pose(x=10, y=20, z=30))])),
        ]
        segmenter = SimpleNamespace(get_object_point_clouds=AsyncMock(return_value=objects))

        async def transform(pose_in_frame, frame):
            if round(pose_in_frame.pose.x) == 1:
                pose = Pose(x=-5, y=-999, z=4)
            else:
                pose = Pose(x=107, y=-881, z=-5)
            return PoseInFrame(reference_frame="world", pose=pose)

        machine = SimpleNamespace(transform_pose=AsyncMock(side_effect=transform))
        ball, _ = await locate_world(machine, arm, cfg, segmenter, previous_xyz=(107, -881, -5))
        self.assertEqual((round(ball.x), round(ball.y)), (107, -881))


if __name__ == "__main__":
    unittest.main()
