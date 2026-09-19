import json
from pathlib import Path
from types import SimpleNamespace as NS
import asyncio
import unittest
from unittest.mock import AsyncMock
import numpy as np
import cv2
from viam.proto.common import Pose
from motion.live_camera_pose import pose_matrix
from motion.track_red import (locate_target, estimate_distance, servo_step,
                             camera_tilt_deg, clamp_target, in_workspace,
                             red_blob_candidates, SmoothMove)

CONFIG = json.loads((Path(__file__).parents[1]/'track_red.config.json').read_text())
K = NS(fx=600, fy=600, cx=424, cy=240, width=848, height=480)
BALL = {**CONFIG, 'detector': 'ball'}


def red_frame(circles):
    bgr = np.full((K.height, K.width, 3), 40, np.uint8)
    for (u, v), r in circles:
        cv2.circle(bgr, (int(u), int(v)), int(r), (30, 30, 230), -1)
    return bgr


class DetectionTests(unittest.TestCase):
    def test_single_ball_is_located_near_its_drawn_centre(self):
        center, radius, physical, reason = locate_target(red_frame([((500, 200), 25)]), BALL)
        self.assertIsNone(reason)
        np.testing.assert_allclose(center, [500, 200], atol=2)
        self.assertAlmostEqual(radius, 25, delta=3)
        self.assertEqual(physical, CONFIG['ball_radius_mm'])

    def test_empty_and_undersized_frames_report_no_ball(self):
        for circles in ([], [((500, 200), 3)]):
            center, _, _, reason = locate_target(red_frame(circles), BALL)
            self.assertIsNone(center)
            self.assertIn('NO BALL', reason)

    def test_two_similar_circles_are_ambiguous_but_a_dominant_one_wins(self):
        center, _, _, reason = locate_target(red_frame([((300, 200), 25), ((600, 200), 24)]), BALL)
        self.assertIsNone(center)
        self.assertIn('AMBIGUOUS', reason)
        center, radius, _, reason = locate_target(red_frame([((300, 200), 40), ((600, 200), 12)]), BALL)
        self.assertIsNone(reason)
        np.testing.assert_allclose(center, [300, 200], atol=2)

    def test_can_mode_measures_width_against_the_can_diameter(self):
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        cv2.rectangle(bgr, (600, 240), (672, 360), (30, 30, 230), -1)
        config = {**CONFIG, 'detector': 'can'}
        center, width, physical, reason = locate_target(bgr, config)
        self.assertIsNone(reason)
        self.assertAlmostEqual(width, 72, delta=3)
        self.assertEqual(physical, CONFIG['can_diameter_mm'])
        np.testing.assert_allclose(center, [636, 300], atol=3)

    def test_an_elongated_red_blob_is_rejected_as_a_ball(self):
        # The live Coke can measured circularity 0.46 against the 0.60 ball floor.
        # A clean rectangle scores higher, so this uses an aspect ratio that falls
        # below it either way; can mode still accepts the same blob.
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        cv2.rectangle(bgr, (600, 150), (660, 430), (30, 30, 230), -1)
        self.assertIsNone(locate_target(bgr, BALL)[0])
        self.assertIsNotNone(locate_target(bgr, {**CONFIG, 'detector': 'can'})[0])


class DistanceTests(unittest.TestCase):
    def test_projected_radius_inverts_to_range(self):
        self.assertAlmostEqual(estimate_distance(60, K, 30), 300)
        self.assertAlmostEqual(estimate_distance(30, K, 30), 600)
        with self.assertRaises(ValueError):
            estimate_distance(0, K, 30)


class ServoTests(unittest.TestCase):
    def setUp(self):
        self.flange = pose_matrix(Pose(x=238, y=-53, z=400, o_y=-1))
        self.camera = np.eye(3)  # camera X/Y/Z aligned with world for readable assertions

    def step(self, center, radius_px=60, **overrides):
        config = {**CONFIG, **overrides}
        distance = estimate_distance(radius_px, K, config['ball_radius_mm'])
        return servo_step(np.asarray(center, float), distance, self.flange, self.camera, K, config)

    def test_ball_right_of_centre_moves_the_flange_along_camera_x(self):
        target, info = self.step((K.cx+100, K.cy))
        self.assertIsNotNone(target)
        self.assertGreater(target.x, self.flange[0][0])
        self.assertAlmostEqual(target.y, self.flange[0][1], places=6)
        self.assertAlmostEqual(info.distance, 300)

    def test_ball_below_centre_moves_the_flange_along_camera_y(self):
        target, _ = self.step((K.cx, K.cy+100))
        self.assertGreater(target.y, self.flange[0][1])
        self.assertAlmostEqual(target.x, self.flange[0][0], places=6)

    def test_centred_ball_at_standoff_holds_position(self):
        radius = (K.fx+K.fy)*.5*CONFIG['ball_radius_mm']/CONFIG['standoff_mm']
        target, reason = self.step((K.cx, K.cy), radius_px=radius)
        self.assertIsNone(target)
        self.assertIn('HOLD', reason)

    def test_step_is_bounded_and_clipped_to_the_workspace(self):
        target, info = self.step((K.cx+400, K.cy+300))
        self.assertLessEqual(info.step, CONFIG['max_step_mm']+1e-6)
        lower, upper = np.array(CONFIG['workspace_mm'], float)
        p, _ = pose_matrix(target)
        self.assertTrue(((p >= lower) & (p <= upper)).all())

    def test_orientation_is_passed_through_unchanged(self):
        target, _ = self.step((K.cx+100, K.cy))
        np.testing.assert_allclose(pose_matrix(target)[1], self.flange[1], atol=1e-9)

    def test_range_outside_limits_is_rejected(self):
        target, reason = self.step((K.cx+100, K.cy), radius_px=300)
        self.assertIsNone(target)
        self.assertIn('OUT OF RANGE', reason)

    def test_standoff_is_off_by_default_so_the_arm_never_drives_at_the_operator(self):
        self.assertEqual(CONFIG['forward_gain'], 0)
        far = (K.fx+K.fy)*.5*CONFIG['ball_radius_mm']/700
        target, reason = self.step((K.cx, K.cy), radius_px=far)
        self.assertIsNone(target)
        self.assertIn('HOLD', reason)

    def test_standoff_pulls_the_camera_forward_when_enabled(self):
        far = (K.fx+K.fy)*.5*CONFIG['ball_radius_mm']/700
        target, info = self.step((K.cx, K.cy), radius_px=far, forward_gain=.35)
        self.assertAlmostEqual(info.distance, 700)
        self.assertGreater(target.z, self.flange[0][2])  # camera Z is world Z here

    def test_ball_swung_up_and_down_moves_the_flange_the_same_way(self):
        # Camera aimed horizontally along world -Y, image Y (down) mapped to world -Z.
        level = np.array([[1., 0, 0], [0, 0, -1.], [0, -1., 0]])
        self.assertAlmostEqual(camera_tilt_deg(level), 0, places=6)
        distance = estimate_distance(60, K, CONFIG['ball_radius_mm'])
        up, _ = servo_step(np.array([K.cx, K.cy-120.]), distance, self.flange, level, K, CONFIG)
        down, _ = servo_step(np.array([K.cx, K.cy+120.]), distance, self.flange, level, K, CONFIG)
        self.assertGreater(up.z, self.flange[0][2])
        self.assertLess(down.z, self.flange[0][2])
        for pose in (up, down):
            np.testing.assert_allclose(pose_matrix(pose)[1], self.flange[1], atol=1e-9)


class TiltTests(unittest.TestCase):
    def test_tilt_is_signed_and_zero_when_forward_axis_is_horizontal(self):
        level = np.array([[1., 0, 0], [0, 0, -1.], [0, -1., 0]])
        self.assertAlmostEqual(camera_tilt_deg(level), 0, places=6)
        down = np.array([[1., 0, 0], [0, -1., 0], [0, 0, -1.]])
        self.assertAlmostEqual(camera_tilt_deg(down), -90, places=6)
        self.assertAlmostEqual(camera_tilt_deg(np.eye(3)), 90, places=6)


if __name__ == '__main__':
    unittest.main()


class WorkspaceTests(unittest.TestCase):
    """Bounds come from the configured obstacles: table top Z=-38, front wall X>=690."""

    def test_configured_box_clears_the_table_and_the_front_wall(self):
        lower, upper = np.array(CONFIG['workspace_mm'], float)
        self.assertGreater(lower[2], -38)      # above the table top
        self.assertLess(upper[0], 690)         # clear of the front wall

    def test_known_taught_poses_are_inside(self):
        for pose in ([238.1, -53.1, 223.3], [-58, -172, 690], [-573.5, 112.4, 364.7]):
            self.assertTrue(in_workspace(np.array(pose, float), CONFIG), pose)

    def test_far_target_is_pulled_back_inside_the_reach_sphere(self):
        goal = clamp_target(np.array([-690., 400., 600.]), CONFIG)
        self.assertLessEqual(np.linalg.norm(goal), CONFIG['max_reach_mm']+1e-6)
        self.assertTrue(in_workspace(goal, CONFIG))

    def test_target_beyond_the_wall_is_clipped_in_x(self):
        goal = clamp_target(np.array([900., 0., 400.]), CONFIG)
        self.assertLessEqual(goal[0], CONFIG['workspace_mm'][1][0]+1e-6)
        self.assertLess(goal[0], 690)

    def test_target_below_the_table_is_lifted_above_it(self):
        goal = clamp_target(np.array([300., 0., -200.]), CONFIG)
        self.assertGreaterEqual(goal[2], CONFIG['workspace_mm'][0][2]-1e-6)
        self.assertGreater(goal[2], -38)

    def test_target_on_the_base_is_rejected_rather_than_divided_by_zero(self):
        with self.assertRaises(ValueError):
            clamp_target(np.zeros(3), {**CONFIG, 'workspace_mm': [[-700, -700, -700], [640, 700, 700]]})


class BlobDetectorTests(unittest.TestCase):
    """The follower needs direction, so it keeps blobs the shape detectors reject."""

    def blob(self, bgr):
        return red_blob_candidates(bgr, CONFIG)

    def test_a_hand_occluded_crescent_is_kept(self):
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        cv2.circle(bgr, (500, 240), 40, (30, 30, 230), -1)
        cv2.circle(bgr, (470, 215), 34, (40, 40, 40), -1)   # a hand across the ball
        self.assertEqual(len(self.blob(bgr)), 1)
        center, _, _, reason = locate_target(bgr, CONFIG)
        self.assertIsNone(reason)
        np.testing.assert_allclose(center, [500, 240], atol=40)

    def test_a_duller_blurred_red_is_kept(self):
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        cv2.circle(bgr, (500, 240), 40, (60, 60, 150), -1)  # low saturation/value
        bgr = cv2.GaussianBlur(bgr, (21, 21), 0)
        self.assertTrue(self.blob(bgr))
        self.assertIsNone(locate_target(bgr, BALL)[0])
        self.assertIsNotNone(locate_target(bgr, CONFIG)[0])

    def test_a_blob_on_the_border_is_tracked_but_reports_no_range(self):
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        cv2.circle(bgr, (K.width-10, 240), 45, (30, 30, 230), -1)
        candidates = self.blob(bgr)
        self.assertTrue(candidates[0][2])                   # truncated
        center, size_px, physical_mm, reason = locate_target(bgr, CONFIG)
        self.assertIsNone(reason)
        self.assertIsNotNone(center)
        self.assertIsNone(physical_mm)                      # range withheld
        self.assertGreater(size_px, 0)

    def test_a_fully_visible_blob_reports_a_usable_range(self):
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        cv2.circle(bgr, (500, 240), 40, (30, 30, 230), -1)
        _, _, physical_mm, _ = locate_target(bgr, CONFIG)
        self.assertEqual(physical_mm, CONFIG['ball_radius_mm'])

    def test_speck_noise_is_still_rejected(self):
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        cv2.circle(bgr, (500, 240), 3, (30, 30, 230), -1)
        self.assertFalse(self.blob(bgr))


class SmoothMoveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.arm = AsyncMock()
        self.arm.is_moving.return_value = False
        self.motion = AsyncMock()
        self.events = []

    def hold(self):
        async def move(**kwargs):
            self.events.append('move')
            try:
                await asyncio.Event().wait()
            finally:
                self.events.append('cancel')
        self.motion.move.side_effect = move
        self.arm.stop.side_effect = lambda **kwargs: self.events.append('stop')

    async def test_an_inflight_move_is_never_cancelled_by_default(self):
        self.hold()
        mover = SmoothMove(self.arm, self.motion, 'arm', 'world')
        self.assertTrue(await mover.update(Pose(x=100, z=300, o_z=-1)))
        await asyncio.sleep(.01)
        self.assertFalse(await mover.update(Pose(x=400, z=300, o_z=-1)))
        self.assertEqual(self.events, ['move'])
        self.assertEqual(mover.moves, 1)
        await mover.stop()

    async def test_the_newest_target_is_commanded_once_the_arm_is_free(self):
        release = asyncio.Event()
        async def move(**kwargs):
            await release.wait()
            return True
        self.motion.move.side_effect = move
        mover = SmoothMove(self.arm, self.motion, 'arm', 'world')
        self.assertTrue(await mover.update(Pose(x=100, z=300, o_z=-1)))
        await asyncio.sleep(.01)
        # Both arrive while the first move is still running: neither commands,
        # and the newest overwrites the queued one.
        self.assertFalse(await mover.update(Pose(x=400, z=300, o_z=-1)))
        self.assertFalse(await mover.update(Pose(x=500, z=300, o_z=-1)))
        self.assertEqual(mover.moves, 1)
        release.set()
        await asyncio.sleep(.01)
        await mover.settle()
        self.assertTrue(await mover.submit())
        self.assertEqual(mover.moves, 2)
        np.testing.assert_allclose(mover.target, [500, 0, 300])
        release.set()
        await mover.stop()

    async def test_preempt_cancels_only_past_its_threshold(self):
        self.hold()
        mover = SmoothMove(self.arm, self.motion, 'arm', 'world', preempt_mm=100)
        await mover.update(Pose(x=100, z=300, o_z=-1))
        await asyncio.sleep(.01)
        self.assertFalse(await mover.update(Pose(x=150, z=300, o_z=-1)))
        self.assertEqual(self.events, ['move'])
        self.assertTrue(await mover.update(Pose(x=400, z=300, o_z=-1)))
        await asyncio.sleep(.01)
        self.assertEqual(self.events, ['move', 'cancel', 'stop', 'move'])
        await mover.stop()

    async def test_move_failures_are_counted_not_raised(self):
        self.motion.move.return_value = False
        mover = SmoothMove(self.arm, self.motion, 'arm', 'world')
        await mover.update(Pose(x=100, z=300, o_z=-1))
        await asyncio.sleep(.01)
        await mover.settle()
        self.assertEqual(mover.failures, 1)
        async def boom(**kwargs):
            raise asyncio.TimeoutError('planner timed out')
        self.motion.move.side_effect = boom
        await mover.update(Pose(x=200, z=300, o_z=-1))
        await asyncio.sleep(.01)
        await mover.settle()
        self.assertEqual(mover.failures, 2)
        await mover.stop()


class ContinuityTests(unittest.TestCase):
    """Several reds in view is normal with a permissive mask; stay on the same one."""

    def scene(self):
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        cv2.circle(bgr, (200, 240), 50, (30, 30, 230), -1)   # a larger distractor
        cv2.circle(bgr, (600, 240), 30, (30, 30, 230), -1)   # the followed ball
        return bgr

    def test_without_history_the_largest_red_wins(self):
        center, _, _, reason = locate_target(self.scene(), CONFIG)
        self.assertIsNone(reason)
        np.testing.assert_allclose(center, [200, 240], atol=4)

    def test_a_tracked_ball_is_kept_over_a_larger_distractor(self):
        center, _, _, reason = locate_target(self.scene(), CONFIG, previous=np.array([610., 245.]))
        self.assertIsNone(reason)
        np.testing.assert_allclose(center, [600, 240], atol=4)

    def test_several_reds_no_longer_abort_the_frame(self):
        bgr = np.full((K.height, K.width, 3), 40, np.uint8)
        for x in (120, 300, 480, 660):
            cv2.circle(bgr, (x, 240), 28, (30, 30, 230), -1)
        self.assertEqual(len(red_blob_candidates(bgr, CONFIG)), 4)
        self.assertIsNotNone(locate_target(bgr, CONFIG)[0])

    def test_a_jump_beyond_the_gate_falls_back_to_the_largest(self):
        center, _, _, _ = locate_target(self.scene(), CONFIG, previous=np.array([600., 240.])+10_000)
        np.testing.assert_allclose(center, [200, 240], atol=4)

    def test_the_gate_follows_a_ball_moving_at_a_plausible_speed(self):
        previous = np.array([600., 240.])
        for step in range(1, 4):                      # 60 px per frame at 60 Hz
            bgr = np.full((K.height, K.width, 3), 40, np.uint8)
            cv2.circle(bgr, (200, 240), 50, (30, 30, 230), -1)
            cv2.circle(bgr, (600-60*step, 240), 30, (30, 30, 230), -1)
            center, _, _, _ = locate_target(bgr, CONFIG, previous)
            np.testing.assert_allclose(center, [600-60*step, 240], atol=4)
            previous = center
