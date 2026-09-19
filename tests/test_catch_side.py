import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from motion.catch_side import SideTracker, load_calibration, moving_fit, new_fit, pixel_motion

CONFIG = dict(min_samples=4, max_samples=7, max_gap_s=.25,
              max_residual_mm=60, release_speed_mm_s=1200, min_span_s=.05,
              max_frame_age_s=.12, side_min_speed_mm_s=500)


class SideFlightTests(unittest.TestCase):
    def test_range_jitter_without_pixel_motion_is_not_a_throw(self):
        history = [(t, np.array([750.+np.sin(t*100), 300.]))
                   for t in np.arange(0, .12, 1/60)]
        self.assertFalse(pixel_motion(history, 0., 80.))
        moving = [(t, np.array([400.+600*t, 300.-100*t])) for t, _ in history]
        self.assertTrue(pixel_motion(moving, 0., 80.))

    def test_static_blob_does_not_create_flight(self):
        fit = new_fit(CONFIG)
        for t in np.arange(0, .5, 1/60):
            self.assertIsNone(moving_fit(fit, t, np.array([0., 0., 800.]), 500))

    def test_lob_predicts_descending_catch_plane(self):
        from motion.ballistic import plane_crossing
        fit = new_fit(CONFIG)
        for t in np.arange(0, .2, 1/60):
            p = np.array([-1000.+2000*t, 30., 800.+1000*t-4905*t*t])
            flight = moving_fit(fit, 10+t, p, 500)
        crossing = plane_crossing(flight, 320., 10.2)
        self.assertIsNotNone(crossing)
        when, point, velocity = crossing
        self.assertAlmostEqual(point[2], 320.)
        self.assertLess(velocity[2], 0)
        self.assertAlmostEqual(point[0], -1000+2000*(when-10))

    def test_stale_and_lost_observations_disable_prediction(self):
        tracker = SideTracker(None, CONFIG, None, None, None, None, None)
        marker = object()
        tracker.timestamp, tracker.flight = 10., marker
        self.assertIs(tracker.current(10.1), marker)
        self.assertIsNone(tracker.current(10.13))
        self.assertIsNone(tracker.current(9.9))
        tracker.invalidate('lost')
        self.assertIsNone(tracker.current(10.01))
        self.assertEqual(tracker.fit.samples, [])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.k = SimpleNamespace(fx=606., fy=606., cx=424., cy=240., width=848, height=480)
        points = np.random.default_rng(20).uniform([-300, -200, 900], [300, 200, 1600], (16, 3))
        pixels = points[:, :2]/points[:, 2:]*606+[424, 240]
        self.data = dict(camera='cam2', intrinsics=vars(self.k), samples=[
            dict(world=p.tolist(), pixel=u.tolist()) for p, u in zip(points, pixels)])

    def load(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'cal.json'
            p.write_text(json.dumps(self.data))
            return load_calibration(p, 'cam2', self.k)

    def test_valid_geometry_passes(self):
        origin, rotation, trusted, _ = self.load()
        self.assertTrue(trusted)
        np.testing.assert_allclose(origin, np.zeros(3), atol=.01)
        np.testing.assert_allclose(rotation, np.eye(3), atol=.001)

    def test_preview_offset_cannot_authorize_execution(self):
        self.data['preview_origin_offset_world_mm'] = [0., 0., -63.]
        origin, rotation, trusted, report = self.load()
        np.testing.assert_allclose(origin, [0., 0., -63.], atol=.01)
        self.assertFalse(trusted)
        self.assertIn('preview-only world offset', report)

    def test_invalid_preview_offset_rejected(self):
        self.data['preview_origin_offset_world_mm'] = [0., float('nan'), 0.]
        with self.assertRaisesRegex(ValueError, 'Invalid preview'):
            self.load()

    def test_mostly_rejected_calibration_is_preview_only(self):
        errors = np.array([1.]*5+[80.]*11)
        mask = errors < 3
        with patch('motion.catch_side.solve_side_pose', return_value=(
                np.eye(3), np.zeros(3), errors, mask)):
            self.assertFalse(self.load()[2])

    def test_changed_intrinsics_rejected(self):
        self.data['intrinsics'] = dict(vars(self.k), width=640)
        with self.assertRaisesRegex(ValueError, 'intrinsics changed'):
            self.load()

    def test_failed_independent_validation_blocks_execution(self):
        self.data['independent_validation'] = {'error_px': 6.6}
        self.assertFalse(self.load()[2])


class PollingTests(unittest.IsolatedAsyncioTestCase):
    async def test_untrusted_calibration_refuses_side_execution(self):
        from unittest.mock import AsyncMock
        from motion.catch_side import prepare_side
        camera = SimpleNamespace(get_properties=AsyncMock(return_value=SimpleNamespace(
            intrinsic_parameters=SimpleNamespace(focal_x_px=606., focal_y_px=606.,
                center_x_px=424., center_y_px=240., width_px=848, height_px=480))))
        args = SimpleNamespace(side_camera='cam2', side_calibration='unused.json',
                               execute=True, trajectory_source='side', color_source='color')
        with patch('viam.components.camera.Camera.from_robot', return_value=camera), patch(
                'motion.catch_side.load_calibration', return_value=(
                    np.zeros(3), np.eye(3), False, '5/12 inliers')):
            with self.assertRaisesRegex(ValueError, 'Side calibration quality failed'):
                await prepare_side(None, args, CONFIG, None, None)

    async def test_duplicate_frames_do_not_accumulate_and_task_cancels(self):
        class Camera:
            async def get_images(self, **kwargs):
                return [SimpleNamespace(name='color')], SimpleNamespace(
                    captured_at=SimpleNamespace(seconds=99, nanos=950_000_000))
        k = SimpleNamespace(width=2, height=2)
        tracker = SideTracker(Camera(), CONFIG, k, np.zeros(3), np.eye(3),
                              lambda *a: [(np.zeros(2), 10., 10.)],
                              lambda *a: (np.array([0., 0., 800.]), 800., None, None))
        with patch('motion.catch_side.time.time', return_value=100.), patch(
                'motion.catch_side.decode_color', return_value=np.zeros((2, 2, 3), np.uint8)):
            task = asyncio.create_task(tracker.run())
            await asyncio.sleep(.035)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(tracker.frames, 1)
        self.assertEqual(len(tracker.fit.samples), 1)
        self.assertTrue(task.cancelled())

class CatchLoopTests(unittest.IsolatedAsyncioTestCase):
    async def run_preview(self, source, wrist_detected):
        import contextlib
        import io
        import time
        from unittest.mock import AsyncMock, MagicMock
        from motion import catch_plane
        config_path = Path(__file__).parents[1]/'catch_plane.config.json'
        stamp = time.time()
        image = SimpleNamespace(name='color')
        meta = SimpleNamespace(captured_at=SimpleNamespace(
            seconds=int(stamp), nanos=int((stamp-int(stamp))*1e9)))
        camera = SimpleNamespace(
            get_images=AsyncMock(return_value=([image], meta)),
            get_properties=AsyncMock(return_value=SimpleNamespace(intrinsic_parameters=
                SimpleNamespace(focal_x_px=600, focal_y_px=600, center_x_px=424,
                                center_y_px=240, width_px=848, height_px=480))))
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              get_joint_positions=AsyncMock(return_value=SimpleNamespace(values=[0.]*6)),
                              stop=AsyncMock(), move_to_position=AsyncMock())
        robot = MagicMock()
        robot.__aenter__ = AsyncMock(return_value=robot)
        robot.__aexit__ = AsyncMock(return_value=False)
        class Side:
            sequence = 0
            frames = seen = 0
            status = 'fit ready'
            flight = object()
            def current(self, now):
                return self.flight
            async def run(self):
                while True:
                    self.sequence += 1
                    await asyncio.sleep(.003)
        side = Side()
        target = SimpleNamespace(point=np.array([400., 0., 320.]),
                                 arrival=time.monotonic()+.5, slack_s=.1)
        wrist_flight = SimpleNamespace(velocity=np.array([1500., 0., -500.]),
                                      samples=6, residual_mm=2.)
        gate = MagicMock()
        gate.add.return_value = True
        gate.progress.return_value = (3, 3)
        fit = MagicMock()
        fit.add.return_value = wrist_flight
        args = SimpleNamespace(config=config_path, no_side=False, trajectory_source=source,
                               execute=False, machine_config=None, range_source=None,
                               planned=False, color_source='color', depth_source='depth', duration=.05)
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            def mock(name, **kwargs):
                return stack.enter_context(patch.object(catch_plane, name, **kwargs))
            mock('credentials', return_value=('unused', 'unused'))
            client = mock('RobotClient')
            client.at_address = AsyncMock(return_value=robot)
            mock('Arm').from_robot.return_value = arm
            mock('Camera').from_robot.return_value = camera
            mock('MotionClient')
            mock('prepare_side', new=AsyncMock(return_value=side))
            mock('read_flange', new=AsyncMock(return_value=(np.array([400., 0., 300.]), np.eye(3))))
            mock('read_pose_of', new=AsyncMock(return_value=(np.array([400., 0., 300.]), np.eye(3))))
            mock('camera_transform', new=AsyncMock(return_value=(np.zeros(3), np.eye(3))))
            mock('decode_color', return_value=np.zeros((480, 848, 3), np.uint8))
            mock('green_ball', return_value=[(np.array([400., 200.]), 20., 20.)] if wrist_detected else [])
            mock('ball_point', return_value=(np.array([500., 0., 700.]), 800., None, None))
            mock('BallisticFit', return_value=fit)
            mock('PredictionGate', return_value=gate)
            intercept = mock('plane_intercept', return_value=(target, None))
            stack.enter_context(contextlib.redirect_stdout(output))
            await catch_plane.run(args)
        arm.move_to_position.assert_not_awaited()
        arm.stop.assert_not_awaited()
        return intercept, gate, output.getvalue()

    async def test_side_drives_preview_even_when_wrist_sees_nothing(self):
        intercept, gate, output = await self.run_preview('side', False)
        self.assertGreater(intercept.call_count, 1)
        self.assertGreater(gate.add.call_count, 1)
        self.assertIn('WOULD CATCH', output)  # zero commit slack must not hold forever
        self.assertIn('WRIST: no ball', output)

    async def test_duplicate_wrist_frame_does_not_advance_or_reset_gate(self):
        intercept, gate, output = await self.run_preview('wrist', True)
        self.assertGreater(intercept.call_count, 1)
        self.assertEqual(gate.add.call_count, 1)
        gate.reset.assert_not_called()
        self.assertIn('WOULD CATCH', output)


if __name__ == '__main__':
    unittest.main()
