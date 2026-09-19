import unittest
import numpy as np
from motion.handeye import (kabsch, solve_eye_to_hand, orientation_spread_deg,
                            rotation_to_quaternion, viam_frame)
from motion.live_camera_pose import pose_matrix
from viam.proto.common import Pose


def rotation(axis, degrees):
    axis = np.asarray(axis, float)
    axis = axis/np.linalg.norm(axis)
    angle = np.radians(degrees)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)


class Truth:
    """A known rig: camera pose, ball offset, and a spread of gripper poses."""
    camera_rotation = rotation([0.2, 1, 0.35], 121.)
    camera_translation = np.array([-820., 430., 610.])
    ball_offset = np.array([3.5, -2.0, 128.0])       # between the jaws, past the flange

    @classmethod
    def poses(cls, count=14, seed=3):
        rng = np.random.default_rng(seed)
        rotations, positions = [], []
        for _ in range(count):
            rotations.append(rotation(rng.normal(size=3), rng.uniform(-75, 75)))
            positions.append(rng.uniform([-400, -500, 150], [500, 400, 650]))
        return np.array(rotations), np.array(positions)

    @classmethod
    def observe(cls, rotations, positions, noise_mm=0., seed=11):
        world = np.einsum('nij,j->ni', rotations, cls.ball_offset)+positions
        points = (world-cls.camera_translation) @ cls.camera_rotation
        if noise_mm:
            points = points+np.random.default_rng(seed).normal(0, noise_mm, points.shape)
        return points


class KabschTests(unittest.TestCase):
    def test_recovers_a_known_rigid_transform(self):
        truth_rotation = rotation([1, -2, .5], 47.)
        truth_translation = np.array([120., -30., 55.])
        source = np.random.default_rng(0).uniform(-300, 300, (12, 3))
        target = source@truth_rotation.T+truth_translation
        found_rotation, found_translation = kabsch(source, target)
        np.testing.assert_allclose(found_rotation, truth_rotation, atol=1e-9)
        np.testing.assert_allclose(found_translation, truth_translation, atol=1e-8)

    def test_never_returns_a_reflection_for_degenerate_input(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.]])*100
        found, _ = kabsch(source, source*[1, 1, -1])
        self.assertAlmostEqual(np.linalg.det(found), 1.0, places=9)

    def test_rejects_mismatched_or_tiny_input(self):
        with self.assertRaises(ValueError):
            kabsch(np.zeros((4, 3)), np.zeros((5, 3)))
        with self.assertRaises(ValueError):
            kabsch(np.zeros((2, 3)), np.zeros((2, 3)))


class SolverTests(unittest.TestCase):
    def test_recovers_camera_pose_and_ball_offset_exactly(self):
        rotations, positions = Truth.poses()
        points = Truth.observe(rotations, positions)
        found_rotation, found_translation, offset, residuals, scale = solve_eye_to_hand(
            points, rotations, positions)
        self.assertEqual(scale, 1.0)
        np.testing.assert_allclose(found_rotation, Truth.camera_rotation, atol=1e-7)
        np.testing.assert_allclose(found_translation, Truth.camera_translation, atol=1e-5)
        np.testing.assert_allclose(offset, Truth.ball_offset, atol=1e-5)
        self.assertLess(residuals.max(), 1e-6)

    def test_millimetre_noise_stays_millimetre_accurate(self):
        rotations, positions = Truth.poses(count=20)
        points = Truth.observe(rotations, positions, noise_mm=2.0)
        found_rotation, found_translation, offset, residuals, _ = solve_eye_to_hand(
            points, rotations, positions)
        self.assertLess(np.linalg.norm(found_translation-Truth.camera_translation), 6)
        self.assertLess(np.linalg.norm(offset-Truth.ball_offset), 6)
        angle = np.degrees(np.arccos(np.clip(
            (np.trace(found_rotation.T@Truth.camera_rotation)-1)/2, -1, 1)))
        self.assertLess(angle, 1.5)
        self.assertLess(residuals.mean(), 5)

    def test_a_fixed_wrist_orientation_is_refused_not_silently_absorbed(self):
        rotations, positions = Truth.poses(count=12)
        fixed = np.repeat(rotations[:1], len(rotations), axis=0)
        points = Truth.observe(fixed, positions)
        with self.assertRaises(ValueError) as caught:
            solve_eye_to_hand(points, fixed, positions)
        self.assertIn('orientation varies by only', str(caught.exception))

    def test_orientation_spread_reports_the_widest_pair(self):
        self.assertAlmostEqual(orientation_spread_deg([np.eye(3)]*4), 0)
        spread = orientation_spread_deg([np.eye(3), rotation([0, 0, 1], 30), rotation([0, 0, 1], 95)])
        self.assertAlmostEqual(spread, 95, places=6)

    def test_residuals_expose_one_bad_observation(self):
        rotations, positions = Truth.poses(count=16)
        points = Truth.observe(rotations, positions)
        points[5] += [70., -40., 25.]                 # a misdetected ball
        *_, residuals, _ = solve_eye_to_hand(points, rotations, positions)
        self.assertEqual(int(np.argmax(residuals)), 5)
        self.assertGreater(residuals[5], 10*np.median(residuals))

    def test_too_few_poses_or_bad_numbers_are_rejected(self):
        rotations, positions = Truth.poses(count=4)
        points = Truth.observe(rotations, positions)
        with self.assertRaises(ValueError):
            solve_eye_to_hand(points[:3], rotations[:3], positions[:3])
        broken = points.copy()
        broken[0, 0] = np.nan
        with self.assertRaises(ValueError):
            solve_eye_to_hand(broken, rotations, positions)


class ViamFrameTests(unittest.TestCase):
    def test_quaternion_round_trips_through_the_viam_orientation_conversion(self):
        for axis, degrees in ([0, 0, 1], 90), ([1, 0, 0], 180), ([.3, -.7, .2], 47), ([1, 1, 1], 359):
            truth = rotation(axis, degrees)
            w, x, y, z = rotation_to_quaternion(truth)
            self.assertAlmostEqual(np.linalg.norm([w, x, y, z]), 1, places=12)
            # Rebuild the matrix from the quaternion and compare.
            rebuilt = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            np.testing.assert_allclose(rebuilt, truth, atol=1e-9)

    def test_frame_block_has_the_shape_viam_expects(self):
        frame = viam_frame(Truth.camera_rotation, Truth.camera_translation)
        self.assertEqual(frame['parent'], 'world')
        self.assertEqual(set(frame['translation']), {'X', 'Y', 'Z'})
        self.assertEqual(frame['orientation']['type'], 'quaternion')
        self.assertEqual(set(frame['orientation']['value']), {'W', 'X', 'Y', 'Z'})
        self.assertAlmostEqual(frame['translation']['X'], Truth.camera_translation[0])


if __name__ == '__main__':
    unittest.main()


class GeometryTests(unittest.TestCase):
    """The two independent ways a capture turns one blob into a 3D centre."""
    from types import SimpleNamespace as _NS
    K = _NS(fx=606.1, fy=606.1, cx=424.4, cy=244.8, width=848, height=480)

    def test_a_centred_ball_lies_on_the_optical_axis(self):
        from motion.calibrate_cam2 import centre_from_size
        point = centre_from_size((self.K.cx, self.K.cy), 30, self.K, 30)
        np.testing.assert_allclose(point[:2], [0, 0], atol=1e-9)
        self.assertAlmostEqual(point[2], 606.1, places=6)

    def test_size_and_depth_agree_on_a_synthetic_ball(self):
        from motion.calibrate_cam2 import centre_from_size, centre_from_depth, ray
        radius_mm, distance = 30., 900.
        pixel = (self.K.cx+140, self.K.cy-60)
        direction = ray(pixel, self.K)
        centre = direction*distance
        radius_px = (self.K.fx+self.K.fy)*.5*radius_mm/distance
        # Depth map holding the sphere's front surface along the optical axis.
        surface = (centre-direction*radius_mm)[2]
        depth = np.full((self.K.height, self.K.width), surface)
        by_size = centre_from_size(pixel, radius_px, self.K, radius_mm)
        by_depth = centre_from_depth(pixel, radius_px, depth, self.K, radius_mm)
        np.testing.assert_allclose(by_size, centre, atol=1e-6)
        np.testing.assert_allclose(by_depth, centre, atol=1e-6)

    def test_missing_depth_returns_nothing_rather_than_a_guess(self):
        from motion.calibrate_cam2 import centre_from_depth
        depth = np.zeros((self.K.height, self.K.width))
        self.assertIsNone(centre_from_depth((400, 240), 20, depth, self.K, 30))

    def test_background_bleed_does_not_push_the_range_outwards(self):
        from motion.calibrate_cam2 import centre_from_depth, ray
        radius_mm, distance = 30., 800.
        pixel = (self.K.cx, self.K.cy)
        surface = distance-radius_mm
        depth = np.full((self.K.height, self.K.width), 4000.)   # far background
        u, v = int(pixel[0]), int(pixel[1])
        depth[v-12:v+13, u-12:u+13] = surface                   # the ball
        point = centre_from_depth(pixel, 30, depth, self.K, radius_mm)
        np.testing.assert_allclose(point, ray(pixel, self.K)*distance, atol=1e-6)


class ScaleTests(unittest.TestCase):
    """A wrong assumed ball radius scales every camera point by one factor."""

    def setUp(self):
        self.rotations, self.positions = Truth.poses(count=14)
        self.points = Truth.observe(self.rotations, self.positions)

    def test_a_radius_error_is_recovered_exactly(self):
        for truth_scale in (1.0, 1.49, 0.7):
            *_, scale = solve_eye_to_hand(self.points/truth_scale, self.rotations,
                                          self.positions, estimate_scale=True)
            self.assertAlmostEqual(scale, truth_scale, places=9)

    def test_scaled_input_still_recovers_the_camera_pose_and_offset(self):
        rotation, translation, offset, residuals, _ = solve_eye_to_hand(
            self.points/1.49, self.rotations, self.positions, estimate_scale=True)
        np.testing.assert_allclose(translation, Truth.camera_translation, atol=1e-6)
        np.testing.assert_allclose(offset, Truth.ball_offset, atol=1e-6)
        np.testing.assert_allclose(rotation, Truth.camera_rotation, atol=1e-8)
        self.assertLess(residuals.max(), 1e-6)

    def test_a_rigid_fit_cannot_absorb_it_and_says_so_through_residuals(self):
        *_, residuals, _ = solve_eye_to_hand(self.points/1.49, self.rotations, self.positions)
        self.assertGreater(residuals.max(), 100)

    def test_umeyama_scale_matches_a_hand_built_similarity(self):
        source = np.random.default_rng(5).uniform(-200, 200, (10, 3))
        truth_rotation = rotation([.2, 1, -.4], 63.)
        target = 2.5*source@truth_rotation.T+np.array([40., -15., 90.])
        found_rotation, found_translation, scale = kabsch(source, target, estimate_scale=True)
        self.assertAlmostEqual(scale, 2.5, places=9)
        np.testing.assert_allclose(found_rotation, truth_rotation, atol=1e-9)
        np.testing.assert_allclose(found_translation, [40, -15, 90], atol=1e-7)
