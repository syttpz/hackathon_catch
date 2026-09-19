import unittest
from types import SimpleNamespace as NS
import numpy as np
import cv2
from motion.calibrate_cam2_pnp import solve_pose, ball_world

K = NS(fx=606.1, fy=606.1, cx=424.4, cy=244.8, width=848, height=480)


def rotation(axis, degrees):
    axis = np.asarray(axis, float)
    axis = axis/np.linalg.norm(axis)
    return cv2.Rodrigues(axis*np.radians(degrees))[0]


class Rig:
    """A camera off to the side, looking back at the arm's workspace."""
    camera_rotation = rotation([0.1, 1, 0.2], 118.)
    camera_translation = np.array([-900., 1150., 520.])

    @classmethod
    def project(cls, points, noise_px=0., seed=0):
        world_to_camera = cls.camera_rotation.T
        tvec = -world_to_camera@cls.camera_translation
        matrix = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1.]])
        pixels, _ = cv2.projectPoints(np.asarray(points, float),
                                      cv2.Rodrigues(world_to_camera)[0], tvec, matrix, None)
        pixels = pixels.reshape(-1, 2)
        if noise_px:
            pixels = pixels+np.random.default_rng(seed).normal(0, noise_px, pixels.shape)
        return pixels

    @classmethod
    def points(cls, count=14, seed=2):
        rng = np.random.default_rng(seed)
        return rng.uniform([-700, -400, 150], [200, 500, 800], (count, 3))


class PnPTests(unittest.TestCase):
    def test_recovers_the_camera_pose_exactly_without_noise(self):
        points = Rig.points()
        found_rotation, found_translation, errors, inliers = solve_pose(
            points, Rig.project(points), K)
        np.testing.assert_allclose(found_translation, Rig.camera_translation, atol=1e-3)
        np.testing.assert_allclose(found_rotation, Rig.camera_rotation, atol=1e-6)
        self.assertLess(errors.max(), 1e-3)
        self.assertTrue(inliers.all())

    def test_one_pixel_of_noise_stays_within_a_few_millimetres(self):
        points = Rig.points(count=18)
        found_rotation, found_translation, errors, inliers = solve_pose(
            points, Rig.project(points, noise_px=1.0, seed=5), K)
        self.assertLess(np.linalg.norm(found_translation-Rig.camera_translation), 25)
        angle = np.degrees(np.arccos(np.clip(
            (np.trace(found_rotation.T@Rig.camera_rotation)-1)/2, -1, 1)))
        self.assertLess(angle, 1.0)
        # The solver reports the inlier mean; RANSAC has already set the rest aside.
        self.assertLess(errors[inliers].mean(), 2.0)
        self.assertGreaterEqual(int(inliers.sum()), 12)

    def test_a_misdetected_sample_is_rejected_not_averaged_in(self):
        points = Rig.points(count=16)
        pixels = Rig.project(points, noise_px=0.4, seed=7)
        pixels[6] += [70., -45.]                     # a blob that was not the ball
        found_rotation, found_translation, errors, inliers = solve_pose(points, pixels, K)
        self.assertFalse(inliers[6])
        self.assertGreater(errors[6], 20)
        self.assertLess(np.linalg.norm(found_translation-Rig.camera_translation), 25)

    def test_coplanar_samples_are_refused_with_an_explanation(self):
        flat = Rig.points(count=12)
        flat[:, 2] = 400.
        with self.assertRaises(ValueError) as caught:
            solve_pose(flat, Rig.project(flat), K)
        self.assertIn('coplanar', str(caught.exception))

    def test_too_few_or_mismatched_samples_are_refused(self):
        points = Rig.points(count=3)
        with self.assertRaises(ValueError):
            solve_pose(points, Rig.project(points), K)
        points = Rig.points(count=8)
        with self.assertRaises(ValueError):
            solve_pose(points, Rig.project(points)[:5], K)

    def test_the_result_maps_camera_points_back_into_the_world(self):
        points = Rig.points()
        found_rotation, found_translation, *_ = solve_pose(points, Rig.project(points), K)
        in_camera = (points-Rig.camera_translation)@Rig.camera_rotation
        np.testing.assert_allclose(in_camera@found_rotation.T+found_translation, points, atol=1e-3)


class BallWorldTests(unittest.TestCase):
    def test_the_ball_is_placed_by_kinematics_not_by_the_camera(self):
        config = {'bowl_offset_gripper_mm': [15., 3., 148.]}
        position = np.array([-497.3, -81., 299.1])
        found = ball_world(position, np.eye(3), config)
        np.testing.assert_allclose(found, position+[15, 3, 148])

    def test_the_offset_follows_the_gripper_orientation(self):
        config = {'bowl_offset_gripper_mm': [0., 0., 148.]}
        turned = rotation([1, 0, 0], 90)             # gripper +Z now points along world -Y
        found = ball_world(np.zeros(3), turned, config)
        np.testing.assert_allclose(found, [0, -148, 0], atol=1e-9)


if __name__ == '__main__':
    unittest.main()


class JointSolveTests(unittest.TestCase):
    """The ball's place in the gripper is not known: it must come out of the fit."""

    OFFSET = np.array([12., -4., 131.])

    def rig(self, count=14, seed=3, spread=True):
        rng = np.random.default_rng(seed)
        positions = rng.uniform([-1100, -500, 100], [-500, 600, 800], (count, 3))
        rotations = np.array([rotation(rng.normal(size=3), rng.uniform(-70, 70) if spread else 0.)
                              for _ in range(count)])
        world = np.einsum('nij,j->ni', rotations, self.OFFSET)+positions
        return positions, rotations, world

    def test_recovers_both_the_camera_pose_and_the_offset(self):
        from motion.calibrate_cam2_pnp import solve_pose_and_offset
        positions, rotations, world = self.rig()
        found_rotation, found_translation, offset, errors, inliers = solve_pose_and_offset(
            positions, rotations, Rig.project(world), K, offset_guess=(0., 0., 100.))
        np.testing.assert_allclose(offset, self.OFFSET, atol=1.0)
        np.testing.assert_allclose(found_translation, Rig.camera_translation, atol=3.0)
        self.assertLess(errors[inliers].mean(), .5)

    def test_a_wrong_starting_offset_does_not_trap_the_fit(self):
        from motion.calibrate_cam2_pnp import solve_pose_and_offset
        positions, rotations, world = self.rig(count=16, seed=9)
        *_, offset, errors, inliers = solve_pose_and_offset(
            positions, rotations, Rig.project(world, noise_px=.5, seed=4), K,
            offset_guess=(15., 3., 148.))
        np.testing.assert_allclose(offset, self.OFFSET, atol=6.0)
        self.assertLess(errors[inliers].mean(), 1.5)

    def test_a_fixed_wrist_orientation_is_refused(self):
        from motion.calibrate_cam2_pnp import solve_pose_and_offset
        positions, rotations, world = self.rig(count=10, spread=False)
        with self.assertRaises(ValueError) as caught:
            solve_pose_and_offset(positions, rotations, Rig.project(world), K)
        self.assertIn('orientation varies by only', str(caught.exception))

    def test_a_misdetected_pixel_is_dropped_from_the_fit(self):
        from motion.calibrate_cam2_pnp import solve_pose_and_offset
        positions, rotations, world = self.rig(count=16, seed=11)
        pixels = Rig.project(world, noise_px=.4, seed=6)
        pixels[5] += [90., -60.]
        _, found_translation, offset, errors, inliers = solve_pose_and_offset(
            positions, rotations, pixels, K, offset_guess=(0., 0., 120.))
        self.assertFalse(inliers[5])
        np.testing.assert_allclose(offset, self.OFFSET, atol=8.0)
        self.assertLess(np.linalg.norm(found_translation-Rig.camera_translation), 30)
