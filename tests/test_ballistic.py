import unittest
import numpy as np
from motion.ballistic import (GRAVITY_MM_S2, ArmTiming, BallisticFit, Flight,
                              intercept, reachable, required_lead)

CONFIG = {
    'catch_box_mm': [[-300, -700, 150], [640, 700, 900]],
    'min_reach_mm': 200, 'max_reach_mm': 760,
    'max_prediction_s': 1.5, 'search_step_s': .01,
    'close_lead_s': .12, 'arrival_margin_s': .08,
}
QUICK = ArmTiming(latency_s=.12, speed_mm_s=900.)
SLOW = ArmTiming(latency_s=.60, speed_mm_s=300.)


def toss(origin, velocity, times, noise_mm=0., seed=0):
    origin, velocity = np.asarray(origin, float), np.asarray(velocity, float)
    points = np.array([origin+velocity*t+.5*GRAVITY_MM_S2*t*t for t in times])
    if noise_mm:
        points = points+np.random.default_rng(seed).normal(0, noise_mm, points.shape)
    return points


class FitTests(unittest.TestCase):
    def feed(self, times, points, **kwargs):
        fit = BallisticFit(**kwargs)
        flight = None
        for t, p in zip(times, points):
            flight = fit.add(t, p)
        return fit, flight

    def test_recovers_a_clean_parabola(self):
        origin, velocity = np.array([1800., 300., 900.]), np.array([-2600., -350., 1800.])
        times = np.arange(0, .2, 1/60)
        _, flight = self.feed(times, toss(origin, velocity, times))
        self.assertIsNotNone(flight)
        position, speed = flight.at(0.)
        np.testing.assert_allclose(position, origin, atol=1e-6)
        np.testing.assert_allclose(speed, velocity, atol=1e-6)
        self.assertLess(flight.residual_mm, 1e-6)

    def test_gravity_is_not_absorbed_into_the_velocity(self):
        # A straight-line fit would report a falling ball as constant velocity.
        origin, velocity = np.array([1200., 0., 1000.]), np.array([-2000., 0., 0.])
        times = np.arange(0, .3, 1/60)
        _, flight = self.feed(times, toss(origin, velocity, times))
        far = flight.at(flight.reference_time+.4)[0]
        drop = origin[2]+0-.5*9810*(.4+times[-1])**2
        self.assertAlmostEqual(far[2], drop, delta=1e-3)

    def test_apex_is_where_vertical_speed_vanishes(self):
        times = np.arange(0, .25, 1/60)
        points = toss([0, 0, 500.], [1000., 0, 2000.], times)
        _, flight = self.feed(times, points)
        apex = flight.apex_time()
        self.assertAlmostEqual(flight.at(apex)[1][2], 0, places=6)
        self.assertAlmostEqual(apex, 2000/9810, places=6)

    def test_noise_is_tolerated_and_reported(self):
        times = np.arange(0, .3, 1/60)
        points = toss([1500., 200., 800.], [-2400., -200., 1700.], times, noise_mm=8., seed=4)
        _, flight = self.feed(times, points)
        self.assertIsNotNone(flight)
        self.assertGreater(flight.residual_mm, 0)
        self.assertLess(np.linalg.norm(flight.at(times[-1])[0]
                                       - toss([1500., 200., 800.], [-2400., -200., 1700.],
                                              [times[-1]])[0]), 25)

    def test_too_few_or_too_brief_samples_give_nothing(self):
        times = np.arange(0, .05, 1/60)
        _, flight = self.feed(times, toss([0, 0, 0], [0, 0, 0], times))
        self.assertIsNone(flight)

    def test_a_gap_or_a_jump_restarts_the_fit(self):
        times = list(np.arange(0, .2, 1/60))
        points = list(toss([1500., 0., 800.], [-2400., 0., 1700.], times))
        fit, flight = self.feed(times, points)
        self.assertIsNotNone(flight)
        self.assertIsNone(fit.add(times[-1]+.5, points[-1]))       # gap
        self.assertEqual(len(fit.samples), 1)
        fit, _ = self.feed(times, points)
        self.assertIsNone(fit.add(times[-1]+.02, points[-1]+[2000., 0, 0]))   # jump
        self.assertEqual(len(fit.samples), 1)

    def test_backward_timestamps_are_ignored_not_fitted(self):
        times = list(np.arange(0, .2, 1/60))
        points = list(toss([1500., 0., 800.], [-2400., 0., 1700.], times))
        fit, _ = self.feed(times, points)
        before = len(fit.samples)
        self.assertIsNone(fit.add(times[-1]-.01, points[-1]))
        self.assertEqual(len(fit.samples), before)

    def test_a_non_ballistic_path_is_rejected_by_its_residual(self):
        times = np.arange(0, .3, 1/60)
        points = np.array([[1500-4000*t, 300*np.sin(40*t), 800.] for t in times])
        _, flight = self.feed(times, points)
        self.assertIsNone(flight)


class InterceptTests(unittest.TestCase):
    def flight_towards_robot(self):
        # Thrown from 2.2 m away, rising, arriving over the catch box.
        return Flight(origin=np.array([2200., 100., 700.]),
                      velocity=np.array([-3200., -150., 2400.]),
                      reference_time=0., residual_mm=1., samples=12)

    def test_picks_a_reachable_point_the_arm_can_reach_in_time(self):
        flight = self.flight_towards_robot()
        found, reason = intercept(flight, np.array([400., 0., 500.]), CONFIG, QUICK, now=0.)
        self.assertIsNone(reason)
        self.assertTrue(reachable(found.point, CONFIG))
        self.assertGreaterEqual(found.slack_s, 0)
        np.testing.assert_allclose(flight.at(found.arrival)[0], found.point, atol=1e-9)

    def test_slowest_prefers_apex_and_earliest_prefers_margin(self):
        flight = self.flight_towards_robot()
        flange = np.array([400., 0., 500.])
        slowest, _ = intercept(flight, flange, CONFIG, QUICK, now=0., prefer='slowest')
        earliest, _ = intercept(flight, flange, CONFIG, QUICK, now=0., prefer='earliest')
        self.assertLessEqual(slowest.speed_mm_s, earliest.speed_mm_s)
        self.assertLessEqual(earliest.arrival, slowest.arrival)

    def test_a_slow_arm_is_told_it_cannot_make_it_rather_than_committing(self):
        flight = self.flight_towards_robot()
        found, reason = intercept(flight, np.array([400., 0., 500.]), CONFIG, SLOW, now=0.)
        self.assertIsNone(found)
        self.assertIn('cannot get there in time', reason)

    def test_a_throw_that_misses_the_box_is_refused(self):
        flight = Flight(origin=np.array([2200., 2500., 700.]),
                        velocity=np.array([-3200., 0., 1200.]),
                        reference_time=0., residual_mm=1., samples=12)
        found, reason = intercept(flight, np.array([400., 0., 500.]), CONFIG, QUICK, now=0.)
        self.assertIsNone(found)
        self.assertIn('reachable catch box', reason)

    def test_deciding_late_leaves_no_solution(self):
        flight = self.flight_towards_robot()
        flange = np.array([400., 0., 500.])
        self.assertIsNotNone(intercept(flight, flange, CONFIG, QUICK, now=0.)[0])
        found, reason = intercept(flight, flange, CONFIG, QUICK, now=.55)
        self.assertIsNone(found)
        self.assertIn('NO CATCH', reason)

    def test_slack_matches_the_timing_model(self):
        flight = self.flight_towards_robot()
        flange = np.array([400., 0., 500.])
        found, _ = intercept(flight, flange, CONFIG, QUICK, now=0.)
        expected = (found.arrival-0.)-required_lead(found.distance_mm, CONFIG, QUICK)
        self.assertAlmostEqual(found.slack_s, expected, places=9)

    def test_reachability_respects_the_radial_envelope_not_just_the_box(self):
        corner = np.array([640., 700., 900.])          # inside the box, far outside reach
        self.assertGreater(np.linalg.norm(corner), CONFIG['max_reach_mm'])
        self.assertFalse(reachable(corner, CONFIG))
        self.assertTrue(reachable(np.array([400., 100., 400.]), CONFIG))

    def test_bad_timing_or_search_settings_raise(self):
        flight = self.flight_towards_robot()
        with self.assertRaises(ValueError):
            intercept(flight, np.zeros(3), {**CONFIG, 'search_step_s': 0}, QUICK, now=0.)
        with self.assertRaises(ValueError):
            intercept(flight, np.zeros(3), CONFIG, ArmTiming(.1, 0), now=0.)
        with self.assertRaises(ValueError):
            intercept(flight, np.zeros(3), CONFIG, QUICK, now=0., prefer='cheapest')


if __name__ == '__main__':
    unittest.main()


BOWL = {**CONFIG, 'catcher': 'bowl', 'tool_offset_mm': 180., 'bowl_radius_mm': 100.,
        'bowl_upright': .5, 'catch_orientation': [0, 0, -1, 31.2]}


class BowlTests(unittest.TestCase):
    """A bowl catches passively: no jaw to time, and the mouth sits off the flange."""

    def test_the_mouth_axis_faces_the_incoming_ball_when_upright_is_zero(self):
        from motion.ballistic import bowl_orientation
        velocity = np.array([-2200., 0., -2853.])
        rotation = bowl_orientation(velocity, 0.)
        np.testing.assert_allclose(rotation[:, 2], -velocity/np.linalg.norm(velocity), atol=1e-9)

    def test_full_upright_points_the_mouth_at_the_sky(self):
        from motion.ballistic import bowl_orientation
        rotation = bowl_orientation(np.array([-2200., 0., -2853.]), 1.)
        np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=1e-9)

    def test_blending_lands_between_the_two_and_stays_a_rotation(self):
        from motion.ballistic import bowl_orientation
        velocity = np.array([-2200., 0., -2853.])
        rotation = bowl_orientation(velocity, .5)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=9)
        np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-9)
        facing = -velocity/np.linalg.norm(velocity)
        tilt = np.degrees(np.arccos(np.clip(rotation[:, 2]@[0, 0, 1], -1, 1)))
        self.assertLess(tilt, np.degrees(np.arccos(np.clip(facing@[0, 0, 1], -1, 1))))
        self.assertGreater(tilt, 0)

    def test_a_stationary_or_degenerate_velocity_falls_back_to_upright(self):
        from motion.ballistic import bowl_orientation
        np.testing.assert_allclose(bowl_orientation(np.zeros(3), .5)[:, 2], [0, 0, 1], atol=1e-9)
        # -v exactly cancels the up blend
        np.testing.assert_allclose(bowl_orientation(np.array([0, 0, -1.]), .5)[:, 2],
                                   [0, 0, 1], atol=1e-9)

    def test_upright_outside_zero_to_one_is_rejected(self):
        from motion.ballistic import bowl_orientation
        with self.assertRaises(ValueError):
            bowl_orientation(np.array([0, 0, -1.]), 1.5)

    def test_the_flange_stands_back_from_the_mouth_by_the_tool_offset(self):
        from motion.ballistic import catch_pose
        point = np.array([400., 0., 600.])
        velocity = np.array([-2200., 0., -2853.])
        flange, rotation = catch_pose(point, velocity, BOWL)
        self.assertAlmostEqual(np.linalg.norm(point-flange), BOWL['tool_offset_mm'], places=9)
        np.testing.assert_allclose(flange+rotation@[0, 0, BOWL['tool_offset_mm']], point, atol=1e-9)

    def test_dropping_the_closing_lead_widens_the_feasible_window(self):
        flight = Flight(origin=np.array([1500., 150., 800.]),
                        velocity=np.array([-2200., -300., 2052.]),
                        reference_time=0., residual_mm=1., samples=12)
        park = np.array([380., 0., 520.])
        timing = ArmTiming(latency_s=.12, speed_mm_s=900.)
        jaws, _ = intercept(flight, park, {**BOWL, 'catcher': 'gripper'}, timing, now=.15)
        bowl, _ = intercept(flight, park, BOWL, timing, now=.15)
        self.assertIsNotNone(bowl)
        if jaws is not None:
            self.assertGreater(bowl.slack_s, jaws.slack_s)

    def test_required_lead_drops_by_exactly_the_closing_lead(self):
        timing = ArmTiming(latency_s=.12, speed_mm_s=900.)
        jaws = required_lead(250, {**BOWL, 'catcher': 'gripper'}, timing)
        bowl = required_lead(250, BOWL, timing)
        self.assertAlmostEqual(jaws-bowl, BOWL['close_lead_s'], places=9)

    def test_reachability_is_judged_on_the_flange_not_the_mouth(self):
        from motion.ballistic import catch_pose
        # A mouth position just inside the envelope whose flange lands outside it.
        config = {**BOWL, 'max_reach_mm': 700.}
        point = np.array([0., 0., 690.])
        velocity = np.array([0., 0., -3000.])     # straight down: bowl faces up
        flange, _ = catch_pose(point, velocity, config)
        self.assertLess(np.linalg.norm(point), config['max_reach_mm'])
        self.assertGreater(np.linalg.norm(flange), 0)
        self.assertAlmostEqual(flange[2], 690-180, places=6)


PLANE = {**BOWL, 'catch_plane_z_mm': 600., 'bowl_offset_gripper_mm': [0., 0., 0.],
         'bowl_axis_flange': [1., 0., 0.], 'bowl_upright': .5,
         'arm_latency_s': .159, 'arm_speed_mm_s': 435.}
MEASURED = ArmTiming(latency_s=.159, speed_mm_s=435.)


def lob(peak_z, catch_z, horizontal, flight_s):
    """A toss that rises to `peak_z` and comes back down through `catch_z`."""
    from motion.ballistic import GRAVITY_MM_S2
    speed_z = np.sqrt(2*9810*(peak_z-catch_z))
    return Flight(origin=np.array([horizontal, 0., catch_z]),
                  velocity=np.array([-horizontal/flight_s, 0., speed_z]),
                  reference_time=0., residual_mm=2., samples=10)


class PlaneCrossingTests(unittest.TestCase):
    def test_finds_the_descending_crossing_not_the_rising_one(self):
        from motion.ballistic import plane_crossing
        flight = lob(peak_z=1100., catch_z=600., horizontal=1400., flight_s=.8)
        when, point, velocity = plane_crossing(flight, 600., not_before=0.)
        self.assertLess(velocity[2], 0)                       # descending
        self.assertAlmostEqual(point[2], 600., places=6)
        self.assertGreater(when, 0)
        # The rising crossing is at t=0 and must have been skipped.
        self.assertGreater(when, 1e-3)

    def test_an_arc_that_never_reaches_the_plane_returns_nothing(self):
        from motion.ballistic import plane_crossing
        low = Flight(origin=np.array([1400., 0., 500.]), velocity=np.array([-1500., 0., 300.]),
                     reference_time=0., residual_mm=1., samples=8)
        self.assertIsNone(plane_crossing(low, 900., not_before=0.))

    def test_a_crossing_already_in_the_past_is_not_returned(self):
        from motion.ballistic import plane_crossing
        flight = lob(peak_z=1100., catch_z=600., horizontal=1400., flight_s=.8)
        when, *_ = plane_crossing(flight, 600., not_before=0.)
        self.assertIsNone(plane_crossing(flight, 600., not_before=when+.01))

    def test_upward_gravity_is_rejected_rather_than_silently_solved(self):
        from motion.ballistic import plane_crossing
        broken = Flight(origin=np.zeros(3), velocity=np.zeros(3), reference_time=0.,
                        residual_mm=0., samples=6, gravity=np.array([0., 0., 9810.]))
        with self.assertRaises(ValueError):
            plane_crossing(broken, 600., not_before=0.)


class PlaneInterceptTests(unittest.TestCase):
    def test_a_lob_landing_near_the_parked_bowl_is_catchable_on_measured_timing(self):
        from motion.ballistic import plane_intercept
        flight = lob(peak_z=1150., catch_z=600., horizontal=1200., flight_s=.75)
        # Park where the ball is heading, so only a short correction is needed.
        found, reason = plane_intercept(flight, np.array([120., 0., 400.]), np.zeros(3), PLANE, MEASURED, now=.15)
        self.assertIsNone(reason)
        self.assertGreaterEqual(found.slack_s, 0)
        self.assertAlmostEqual(found.point[2], PLANE['catch_plane_z_mm'], places=6)

    def test_a_lob_landing_far_from_the_bowl_is_refused_with_the_numbers(self):
        from motion.ballistic import plane_intercept
        flight = lob(peak_z=1150., catch_z=600., horizontal=1200., flight_s=.75)
        found, reason = plane_intercept(flight, np.array([-650., 600., 400.]), np.zeros(3), PLANE, MEASURED, now=.15)
        self.assertIsNone(found)
        self.assertIn('needs', reason)
        self.assertIn('margin, but the ball arrives in', reason)

    def test_only_horizontal_travel_is_charged_for(self):
        from motion.ballistic import plane_intercept
        flight = lob(peak_z=1150., catch_z=600., horizontal=1200., flight_s=.75)
        low = plane_intercept(flight, np.array([120., 0., 200.]), np.zeros(3), PLANE, MEASURED, now=.15)[0]
        high = plane_intercept(flight, np.array([120., 0., 750.]), np.zeros(3), PLANE, MEASURED, now=.15)[0]
        self.assertIsNotNone(low)
        self.assertAlmostEqual(low.distance_mm, high.distance_mm, places=9)
        self.assertAlmostEqual(low.slack_s, high.slack_s, places=9)

    def test_a_crossing_outside_the_box_is_refused(self):
        from motion.ballistic import plane_intercept
        flight = lob(peak_z=1150., catch_z=600., horizontal=1200., flight_s=.75)
        # Narrow enough that the FLANGE, once stood back by the bowl offset,
        # falls outside it even though the crossing itself looks close in.
        tight = {**PLANE, 'catch_box_mm': [[-50, -50, 150], [20, 50, 900]]}
        found, reason = plane_intercept(flight, np.array([0., 0., 400.]), np.zeros(3), tight, MEASURED, now=.15)
        self.assertIsNone(found)
        self.assertIn('outside the reachable catch box', reason)

class RoughAttemptTests(unittest.TestCase):
    def test_late_short_attempt_but_not_far_or_expired(self):
        from motion.ballistic import plane_crossing, plane_intercept
        flight = lob(peak_z=1150., catch_z=600., horizontal=1200., flight_s=.75)
        arrival, point, _ = plane_crossing(flight, PLANE['catch_plane_z_mm'], 0.)
        park = point - np.array([24., 0., 0.])
        slow = ArmTiming(1., 100.)
        config = dict(PLANE, rough_attempt=True)
        self.assertIsNone(plane_intercept(flight, park, np.zeros(3), PLANE, slow, arrival-.12)[0])
        target, _ = plane_intercept(flight, park, np.zeros(3), config, slow, arrival-.12)
        self.assertIsNotNone(target)
        self.assertLess(target.slack_s, 0)
        self.assertIsNotNone(plane_intercept(flight, point-[120.,0.,0.], np.zeros(3), config, slow, arrival-.12)[0])
        self.assertIsNone(plane_intercept(flight, point-[151.,0.,0.], np.zeros(3), config, slow, arrival-.12)[0])
        self.assertIsNone(plane_intercept(flight, park, np.zeros(3), config, slow, arrival-.01)[0])
        tight = dict(config, catch_box_mm=[[-1,-1,-1],[1,1,1]])
        self.assertIsNone(plane_intercept(flight, park, np.zeros(3), tight, slow, arrival-.12)[0])
