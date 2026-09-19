import unittest
from types import SimpleNamespace
import numpy as np
from motion.stereo_tracking import StereoTracker, bearing, triangulate
from motion.ballistic import plane_crossing

K = SimpleNamespace(fx=600., fy=600., cx=424., cy=240.)
SIDE_O = np.array([-1500., -1000., 600.])
SIDE_R = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
WRIST_O = np.array([-400., 0., 400.])
WRIST_R = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
CONFIG = dict(min_samples=4, max_gap_s=.15, max_samples=7, max_residual_mm=30.,
              release_speed_mm_s=1200., min_span_s=.05, max_frame_age_s=.12,
              throw_volume_mm=[[-3000., -2000., -1000.], [1000., 2000., 2000.]])

def project(point, origin, rotation):
    q = rotation.T@(point-origin)
    return np.array([K.cx+K.fx*q[0]/q[2], K.cy+K.fy*q[1]/q[2]])

def candidate(point, origin, rotation):
    return [(project(point, origin, rotation), 20., 20.)]

class GeometryTests(unittest.TestCase):
    def test_intersecting_rays_recover_world_point(self):
        point = np.array([-900., 30., 800.])
        found, error = triangulate(SIDE_O, bearing(project(point,SIDE_O,SIDE_R),SIDE_R,K),
                                   WRIST_O, bearing(project(point,WRIST_O,WRIST_R),WRIST_R,K))
        np.testing.assert_allclose(found, point, atol=1e-8)
        self.assertLess(error, 1e-8)

    def test_parallel_rays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'parallel'):
            triangulate([0,0,0], [0,0,1], [10,0,0], [0,0,1])

    def test_large_camera_disagreement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            triangulate([0,0,0], [1,0,1], [100,100,0], [-1,0,1])

    def test_intersection_behind_camera_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'behind'):
            triangulate([0,0,0], [1,0,1], [100,0,0], [2,0,1])

class TrackingTests(unittest.TestCase):
    def test_side_frame_waits_for_later_wrist_frame_without_being_overwritten(self):
        tracker=StereoTracker(CONFIG);p=np.array([-900.,30.,600.])
        tracker.add_wrist(10.,candidate(p,WRIST_O,WRIST_R),WRIST_O,WRIST_R,K)
        tracker.add_side(10.008,candidate(p,SIDE_O,SIDE_R),SIDE_O,SIDE_R,K)
        tracker.update(10.010)
        self.assertEqual(len(tracker.fit.samples),0)
        tracker.add_side(10.024,candidate(p,SIDE_O,SIDE_R),SIDE_O,SIDE_R,K)
        tracker.add_wrist(10.016,candidate(p,WRIST_O,WRIST_R),WRIST_O,WRIST_R,K)
        tracker.update(10.026)
        self.assertEqual(tracker.fit.samples[-1][0],10.008)
        self.assertEqual(len(tracker.pending),1)

    def test_staggered_frames_recover_ballistic_crossing(self):
        tracker = StereoTracker(CONFIG)
        def position(t):return np.array([-1200+1200*t, 50., 600+600*t-4905*t*t])
        tracker.add_wrist(10-.004,candidate(position(-.004),WRIST_O,WRIST_R),WRIST_O,WRIST_R,K)
        for t in np.arange(0,.15,.016):
            tracker.add_wrist(10+t+.012,candidate(position(t+.012),WRIST_O,WRIST_R),WRIST_O,WRIST_R,K)
            tracker.add_side(10+t,candidate(position(t),SIDE_O,SIDE_R),SIDE_O,SIDE_R,K)
            tracker.update(10+t+.016)
        self.assertIsNotNone(tracker.flight)
        np.testing.assert_allclose(tracker.flight.at(10+.12)[0],position(.12),atol=2.)
        when, point, speed = plane_crossing(tracker.flight, 320., 10.15)
        self.assertLess(speed[2], 0)
        np.testing.assert_allclose(point, position(when-10), atol=3.)
        self.assertIsNone(tracker.current(11.))

    def test_missing_wrist_detection_is_not_bridged(self):
        tracker=StereoTracker(CONFIG);p=np.array([-900.,30.,600.])
        tracker.add_wrist(10.,candidate(p,WRIST_O,WRIST_R),WRIST_O,WRIST_R,K)
        tracker.add_wrist(10.016,[],WRIST_O,WRIST_R,K)
        tracker.add_side(10.008,candidate(p,SIDE_O,SIDE_R),SIDE_O,SIDE_R,K)
        tracker.update(10.02)
        self.assertIsNone(tracker.flight)
        self.assertIn('missing',tracker.status)
        self.assertEqual(len(tracker.fit.samples),0)

    def test_static_ball_does_not_invent_flight_and_duplicate_is_not_added(self):
        tracker=StereoTracker(CONFIG);p=np.array([-900.,30.,600.])
        for t in np.arange(10,10.2,.016):
            tracker.add_wrist(t,candidate(p,WRIST_O,WRIST_R),WRIST_O,WRIST_R,K)
            tracker.add_side(t,candidate(p,SIDE_O,SIDE_R),SIDE_O,SIDE_R,K)
            tracker.update(t+.002)
            seq=tracker.sequence;tracker.update(t+.004)
            self.assertEqual(tracker.sequence,seq)
            self.assertIsNone(tracker.flight)

    def test_ambiguous_side_detection_invalidates_prediction(self):
        tracker=StereoTracker(CONFIG);p=np.array([-900.,30.,600.])
        tracker.flight=object()
        tracker.add_side(10.,candidate(p,SIDE_O,SIDE_R)*2,SIDE_O,SIDE_R,K)
        tracker.update(10.001)
        self.assertIsNone(tracker.flight)
        self.assertIn('ambiguous',tracker.status)

if __name__=='__main__':unittest.main()
