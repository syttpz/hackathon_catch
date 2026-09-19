import unittest
from types import SimpleNamespace
import numpy as np
from motion.rough_cycle import RoughCycle, approaching_bowl

class RoughCycleTests(unittest.TestCase):
    def test_same_throw_cannot_retrigger(self):
        c=RoughCycle()
        self.assertTrue(c.update(0,True))
        c.fired(0)
        self.assertFalse(c.update(3,True))
        self.assertFalse(c.update(3.1,False))
        self.assertFalse(c.update(3.7,False))
        self.assertTrue(c.update(3.8,True))
    def test_brief_loss_does_not_rearm(self):
        c=RoughCycle();c.fired(0)
        c.update(3,False)
        self.assertFalse(c.update(3.2,True))
        self.assertFalse(c.ready)
    def test_approach_requires_motion_height_range_and_fit(self):
        def f(p,v,r=10):
            return SimpleNamespace(samples=5,residual_mm=r,at=lambda t:(np.array(p),np.array(v)))
        mouth=np.array([0,0,300])
        self.assertTrue(approaching_bowl(f([600,0,500],[-1000,0,0]),0,mouth))
        for flight in [None,f([600,0,500],[0,0,0]),f([600,0,500],[1000,0,0]),f([1600,0,500],[-1000,0,0]),f([600,0,300],[-1000,0,0]),f([600,0,500],[-1000,0,0],60)]:
            self.assertFalse(approaching_bowl(flight,0,mouth))

class RoughHandoffTests(unittest.TestCase):
    def test_wrist_then_side_and_side_then_wrist(self):
        from motion.rough_cycle import RoughHandoff
        for reverse in [False,True]:
            h=RoughHandoff(); f=object()
            if reverse:
                h.observe_approach(1.);h.observe_wrist(1.15,f)
            else:
                h.observe_wrist(1.,f);h.observe_approach(1.15)
            self.assertIs(h.candidate(1.2),f)
            self.assertIsNone(h.candidate(1.26))
    def test_duplicate_frames_never_extend_expiry(self):
        from motion.rough_cycle import RoughHandoff
        h=RoughHandoff();f=object()
        h.observe_wrist(1.,f);h.observe_approach(1.)
        h.observe_wrist(1.,f);h.observe_approach(1.)
        self.assertIsNone(h.candidate(1.3))
    def test_invalid_new_prediction_and_clear_discard_cached_target(self):
        from motion.rough_cycle import RoughHandoff
        h=RoughHandoff();f=object()
        h.observe_wrist(1.,f);h.observe_approach(1.)
        h.observe_wrist(1.1,None)
        self.assertIsNone(h.candidate(1.15))
        h.observe_wrist(1.2,f);h.clear()
        self.assertIsNone(h.candidate(1.21))
    def test_future_exposure_rejected(self):
        from motion.rough_cycle import RoughHandoff
        h=RoughHandoff();h.observe_wrist(2.,object());h.observe_approach(2.)
        self.assertIsNone(h.candidate(1.))
