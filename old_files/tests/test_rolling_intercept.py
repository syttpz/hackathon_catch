import unittest
from types import SimpleNamespace
import numpy as np
from old_files.rolling.rolling_intercept import line_crossing, point_on_ball_plane, RollingFit

class InterceptionTests(unittest.TestCase):
    def test_future_segment_hit(self):
        hit=line_crossing([50,-100,30],[0,100,0],[[0,0,175],[100,0,175]])
        self.assertAlmostEqual(hit[0],1)
        np.testing.assert_allclose(hit[2],[50,0,175])
    def test_rejects_parallel_past_outside_and_late(self):
        line=[[0,0,175],[100,0,175]]
        for p,v in [([50,-100],[100,0]),([50,100],[0,100]),([150,-100],[0,100]),([50,-100],[0,1])]:
            self.assertIsNone(line_crossing(p,v,line))
    def test_plane_projection(self):
        k=SimpleNamespace(fx=100,fy=100,cx=50,cy=50)
        np.testing.assert_allclose(point_on_ball_plane([50,50],k,np.array([10,20,300]),np.diag([1,-1,-1]),30),[10,20,30])
    def test_fit_and_gap(self):
        fit=RollingFit()
        result=None
        for i in range(12): result=fit.add(i/60,np.array([100+i/60*200,20,30]))
        np.testing.assert_allclose(result[1],[200,0,0],atol=1e-8)
        self.assertIsNone(fit.add(1,np.array([300,20,30])))
        self.assertEqual(len(fit.samples),1)
