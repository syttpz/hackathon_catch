import unittest
import cv2
import numpy as np
from types import SimpleNamespace
from old_files.red_ball.trajectory_local import red_center
from old_files.rolling.rolling_preview import locate_ball
from old_files.rolling.table_plane import TablePlane

class RedCandidatesTests(unittest.TestCase):
    def test_touching_skin_does_not_merge_with_red_ball(self):
        hsv=np.zeros((200,300,3),np.uint8)
        cv2.rectangle(hsv,(50,70),(150,130),(8,140,110),-1)
        cv2.circle(hsv,(165,100),25,(1,235,100),-1)
        center=red_center(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR))
        self.assertIsNotNone(center)
        np.testing.assert_allclose(center,[165,100],atol=1)
    def test_background_circle_with_wrong_size_is_filtered(self):
        bgr=np.zeros((300,400,3),np.uint8)
        cv2.circle(bgr,(200,150),20,(0,0,230),-1)
        cv2.circle(bgr,(50,80),7,(0,0,230),-1)
        k=SimpleNamespace(fx=300,fy=300,cx=200,cy=150)
        p,reason=locate_ball(bgr,k,np.array([0,0,480]),np.diag([1,-1,-1]),TablePlane(np.array([0,0,1]),0,.8,1),30)
        self.assertIsNone(reason)
        np.testing.assert_allclose(p,[0,0,30],atol=1)
    def test_invalid_projection_does_not_crash_detection(self):
        bgr=np.zeros((300,400,3),np.uint8);cv2.circle(bgr,(200,150),20,(0,0,230),-1)
        p,reason=locate_ball(bgr,SimpleNamespace(fx=300,fy=300,cx=200,cy=150),np.array([0,0,480]),np.eye(3),TablePlane(np.array([0,0,1]),0,.8,1),30)
        self.assertIsNone(p);self.assertIn('REJECTED',reason)
    def test_multiple_plausible_balls_rejected(self):
        bgr=np.zeros((300,400,3),np.uint8)
        for x in (100,250):cv2.circle(bgr,(x,150),20,(0,0,230),-1)
        p,reason=locate_ball(bgr,SimpleNamespace(fx=300,fy=300,cx=200,cy=150),np.array([0,0,480]),np.diag([1,-1,-1]),TablePlane(np.array([0,0,1]),0,.8,1),30)
        self.assertIsNone(p);self.assertIn('AMBIGUOUS',reason)
