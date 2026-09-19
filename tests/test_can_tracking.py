import unittest
import cv2
import numpy as np
from motion.can_tracking import red_can_candidates, can_center_height

class CanTrackingTests(unittest.TestCase):
    def test_long_red_label_with_white_lettering(self):
        image=np.zeros((240,320,3),np.uint8)
        cv2.rectangle(image,(90,90),(210,140),(0,0,200),-1)
        cv2.putText(image,'Coke',(95,125),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        candidates=red_can_candidates(image)
        self.assertEqual(len(candidates),1)
        np.testing.assert_allclose(candidates[0].center_px,[150,115],atol=2)
        self.assertGreater(candidates[0].long_side_px/candidates[0].short_side_px,2)
    def test_multiple_cans_remain_ambiguous(self):
        image=np.zeros((240,320,3),np.uint8)
        for y in (50,150):cv2.rectangle(image,(80,y),(180,y+35),(0,0,200),-1)
        self.assertEqual(len(red_can_candidates(image)),2)
    def test_center_plane_depends_on_orientation(self):
        self.assertEqual(can_center_height(66,122,on_side=True),33)
        self.assertEqual(can_center_height(66,122,on_side=False),61)
        with self.assertRaises(ValueError):can_center_height(float('nan'),122,on_side=True)
