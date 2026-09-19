"""Can-specific image candidates; not yet connected to physical motion.

A can on its side is not circular in the image. Its red label may be broken by
white lettering. These candidates still require calibrated size/plane filtering.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class CanCandidate:
    center_px: tuple[float, float]
    short_side_px: float
    long_side_px: float
    angle_deg: float


def red_can_candidates(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(0,180,45),(10,255,255))
    mask|=cv2.inRange(hsv,(170,180,45),(179,255,255))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<100: continue
        (u,v),(w,h),angle=cv2.minAreaRect(contour)
        short,long=sorted((w,h))
        if short<6 or long/short>5 or area/(w*h)<.55: continue
        x,y,bw,bh=cv2.boundingRect(contour)
        if x<=0 or y<=0 or x+bw>=bgr.shape[1] or y+bh>=bgr.shape[0]: continue
        candidates.append(CanCandidate((u,v),short,long,angle))
    return candidates


def can_center_height(diameter_mm, length_mm, *, on_side):
    dimensions=np.asarray([diameter_mm,length_mm],float)
    if not np.isfinite(dimensions).all() or (dimensions<=0).any():
        raise ValueError('Measure a positive finite can diameter and length in mm')
    return float(diameter_mm/2 if on_side else length_mm/2)
