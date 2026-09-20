"""Robust, near-horizontal plane estimation for the read-only rolling preview."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TablePlane:
    # Unit upward normal; plane equation normal @ point = offset_mm.
    normal: np.ndarray
    offset_mm: float
    support: float
    residual_mm: float

    def height_at(self, x, y):
        return (self.offset_mm-self.normal[0]*x-self.normal[1]*y)/self.normal[2]


def fit_table_plane(points):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('Expected XYZ point cloud')
    points=points[np.isfinite(points).all(axis=1)&(abs(points[:,2])<150)]
    if len(points)<300:
        raise ValueError(f'Insufficient table points near world Z +/-150 mm: {len(points)}')
    rng=np.random.default_rng(7)
    sample=points[rng.choice(len(points),min(4000,len(points)),replace=False)]
    min_vertical=np.cos(np.deg2rad(15))
    best_count=0
    best=None
    for _ in range(400):
        a,b,c=sample[rng.choice(len(sample),3,replace=False)]
        normal=np.cross(b-a,c-a)
        length=np.linalg.norm(normal)
        if length<1e-6: continue
        normal/=length
        if normal[2]<0: normal=-normal
        if normal[2]<min_vertical: continue
        offset=float(normal@a)
        count=int(np.sum(abs(sample@normal-offset)<5))
        if count>best_count:
            best_count=count
            best=(normal,offset)
    if best is None:
        raise ValueError('No near-horizontal table plane found (maximum tilt 15 degrees)')
    normal,offset=best
    for _ in range(3):
        inliers=points[abs(points@normal-offset)<5]
        if len(inliers)<300: raise ValueError('Too few table-plane inliers')
        center=np.mean(inliers,axis=0)
        _,_,vectors=np.linalg.svd(inliers-center,full_matrices=False)
        normal=vectors[-1]
        if normal[2]<0: normal=-normal
        offset=float(normal@center)
    inliers=points[abs(points@normal-offset)<5]
    support=len(inliers)/len(points)
    if support<.25 or len(inliers)<300 or normal[2]<min_vertical:
        raise ValueError(f'Table-plane confidence too low: {support:.0%} support; need 25% and tilt <=15 degrees')
    # Reject a small object or narrow strip even when it contains many pixels.
    spread=np.linalg.svd(inliers-inliers.mean(axis=0),compute_uv=False)/np.sqrt(len(inliers))
    if spread[1]<40:
        raise ValueError('Visible table patch too narrow to calibrate reliably')
    residual=float(np.median(abs(inliers@normal-offset)))
    if residual>3: raise ValueError(f'Table plane too noisy: {residual:.1f} mm residual')
    return TablePlane(normal,offset,support,residual)


class InvalidProjection(ValueError):
    """An image candidate does not project onto the visible ball-center plane."""


def point_on_offset_plane(pixel, intrinsics, origin, rotation, plane, radius_mm):
    """Intersect a color ray with the plane one ball radius above the table."""
    ray=np.array([(pixel[0]-intrinsics.cx)/intrinsics.fx,
                  (pixel[1]-intrinsics.cy)/intrinsics.fy,1.0])
    direction=rotation@ray
    denominator=float(plane.normal@direction)
    if not np.isfinite(direction).all() or abs(denominator)<1e-6:
        raise InvalidProjection('Viewing ray is invalid or parallel to table')
    distance=(plane.offset_mm+radius_mm-plane.normal@origin)/denominator
    if not np.isfinite(distance) or distance<=0:
        raise InvalidProjection('Candidate does not intersect table in front of camera')
    return origin+distance*direction
