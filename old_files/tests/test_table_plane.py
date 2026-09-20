import unittest
from types import SimpleNamespace
import numpy as np
from old_files.rolling.table_plane import fit_table_plane, point_on_offset_plane, TablePlane

class TablePlaneTests(unittest.TestCase):
    def test_tilted_table_with_majority_clutter(self):
        rng=np.random.default_rng(123)
        xy=rng.uniform([-400,-1000],[500,-200],size=(2400,2))
        z=.045*xy[:,1]+20+rng.normal(0,1,len(xy))
        table=np.column_stack((xy,z))
        clutter=rng.uniform([-500,-1200,-140],[700,100,140],size=(3600,3))
        plane=fit_table_plane(np.vstack((table,clutter)))
        self.assertAlmostEqual(plane.height_at(0,-600),-7,delta=1)
        self.assertGreater(plane.support,.35)
        self.assertLess(plane.residual_mm,2)
    def test_rejects_unstructured_cloud(self):
        rng=np.random.default_rng(3)
        with self.assertRaises(ValueError):
            fit_table_plane(rng.uniform(-140,140,(4000,3)))
    def test_rejects_narrow_patch(self):
        rng=np.random.default_rng(1)
        with self.assertRaises(ValueError):
            fit_table_plane(np.column_stack((rng.uniform(0,500,2000),rng.uniform(0,10,2000),rng.normal(0,.2,2000))))
    def test_sphere_offset_is_perpendicular_to_plane(self):
        normal=np.array([0,-.1,1]);normal/=np.linalg.norm(normal)
        plane=TablePlane(normal,10,.8,1)
        point=point_on_offset_plane([50,50],SimpleNamespace(fx=100,fy=100,cx=50,cy=50),np.array([10,20,300]),np.diag([1,-1,-1]),plane,30)
        self.assertAlmostEqual(float(normal@point),40)
    def test_candidate_behind_camera_is_rejected_specifically(self):
        from old_files.rolling.table_plane import InvalidProjection
        with self.assertRaises(InvalidProjection):
            point_on_offset_plane([50,50],SimpleNamespace(fx=100,fy=100,cx=50,cy=50),
                np.array([0,0,300]),np.eye(3),TablePlane(np.array([0,0,1]),0,.8,1),30)
    def test_candidate_at_horizon_is_rejected(self):
        from old_files.rolling.table_plane import InvalidProjection
        rotation=np.array([[1,0,0],[0,0,-1],[0,1,0]])
        with self.assertRaises(InvalidProjection):
            point_on_offset_plane([50,50],SimpleNamespace(fx=100,fy=100,cx=50,cy=50),
                np.array([0,0,300]),rotation,TablePlane(np.array([0,0,1]),0,.8,1),30)
