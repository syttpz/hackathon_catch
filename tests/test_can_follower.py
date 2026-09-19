import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
import numpy as np
from viam.proto.common import Pose
from motion.live_camera_pose import PoseBuffer,pose_matrix,matrix_pose,mix_rotation
from motion.can_follower import LatestMove,next_target,visible,can_grab,start_pose,move_to_start

CONFIG=json.loads((Path(__file__).parents[1]/'can_follow.config.json').read_text())
K=NS(fx=600,fy=600,cx=424,cy=240,width=848,height=480)

class CameraPoseTests(unittest.TestCase):
    def test_orientation_roundtrip_including_poles(self):
        for p in [Pose(o_z=1,theta=90),Pose(o_z=-1,theta=31.2),Pose(o_x=.023,o_y=-.954,o_z=-.3,theta=-178.8)]:
            t,r=pose_matrix(p)
            _,actual=pose_matrix(matrix_pose(t,r))
            np.testing.assert_allclose(actual,r,atol=1e-8)
            np.testing.assert_allclose(r[:,2],np.array([p.o_x,p.o_y,p.o_z])/np.linalg.norm([p.o_x,p.o_y,p.o_z]),atol=1e-8)
    def test_pose_bracketing_interpolates_camera_motion(self):
        buf=PoseBuffer()
        buf.add(1,1.002,Pose(x=0,o_z=1))
        buf.add(1.02,1.022,Pose(x=20,o_z=1,theta=10))
        p,r=buf.at(1.011)
        self.assertAlmostEqual(p[0],10)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-8)
        with self.assertRaises(ValueError):buf.at(.9)
    def test_slow_or_widely_spaced_feedback_rejected(self):
        buf=PoseBuffer();buf.add(0,.1,Pose(o_z=1));self.assertFalse(buf.samples)
        buf.add(1,1.002,Pose(o_z=1));buf.add(1.2,1.202,Pose(o_z=1))
        with self.assertRaises(ValueError):buf.at(1.1)
    def test_moving_camera_maps_stationary_object_consistently(self):
        point=np.array([20,30,10.])
        for x in (0,20,40):
            p,r=pose_matrix(Pose(x=x,z=300,o_z=-1,theta=45))
            local=r.T@(point-p)
            np.testing.assert_allclose(p+r@local,point,atol=1e-8)

class FollowGeometryTests(unittest.TestCase):
    def test_step_bounded_and_keeps_point_visible(self):
        current=pose_matrix(Pose(x=200,y=-300,z=300,o_z=-1))
        target=next_target(np.array([210,-310,32.5]),np.zeros(3),current,(np.zeros(3),np.eye(3)),K,CONFIG)
        self.assertIsNotNone(target)
        p,r=pose_matrix(target)
        self.assertLessEqual(np.linalg.norm(p-current[0]),60.001)
        self.assertTrue(visible(np.array([210,-310,32.5]),p,r,np.zeros(3),np.eye(3),K))
    def test_blind_candidate_rejected(self):
        current=pose_matrix(Pose(x=200,y=-300,z=300,o_z=1))
        self.assertIsNone(next_target(np.array([210,-310,32.5]),np.zeros(3),current,(np.zeros(3),np.eye(3)),K,CONFIG))
    def test_capture_requires_close_downward_low_pose(self):
        self.assertTrue(can_grab(np.array([200,-300,32.5]),pose_matrix(Pose(x=200,y=-300,z=175,o_z=-1)),CONFIG))
        self.assertFalse(can_grab(np.array([200,-300,32.5]),pose_matrix(Pose(x=200,y=-300,z=250,o_z=-1)),CONFIG))

class LatestMoveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.arm=AsyncMock();self.arm.is_moving.return_value=False
        self.motion=AsyncMock();self.mover=LatestMove(self.arm,self.motion,'arm','world')
    async def test_replacement_cancels_and_stops_before_new_command(self):
        events=[]
        async def move(**kwargs):
            events.append('move')
            try:await asyncio.Event().wait()
            finally:events.append('cancel')
        async def stop(**kwargs):events.append('stop')
        self.motion.move.side_effect=move;self.arm.stop.side_effect=stop
        await self.mover.update(Pose(x=100,z=175,o_z=-1))
        await asyncio.sleep(.01)
        self.mover.last_submit-=1
        await self.mover.update(Pose(x=200,z=175,o_z=-1))
        await asyncio.sleep(.01)
        self.assertEqual(events[:4],['move','cancel','stop','move'])
        await self.mover.stop()
    async def test_failed_plan_propagates(self):
        self.motion.move.return_value=False
        await self.mover.update(Pose(x=100,z=175,o_z=-1));await asyncio.sleep(.01)
        with self.assertRaises(RuntimeError):await self.mover.check()
        await self.mover.stop()
    async def test_small_update_does_not_stack_commands(self):
        self.motion.move.side_effect=lambda **kwargs: asyncio.sleep(1)
        # A true async function avoids returning an unawaited coroutine from AsyncMock.
        async def move(**kwargs):await asyncio.sleep(1);return True
        self.motion.move.side_effect=move
        await self.mover.update(Pose(x=100,z=175,o_z=-1));await asyncio.sleep(.01)
        self.mover.last_submit-=1
        self.assertFalse(await self.mover.update(Pose(x=105,z=175,o_z=-1)))
        self.motion.move.assert_awaited_once();await self.mover.stop()

class StartPoseTests(unittest.IsolatedAsyncioTestCase):
    def test_configured_start_pose_is_inside_the_follow_workspace(self):
        pose=start_pose(CONFIG)
        p,_=pose_matrix(pose)
        lower,upper=np.asarray(CONFIG['workspace_mm'],float)
        self.assertTrue(((p>=lower)&(p<=upper)).all())
        reference=CONFIG['start_pose_mm']
        np.testing.assert_allclose(p,[reference[k] for k in ('x','y','z')],atol=1e-9)
        _,expected=pose_matrix(Pose(**reference))
        np.testing.assert_allclose(pose_matrix(pose)[1],expected,atol=1e-8)
    def test_start_pose_outside_bounds_or_malformed_is_rejected(self):
        self.assertRaises(ValueError,start_pose,{k:v for k,v in CONFIG.items() if k!='start_pose_mm'})
        for broken in [{'start_pose_mm':'238,-53,223'},{'start_pose_mm':{'x':0}},
                       {'start_pose_mm':dict(CONFIG['start_pose_mm'],z=5000)},
                       {'start_pose_mm':dict(CONFIG['start_pose_mm'],o_x=0,o_y=0,o_z=0)},
                       {'start_pose_mm':dict(CONFIG['start_pose_mm'],x=float('nan'))}]:
            with self.assertRaises(ValueError):start_pose({**CONFIG,**broken})
    async def test_move_to_start_verifies_arrival(self):
        arm=AsyncMock();motion=AsyncMock();motion.move.return_value=True
        goal=start_pose(CONFIG)
        motion.get_pose.return_value=NS(reference_frame='world',pose=goal)
        offset,angle=await move_to_start(arm,motion,CONFIG,goal)
        self.assertLess(offset,1e-6);self.assertLess(angle,1e-6)
        arm.stop.assert_not_awaited()
    async def test_short_arrival_or_failed_plan_stops_the_arm(self):
        goal=start_pose(CONFIG)
        arm=AsyncMock();motion=AsyncMock();motion.move.return_value=False
        motion.get_pose.return_value=NS(reference_frame='world',pose=goal)
        with self.assertRaises(RuntimeError):await move_to_start(arm,motion,CONFIG,goal)
        arm.stop.assert_awaited()
        arm=AsyncMock();motion=AsyncMock();motion.move.return_value=True
        short=Pose(**dict(CONFIG['start_pose_mm'],x=CONFIG['start_pose_mm']['x']+50))
        motion.get_pose.return_value=NS(reference_frame='world',pose=short)
        with self.assertRaises(ValueError):await move_to_start(arm,motion,CONFIG,goal)
        arm.stop.assert_awaited()
    async def test_wrong_arrival_frame_rejected(self):
        goal=start_pose(CONFIG)
        arm=AsyncMock();motion=AsyncMock();motion.move.return_value=True
        motion.get_pose.return_value=NS(reference_frame='arm',pose=goal)
        with self.assertRaises(ValueError):await move_to_start(arm,motion,CONFIG,goal)
        arm.stop.assert_awaited()

class FailureToleranceTests(unittest.IsolatedAsyncioTestCase):
    def mover(self, **kwargs):
        arm = AsyncMock(); arm.is_moving.return_value = False
        motion = AsyncMock(); motion.move.return_value = False
        return arm, LatestMove(arm, motion, 'arm', 'world', **kwargs)

    async def test_tracking_loop_counts_planner_failures_and_keeps_going(self):
        _, mover = self.mover(period=0, deadband=1, strict=False)
        for expected in (1, 2):
            await mover.update(Pose(x=100*expected, z=300, o_z=-1))
            await asyncio.sleep(.01)
            await mover.check()
            self.assertEqual(mover.failures, expected)
        self.assertIsInstance(mover.last_error, RuntimeError)
        await mover.stop()

    async def test_a_raised_move_error_is_also_absorbed_when_not_strict(self):
        arm, mover = self.mover(period=0, deadband=1, strict=False)
        async def boom(**kwargs): raise asyncio.TimeoutError('planner timed out')
        mover.motion.move.side_effect = boom
        await mover.update(Pose(x=100, z=300, o_z=-1))
        await asyncio.sleep(.01)
        await mover.check()
        self.assertEqual(mover.failures, 1)
        await mover.stop()

    async def test_strict_default_still_raises_for_the_one_shot_follower(self):
        _, mover = self.mover()
        await mover.update(Pose(x=100, z=300, o_z=-1))
        await asyncio.sleep(.01)
        with self.assertRaises(RuntimeError):
            await mover.check()
        await mover.stop()
