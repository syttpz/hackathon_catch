import asyncio
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock
from viam.proto.common import Pose, PoseInFrame
from motion.rolling_catch import PredictionGate, execute_catch, minimum_lead

CONFIG=json.loads((Path(__file__).parents[1]/'rolling_catch.config.json').read_text())
TARGET=[sum(v)/2 for v in zip(*CONFIG['catch_line_end_effector_mm'])]

class GateTests(unittest.TestCase):
    def test_stable_prediction_and_reset(self):
        gate=PredictionGate()
        results=[gate.add(i*.02,TARGET,10) for i in range(12)]
        self.assertFalse(results[0]);self.assertTrue(results[-1])
        gate.reset();self.assertFalse(gate.add(.25,TARGET,10))
    def test_drifting_arrival_rejected(self):
        gate=PredictionGate()
        results=[gate.add(i*.02,TARGET,10+i*.1) for i in range(12)]
        self.assertFalse(any(results))
    def test_large_gap_restarts_confirmation(self):
        gate=PredictionGate()
        for i in range(12): gate.add(i*.02,TARGET,10)
        self.assertFalse(gate.add(1,TARGET,10))

class CatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.arm=AsyncMock();self.gripper=AsyncMock();self.motion=AsyncMock()
        self.motion.move.return_value=True
        self.gripper.grab.return_value=True
        self.motion.get_pose.return_value=PoseInFrame(reference_frame='world',pose=Pose(
            x=TARGET[0],y=TARGET[1],z=TARGET[2],o_z=-1,theta=CONFIG['candidate_orientation'][3]))
        self.clock=100.0
        self.events=[]
        async def open_(**kwargs): self.events.append('open');self.clock+=.2
        async def move_(**kwargs): self.events.append('move');self.clock+=1;return True
        async def grab_(**kwargs):self.events.append('grab');return self.gripper.grab.return_value
        self.gripper.open.side_effect=open_;self.motion.move.side_effect=move_;self.gripper.grab.side_effect=grab_
    async def sleep(self,delay):self.events.append('wait');self.clock+=delay
    async def run_catch(self,arrival=104,target=None):
        return await execute_catch(self.arm,self.gripper,self.motion,CONFIG,target or TARGET,arrival,now=lambda:self.clock,sleep=self.sleep)
    async def test_success_targets_arm_flange_and_times_close(self):
        self.assertTrue(await self.run_catch())
        self.assertEqual(self.events,['open','move','wait','grab'])
        self.assertAlmostEqual(self.clock,104-CONFIG['close_lead_s'])
        kwargs=self.motion.move.call_args.kwargs
        self.assertEqual(kwargs['component_name'],'arm');self.assertEqual(kwargs['destination'].pose.z,175)
        self.arm.stop.assert_not_called()
    async def test_miss_is_not_success(self):
        self.gripper.grab.return_value=False
        self.assertFalse(await self.run_catch())
    async def test_late_rejected_before_commands(self):
        with self.assertRaises(ValueError): await self.run_catch(101)
        self.gripper.open.assert_not_called();self.motion.move.assert_not_called()
    async def test_off_line_rejected(self):
        with self.assertRaises(ValueError): await self.run_catch(target=[0,0,175])
        self.gripper.open.assert_not_called()
    async def test_planner_failure_stops_both(self):
        self.motion.move.side_effect=None;self.motion.move.return_value=False
        with self.assertRaises(RuntimeError):await self.run_catch()
        self.arm.stop.assert_awaited_once();self.gripper.stop.assert_awaited_once();self.gripper.grab.assert_not_called()
    async def test_bad_arrival_stops_without_closing(self):
        self.motion.get_pose.return_value.pose.x+=100
        with self.assertRaises(ValueError):await self.run_catch()
        self.gripper.grab.assert_not_called();self.arm.stop.assert_awaited_once()
    async def test_late_arrival_never_grabs(self):
        async def slow(**kwargs):self.clock+=4;return True
        self.motion.move.side_effect=slow
        with self.assertRaises(TimeoutError):await self.run_catch()
        self.gripper.grab.assert_not_called();self.arm.stop.assert_awaited_once()
    async def test_cancel_during_open_stops_both(self):
        self.gripper.open.side_effect=asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):await self.run_catch()
        self.arm.stop.assert_awaited_once();self.gripper.stop.assert_awaited_once()
    async def test_scheduler_lag_never_grabs(self):
        async def late_sleep(delay):self.clock+=delay+.1
        with self.assertRaises(TimeoutError):
            await execute_catch(self.arm,self.gripper,self.motion,CONFIG,TARGET,104,now=lambda:self.clock,sleep=late_sleep)
        self.gripper.grab.assert_not_called()
    async def test_preopened_can_catch_with_two_seconds_and_does_not_reopen(self):
        self.assertAlmostEqual(minimum_lead(CONFIG,gripper_preopened=True),1.8)
        result=await execute_catch(self.arm,self.gripper,self.motion,CONFIG,TARGET,102,
            gripper_preopened=True,now=lambda:self.clock,sleep=self.sleep)
        self.assertTrue(result)
        self.gripper.open.assert_not_called()
        self.assertEqual(self.events,['move','wait','grab'])
    async def test_preopened_still_rejects_insufficient_move_time(self):
        with self.assertRaises(ValueError):
            await execute_catch(self.arm,self.gripper,self.motion,CONFIG,TARGET,101.5,
                gripper_preopened=True,now=lambda:self.clock,sleep=self.sleep)
        self.motion.move.assert_not_called()
    async def test_startup_open_failure_stops_gripper(self):
        from motion.rolling_catch import prepare_gripper
        self.gripper.open.side_effect=TimeoutError('open failed')
        with self.assertRaises(TimeoutError):await prepare_gripper(self.gripper,CONFIG)
        self.gripper.stop.assert_awaited_once()
        self.motion.move.assert_not_called()
    async def test_move_only_does_not_wait_or_grab(self):
        from motion.rolling_catch import move_to_prediction
        self.assertTrue(await move_to_prediction(self.arm,self.gripper,self.motion,CONFIG,TARGET))
        self.assertEqual(self.events,['move'])
        self.gripper.grab.assert_not_called()
        self.assertEqual(self.motion.move.call_args.kwargs['destination'].pose.z,175)
    async def test_move_only_failure_stops(self):
        from motion.rolling_catch import move_to_prediction
        self.motion.move.side_effect=None;self.motion.move.return_value=False
        with self.assertRaises(RuntimeError):
            await move_to_prediction(self.arm,self.gripper,self.motion,CONFIG,TARGET)
        self.arm.stop.assert_awaited_once();self.gripper.grab.assert_not_called()
