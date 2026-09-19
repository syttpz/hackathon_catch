import asyncio
import unittest
from unittest.mock import AsyncMock, patch
import numpy as np
from viam.proto.common import Pose
from motion.catch_plane import DEFAULT_POSE, return_default
from motion.live_camera_pose import pose_matrix

class DefaultReturnTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_only_after_verified_pose(self):
        arm=AsyncMock(); expected=pose_matrix(Pose(**DEFAULT_POSE))
        with patch('motion.catch_plane.move_flange',new=AsyncMock(return_value=(True,None))) as move, patch('motion.catch_plane.read_flange',new=AsyncMock(return_value=expected)):
            position,_=await return_default(arm,object(),{})
            np.testing.assert_allclose(position,expected[0])
            self.assertEqual(move.call_args.args[1]['move_timeout_s'],15.)
            arm.stop.assert_not_called()
    async def test_failed_move_stops_and_raises(self):
        arm=AsyncMock()
        with patch('motion.catch_plane.move_flange',new=AsyncMock(return_value=(False,'refused'))):
            with self.assertRaisesRegex(RuntimeError,'refused'):
                await return_default(arm,object(),{})
        arm.stop.assert_awaited_once()
    async def test_pose_mismatch_stops_and_raises(self):
        arm=AsyncMock();p,r=pose_matrix(Pose(**DEFAULT_POSE))
        with patch('motion.catch_plane.move_flange',new=AsyncMock(return_value=(True,None))), patch('motion.catch_plane.read_flange',new=AsyncMock(return_value=(p+[10,0,0],r))):
            with self.assertRaisesRegex(RuntimeError,'not reached'):
                await return_default(arm,object(),{})
        arm.stop.assert_awaited_once()
    async def test_cancel_stops(self):
        arm=AsyncMock()
        with patch('motion.catch_plane.move_flange',new=AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await return_default(arm,object(),{})
        arm.stop.assert_awaited_once()
