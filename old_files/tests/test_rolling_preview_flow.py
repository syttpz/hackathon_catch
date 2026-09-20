"""Exercise the actual camera-loop handoff, not just the movement helper."""
import json
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import numpy as np
from viam.proto.common import Pose, PoseInFrame
from old_files.rolling import rolling_preview as preview

class FirstPredictionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_target_moves_with_154_seconds_and_no_stable_history(self):
        config_path=Path(__file__).parents[1]/'rolling'/'rolling_catch.config.json'
        config=json.loads(config_path.read_text())
        target=np.mean(config['catch_line_end_effector_mm'],axis=0)
        arm=AsyncMock()
        arm.is_moving.return_value=False
        arm.get_joint_positions.return_value=NS(values=[0,0,0,0,0,0])
        pose=Pose(x=238,y=-53,z=223,o_z=-1)
        arm.get_end_position.return_value=pose
        motion=AsyncMock()
        motion.get_pose.return_value=PoseInFrame(reference_frame='world',pose=pose)
        camera=AsyncMock()
        camera.get_point_cloud.return_value=(b'pcd','pcd')
        camera.get_properties.return_value=NS(intrinsic_parameters=NS(
            focal_x_px=100,focal_y_px=100,center_x_px=0,center_y_px=0,width_px=1,height_px=1))
        timestamp=time.time()
        camera.get_images.return_value=([NS(name='color')],NS(captured_at=NS(
            seconds=int(timestamp),nanos=int((timestamp%1)*1e9))))
        connection=AsyncMock()
        mover=AsyncMock(return_value=True)
        catcher=AsyncMock()
        args=NS(config=config_path,machine_config=None,execute=True,move_on_prediction=True,
                duration=1,horizon=None,move_budget=None,close_lead=None)
        with ExitStack() as stack:
            replacements=[
                (preview,'credentials',lambda _:('00000000-0000-0000-0000-000000000001','test-key')),
                (preview.RobotClient,'at_address',AsyncMock(return_value=connection)),
                (preview.Arm,'from_robot',lambda *_:arm),
                (preview.Camera,'from_robot',lambda *_:camera),
                (preview.Gripper,'from_robot',lambda *_:AsyncMock()),
                (preview.MotionClient,'from_robot',lambda *_:motion),
                (preview,'camera_transform',AsyncMock(return_value=(np.zeros(3),np.eye(3)))),
                (preview,'table_plane',lambda *_:NS(normal=np.array([0,0,1]),support=.8,residual_mm=1)),
                (preview,'prepare_gripper',AsyncMock()),
                (preview,'decode_color',lambda _:np.zeros((1,1,3))),
                (preview,'locate_object',lambda *_:(np.array([100,-600,30]),None)),
                (preview.RollingFit,'add',lambda *_:(np.array([100,-600,30]),np.array([0,300,0]),1)),
                (preview,'line_crossing',lambda *_:(1.54,.5,target)),
                (preview,'move_to_prediction',mover),
                (preview,'execute_catch',catcher),
            ]
            for obj,name,value in replacements:stack.enter_context(patch.object(obj,name,value))
            result=await preview.run(args)
        self.assertTrue(result)
        mover.assert_awaited_once()
        np.testing.assert_allclose(mover.call_args.args[-1],target)
        catcher.assert_not_called()
        camera.get_images.assert_awaited_once()
