"""Park the arm in the catch zone, wrist camera facing the thrower.

The park pose does double duty: the arm must be able to reach the interception
point from here, and the WRIST CAMERA must see the incoming ball, because it is
the tracker. Edit `target` to match where the thrower actually stands -- the
orientation below points the camera along world -Y, copied from default2.

Checked against catch_throw.config.json before moving.
"""
import asyncio
import json
from pathlib import Path

import numpy as np
from connection import connect
from viam.components.arm import Arm
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame

from motion.ballistic import reachable


async def main():
    config = json.loads((Path(__file__).parents[1]/'catch_throw.config.json').read_text())
    target = Pose(
        x=380.0,
        y=-60.0,
        z=500.0,
        # Camera looking along world -Y, as taught for default2.
        o_x=0.023064251413913939,
        o_y=-0.95371981066102718,
        o_z=-0.29981087882098378,
        theta=-178.81110311955516,
    )
    position = np.array([target.x, target.y, target.z])
    if not reachable(position, config):
        raise SystemExit(
            f'Park pose {position.tolist()} mm (reach {np.linalg.norm(position):.0f} mm) is outside '
            f'catch_box_mm {config["catch_box_mm"]} / reach '
            f'{config["min_reach_mm"]}-{config["max_reach_mm"]} mm. Edit it or the config.')

    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")
        current = await arm.get_end_position()
        print(f'Current: ({current.x:.1f}, {current.y:.1f}, {current.z:.1f}) mm')
        print(f'Parking: ({target.x:.1f}, {target.y:.1f}, {target.z:.1f}) mm '
              f'| reach {np.linalg.norm(position):.0f} mm')
        # arm.move_to_position drives ServoJ, which refuses a large repositioning
        # ("Linear speed exceeds limit in ServoJ mode"). Parking is a one-off big
        # move, so plan it properly; the latency that costs does not matter here.
        motion = MotionClient.from_robot(machine, "builtin")
        ok = await motion.move(component_name="arm",
                               destination=PoseInFrame(reference_frame="world", pose=target),
                               timeout=60)
        if not ok:
            raise SystemExit('Planner did not complete the park move')
        landed = await arm.get_end_position()
        error = np.linalg.norm([landed.x-target.x, landed.y-target.y, landed.z-target.z])
        print(f'Parked at ({landed.x:.1f}, {landed.y:.1f}, {landed.z:.1f}) mm, {error:.1f} mm off.')
        print('The thrower must stand where this camera can see them.')


if __name__ == "__main__":
    asyncio.run(main())
