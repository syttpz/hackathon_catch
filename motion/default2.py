import asyncio

from connection import connect
from viam.components.arm import Arm
from viam.proto.common import Pose


async def main():
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")

        current = await arm.get_end_position()
        print(f"Current end position: {current}")

        target = Pose(
            x = 238.07014127025724,
            y = -52.916527852005913,
            z = 223.1463208934436,
            o_x = 0.023064251413913939,
            o_y = -0.95371981066102718,
            o_z = -0.29981087882098378,
            theta = -178.81110311955516
        )

        await arm.move_to_position(pose=target)
        print(f"Moved to: {target}")


if __name__ == "__main__":
    asyncio.run(main())




















