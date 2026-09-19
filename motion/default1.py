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
            x= -392.5960289174601,
            y= -90.034134523461546,
            z= 294.29301759794976,
            o_x= -0.99527936854834231,
            o_y= 0.08567952441382401,
            o_z= 0.045585059375129855,
            theta= -178.77187102202018
        )

        await arm.move_to_position(pose=target)
        print(f"Moved to: {target}")


if __name__ == "__main__":
    asyncio.run(main())