import asyncio
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.generic import Generic as GenericService

load_dotenv()


async def connect():
    api_key = os.getenv("VIAM_API_KEY")
    api_key_id = os.getenv("VIAM_API_KEY_ID")
    address = os.getenv("VIAM_ADDRESS")

    if not api_key or not api_key_id or not address:
        raise RuntimeError(
            "Missing VIAM_API_KEY, VIAM_API_KEY_ID, or VIAM_ADDRESS. "
            "Copy .env.example to .env and fill in your credentials."
        )

    opts = RobotClient.Options.with_api_key(
        api_key=api_key,
        api_key_id=api_key_id,
    )

    return await RobotClient.at_address(address, opts)


async def main():
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)

        # arm
        arm = Arm.from_robot(machine, "arm")
        arm_return_value = await arm.get_end_position()
        print(f"arm get_end_position return value: {arm_return_value}")

        # cam
        cam = Camera.from_robot(machine, "cam")
        cam_return_value = await cam.get_images()
        print(f"cam get_images return value: {cam_return_value}")

        # gripper
        gripper = Gripper.from_robot(machine, "gripper")
        gripper_return_value = await gripper.is_moving()
        print(f"gripper is_moving return value: {gripper_return_value}")

        # table
        table = Gripper.from_robot(machine, "table")
        table_return_value = await table.is_moving()
        print(f"table is_moving return value: {table_return_value}")

        # wall-front
        wall_front = Gripper.from_robot(machine, "wall-front")
        wall_front_return_value = await wall_front.is_moving()
        print(f"wall-front is_moving return value: {wall_front_return_value}")

        # wall-side
        wall_side = Gripper.from_robot(machine, "wall-side")
        wall_side_return_value = await wall_side.is_moving()
        print(f"wall-side is_moving return value: {wall_side_return_value}")

        # ceiling
        ceiling = Gripper.from_robot(machine, "ceiling")
        ceiling_return_value = await ceiling.is_moving()
        print(f"ceiling is_moving return value: {ceiling_return_value}")

if __name__ == '__main__':
    asyncio.run(main())
