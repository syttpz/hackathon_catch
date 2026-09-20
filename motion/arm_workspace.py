"""Workspace checks and low-latency movement shared by active robot tools."""

import asyncio

import numpy as np


def clamp_target(goal, config):
    bounds = np.array(config["workspace_mm"], float)
    goal = np.clip(goal, bounds[0], bounds[1])
    reach = float(np.linalg.norm(goal))
    if reach > config["max_reach_mm"]:
        goal = goal * config["max_reach_mm"] / reach
        goal = np.clip(goal, bounds[0], bounds[1])
    elif reach < config["min_reach_mm"]:
        if reach < 1e-6:
            raise ValueError("Target collapsed onto the arm base")
        goal = goal * config["min_reach_mm"] / reach
        goal = np.clip(goal, bounds[0], bounds[1])
    return goal


def in_workspace(position, config):
    bounds = np.array(config["workspace_mm"], float)
    reach = float(np.linalg.norm(position))
    return bool(
        (position >= bounds[0]).all()
        and (position <= bounds[1]).all()
        and config["min_reach_mm"] <= reach <= config["max_reach_mm"]
    )


class DirectArmMove:
    """Adapt ``arm.move_to_position`` to the MotionClient.move interface."""

    def __init__(self, arm):
        self.arm = arm

    async def move(self, *, component_name, destination, timeout):
        await asyncio.wait_for(
            self.arm.move_to_position(pose=destination.pose, timeout=timeout),
            timeout,
        )
        return True
