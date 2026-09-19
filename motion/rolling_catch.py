"""One-shot rolling interception. Called only when --execute is explicit."""
import asyncio
import math
import time
from collections import deque

import numpy as np
from viam.proto.common import Pose, PoseInFrame


class PredictionGate:
    """Require a stable point AND absolute arrival time over multiple frames.

    The defaults suit a gripper, whose jaws must close on the object within
    centimetres. A wide catcher can accept a looser gate: demanding 15 mm of
    agreement to aim a 90 mm bowl rejects predictions that were always good
    enough.
    """
    def __init__(self, tolerance_mm=15., arrival_spread_s=.12, min_samples=6, min_span_s=.15):
        self.tolerance_mm = tolerance_mm
        self.arrival_spread_s = arrival_spread_s
        self.min_samples = min_samples
        self.min_span_s = min_span_s
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def add(self, now, target, arrival):
        target = np.asarray(target, float)
        if not np.isfinite(target).all() or not math.isfinite(arrival):
            self.reset()
            return False
        if self.samples and (now-self.samples[-1][0] > .1 or now <= self.samples[-1][0]):
            self.reset()
        self.samples.append((now, target, arrival))
        while self.samples and now-self.samples[0][0] > .3:
            self.samples.popleft()
        if len(self.samples) < self.min_samples or now-self.samples[0][0] < self.min_span_s:
            return False
        targets = np.array([s[1] for s in self.samples])
        arrivals = np.array([s[2] for s in self.samples])
        # Agree with the median rather than with every sample. Requiring all of
        # them let one wild frame veto a run of consistent predictions, which is
        # exactly what a short noisy window produces.
        centre = np.median(targets, axis=0)
        spread = np.linalg.norm(targets-centre, axis=1)
        agreeing = int(np.sum(spread <= self.tolerance_mm))
        return bool(agreeing >= self.min_samples
                    and np.linalg.norm(target-centre) <= self.tolerance_mm
                    and np.ptp(np.sort(arrivals)[:max(2, agreeing)]) <= self.arrival_spread_s)

    def consensus(self):
        """Median of the recent targets: steadier than the newest single frame."""
        return np.median(np.array([s[1] for s in self.samples]), axis=0)

    def progress(self):
        return len(self.samples), self.min_samples


def validate_catch_config(config):
    line = np.asarray(config['catch_line_end_effector_mm'], float)
    orientation = np.asarray(config['candidate_orientation'], float)
    if line.shape != (2,3) or not np.isfinite(line).all() or np.linalg.norm(line[1,:2]-line[0,:2]) < 1:
        raise ValueError('Catch line must contain two distinct finite XYZ endpoints')
    if not np.allclose(line[:,2], 175):
        raise ValueError('This catch setup requires taught flange Z=175 mm')
    if orientation.shape != (4,) or not np.isfinite(orientation).all() or not np.allclose(orientation[:3], [0,0,-1]):
        raise ValueError('This catch setup requires a vertically downward orientation')
    for name in ['move_budget_s','open_budget_s','close_lead_s','arrival_margin_s','arrival_tolerance_mm','rpc_timeout_s','max_prediction_s']:
        value = float(config[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    required = minimum_lead(config, gripper_preopened=True)
    if config['max_prediction_s'] <= required:
        raise ValueError(f'Prediction horizon must exceed minimum catch lead {required:.2f}s')


def minimum_lead(config, *, gripper_preopened=False):
    keys = ['move_budget_s','close_lead_s','arrival_margin_s']
    if not gripper_preopened:
        keys.append('open_budget_s')
    return sum(float(config[k]) for k in keys)


async def prepare_gripper(gripper, config):
    """Open before tracking begins, outside the ball's interception deadline."""
    print('PREPARING: opening gripper; wait before rolling the ball.', flush=True)
    try:
        await asyncio.wait_for(gripper.open(timeout=config['rpc_timeout_s']), config['rpc_timeout_s'])
    except BaseException:
        try:
            await asyncio.shield(asyncio.wait_for(gripper.stop(timeout=3), 3))
        except Exception as error:
            print(f'STOP FAILED (gripper): {error}', flush=True)
        raise



def validate_target(target, config):
    target = np.asarray(target, float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Invalid catch target')
    a,b = np.asarray(config['catch_line_end_effector_mm'],float)
    delta = b-a
    u = float((target-a)@delta/(delta@delta))
    if target.shape != (3,) or not np.isfinite(target).all() or not 0 <= u <= 1 or np.linalg.norm(target-(a+u*delta)) > .1:
        raise ValueError('Rejected target outside the taught catch segment')
    return target


async def execute_catch(arm, gripper, motion, config, target, arrival, *, gripper_preopened=False, now=None, sleep=None):
    """Open, move early, verify flange pose, then close once. No lift or retry.

    All deadlines are monotonic seconds. Stop both actuators on a failure or
    cancellation after commands begin. Tracking must be stopped before calling.
    """
    now = now or time.monotonic
    sleep = sleep or asyncio.sleep
    validate_catch_config(config)
    target = validate_target(target, config)
    if not math.isfinite(arrival) or arrival-now() < minimum_lead(config, gripper_preopened=gripper_preopened):
        raise ValueError('Too late to reach this crossing; no command sent')
    ox,oy,oz,theta = config['candidate_orientation']
    destination = PoseInFrame(reference_frame=config['world_frame'],pose=Pose(
        x=float(target[0]),y=float(target[1]),z=float(target[2]),
        o_x=ox,o_y=oy,o_z=oz,theta=theta))
    commanded = False

    async def stop():
        results = await asyncio.gather(
            asyncio.wait_for(arm.stop(timeout=3),3),
            asyncio.wait_for(gripper.stop(timeout=3),3),return_exceptions=True)
        for component, result in zip(('arm','gripper'),results):
            if isinstance(result,BaseException):
                print(f'STOP FAILED ({component}): {result}',flush=True)

    try:
        commanded = True
        if not gripper_preopened:
            print('CATCH: opening gripper',flush=True)
            await asyncio.wait_for(gripper.open(timeout=config['open_budget_s']),config['open_budget_s'])
        close_at = arrival-config['close_lead_s']
        move_time = close_at-config['arrival_margin_s']-now()
        if move_time < config['move_budget_s']:
            raise TimeoutError('Insufficient remaining move budget after opening')
        print(f'CATCH: moving arm flange to {target.round(1).tolist()} mm; tracking frozen',flush=True)
        success = await asyncio.wait_for(motion.move(component_name=config['arm'],
            destination=destination,timeout=move_time),move_time)
        if not success:
            raise RuntimeError('Motion planner did not complete the catch move')
        verify_time = min(config['rpc_timeout_s'],close_at-now())
        if verify_time <= 0:
            raise TimeoutError('Arm arrived after the close deadline')
        # Compare in the same world frame as the planned flange target.
        actual = await asyncio.wait_for(motion.get_pose(config['arm'],config['world_frame'],timeout=verify_time),verify_time)
        if actual.reference_frame != config['world_frame']:
            raise ValueError('Unexpected frame in arrival verification')
        pose = actual.pose
        values = np.array([pose.x,pose.y,pose.z,pose.o_x,pose.o_y,pose.o_z,pose.theta],float)
        if not np.isfinite(values).all() or np.linalg.norm(values[:3]-target) > config['arrival_tolerance_mm'] or pose.o_z > -.98 or abs((pose.theta-theta+180)%360-180)>5:
            raise ValueError('Arm did not reach the required catching pose')
        remaining = close_at-now()
        if remaining < 0:
            raise TimeoutError('Missed closing deadline; gripper will not grab')
        print(f'CATCH: at line; closing in {remaining:.2f}s',flush=True)
        await sleep(remaining)
        if now()-close_at > .05:
            raise TimeoutError('Closing timer missed its deadline')
        grabbed = bool(await asyncio.wait_for(gripper.grab(timeout=config['rpc_timeout_s']),config['rpc_timeout_s']))
        print('CATCH: gripper reports an object' if grabbed else 'MISS: gripper reports no object',flush=True)
        return grabbed
    except BaseException:
        if commanded:
            await asyncio.shield(stop())
        raise


async def move_to_prediction(arm, gripper, motion, config, target):
    """Move once to the first valid target; deliberately do not time a grasp."""
    validate_catch_config(config)
    target = validate_target(target, config)
    ox,oy,oz,theta = config['candidate_orientation']
    destination = PoseInFrame(reference_frame=config['world_frame'],pose=Pose(
        x=float(target[0]),y=float(target[1]),z=float(target[2]),
        o_x=ox,o_y=oy,o_z=oz,theta=theta))
    timeout = 10.0
    try:
        print(f'MOVE ONLY: commanding flange {target.round(1).tolist()} mm now; gripper stays open',flush=True)
        success = await asyncio.wait_for(motion.move(component_name=config['arm'],
            destination=destination,timeout=timeout),timeout)
        if not success:
            raise RuntimeError('Motion planner rejected or failed the target move')
        actual = await asyncio.wait_for(motion.get_pose(config['arm'],config['world_frame'],timeout=3),3)
        pose = actual.pose
        xyz = np.array([pose.x,pose.y,pose.z],float)
        if actual.reference_frame != config['world_frame'] or not np.isfinite(xyz).all() or np.linalg.norm(xyz-target)>config['arrival_tolerance_mm']:
            raise ValueError('Arm did not reach the predicted target')
        print('POSITIONED: reached predicted point. Gripper remains open; no timed catch attempted.',flush=True)
        return True
    except BaseException:
        async def stop():
            results=await asyncio.gather(asyncio.wait_for(arm.stop(timeout=3),3),
                asyncio.wait_for(gripper.stop(timeout=3),3),return_exceptions=True)
            for name,result in zip(('arm','gripper'),results):
                if isinstance(result,BaseException): print(f'STOP FAILED ({name}): {result}',flush=True)
        await asyncio.shield(stop())
        raise
