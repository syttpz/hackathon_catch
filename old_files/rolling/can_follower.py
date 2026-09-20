"""Bounded, latest-target arm following; no move queue and no line restriction."""
import asyncio
import time
import numpy as np
from viam.proto.common import PoseInFrame, Pose
from motion.live_camera_pose import matrix_pose, mix_rotation, pose_matrix


def visible(point,flange_position,flange_rotation,mount_position,mount_rotation,k,margin=20):
    camera_position=flange_position+flange_rotation@mount_position
    camera_rotation=flange_rotation@mount_rotation
    p=camera_rotation.T@(point-camera_position)
    if p[2]<=50:return False
    u=k.fx*p[0]/p[2]+k.cx;v=k.fy*p[1]/p[2]+k.cy
    return margin<u<k.width-margin and margin<v<k.height-margin


def next_target(point,velocity,current,mount,k,config):
    """Short bounded step toward contact, accepting only visible candidate poses."""
    p,r=current;mp,mr=mount
    # This is a short follow offset, not an arrival-time admission gate.
    aim=point+np.clip(velocity,-500,500)*.1
    goal=np.array([aim[0],aim[1],config['contact_flange_z_mm']])
    bounds=np.array(config['workspace_mm'],float)
    goal=np.clip(goal,*bounds)
    _,down=pose_matrix(Pose(o_z=-1,theta=config['grasp_theta_deg']))
    distance=np.linalg.norm(goal-p)
    angle=np.arccos(np.clip((np.trace(r.T@down)-1)/2,-1,1))
    fraction=min(1,config['max_step_mm']/max(distance,1e-9),np.deg2rad(config['max_rotation_step_deg'])/max(angle,1e-9))
    for factor in (1,.5,.25,.125):
        f=fraction*factor
        candidate=p+f*(goal-p);rotation=mix_rotation(r,down,f)
        if (candidate>=bounds[0]).all() and (candidate<=bounds[1]).all() and visible(point,candidate,rotation,mp,mr,k) and visible(aim,candidate,rotation,mp,mr,k):
            if np.linalg.norm(candidate-p)<2 and angle*f<np.deg2rad(1):return None
            return matrix_pose(candidate,rotation)
    return None


class LatestMove:
    def __init__(self,arm,motion,arm_name,world,period=.3,deadband=25,strict=True):
        self.arm=arm;self.motion=motion;self.arm_name=arm_name;self.world=world
        self.period=period;self.deadband=deadband;self.strict=strict
        self.task=None;self.target=None;self.last_submit=float('-inf');self.commanded=False
        self.failures=0;self.last_error=None
    async def settle(self):
        """Consume a finished move. Strict callers get the failure raised; a
        tracking loop counts it and keeps following instead of dying."""
        if self.task is None or not self.task.done():return True
        task=self.task;self.task=None
        try:
            if not bool(await task):raise RuntimeError('Viam planner failed the follow move')
        except asyncio.CancelledError:raise
        except Exception as error:
            self.failures+=1;self.last_error=error
            if self.strict:raise
            return False
        return True
    async def stop(self):
        if self.task is not None:
            task=self.task;self.task=None
            task.cancel()
            try: await asyncio.wait_for(asyncio.gather(task,return_exceptions=True),3)
            finally: await asyncio.wait_for(self.arm.stop(timeout=3),3)
        elif self.commanded:
            await asyncio.wait_for(self.arm.stop(timeout=3),3)
        if self.commanded:
            deadline=time.monotonic()+1.5
            while await asyncio.wait_for(self.arm.is_moving(timeout=.3),.3):
                if time.monotonic()>deadline:raise RuntimeError('Arm did not acknowledge stopped state')
                await asyncio.sleep(.03)
        self.target=None;self.commanded=False
    async def update(self,pose):
        await self.settle()
        if time.monotonic()-self.last_submit<self.period:return False
        p=np.array([pose.x,pose.y,pose.z])
        if self.task is not None:
            if np.linalg.norm(p-self.target)<self.deadband:return False
            await self.stop()  # Cancel and stop before replacing; never stack plans.
        self.last_submit=time.monotonic();self.target=p;self.commanded=True
        destination=PoseInFrame(reference_frame=self.world,pose=pose)
        self.task=asyncio.create_task(asyncio.wait_for(self.motion.move(
            component_name=self.arm_name,destination=destination,timeout=5),5))
        return True
    async def check(self):
        await self.settle()


def can_grab(point,flange,config):
    p,r=flange
    return (np.linalg.norm(point[:2]-p[:2])<=config['capture_xy_mm']
            and abs(p[2]-config['contact_flange_z_mm'])<=5
            and r[2,2]<-.98)


def start_pose(config):
    """Validated taught tracking pose; the follower refuses to start outside bounds."""
    values=config.get('start_pose_mm')
    if not isinstance(values,dict):
        raise ValueError('Set start_pose_mm to the taught tracking pose before using --goto-start')
    missing=[k for k in ('x','y','z','o_x','o_y','o_z','theta') if k not in values]
    if missing:raise ValueError('start_pose_mm is missing '+', '.join(missing))
    position=np.array([values[k] for k in ('x','y','z')],float)
    direction=np.array([values[k] for k in ('o_x','o_y','o_z')],float)
    theta=float(values['theta'])
    if not np.isfinite(position).all() or not np.isfinite(direction).all() or not np.isfinite(theta):
        raise ValueError('start_pose_mm must contain finite values')
    if np.linalg.norm(direction)<1e-9:
        raise ValueError('start_pose_mm needs a non-zero orientation vector')
    bounds=np.array(config['workspace_mm'],float)
    if not ((position>=bounds[0]).all() and (position<=bounds[1]).all()):
        raise ValueError('start_pose_mm lies outside workspace_mm; the follower would reject it')
    return matrix_pose(position,pose_matrix(Pose(x=position[0],y=position[1],z=position[2],
        o_x=direction[0],o_y=direction[1],o_z=direction[2],theta=theta))[1])


async def move_to_start(arm,motion,config,pose,timeout=20):
    """Plan to the taught tracking pose and verify arrival. Stops the arm on failure.

    Reachability and a clear path are the Viam planner's decision, not this check.
    """
    destination=PoseInFrame(reference_frame=config['world_frame'],pose=pose)
    try:
        success=await asyncio.wait_for(motion.move(component_name=config['arm'],
            destination=destination,timeout=timeout),timeout)
        if not success:raise RuntimeError('Motion planner did not complete the move to the tracking pose')
        actual=await asyncio.wait_for(motion.get_pose(config['arm'],config['world_frame'],
            timeout=config['rpc_timeout_s']),config['rpc_timeout_s'])
        if actual.reference_frame!=config['world_frame']:raise ValueError('Unexpected flange frame')
        position,rotation=pose_matrix(actual.pose)
        goal_position,goal_rotation=pose_matrix(pose)
        offset=float(np.linalg.norm(position-goal_position))
        angle=float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@goal_rotation)-1)/2,-1,1))))
        if offset>config['start_tolerance_mm'] or angle>config['start_tolerance_deg']:
            raise ValueError(f'Arm stopped {offset:.1f} mm and {angle:.1f} deg from the tracking pose')
        return offset,angle
    except BaseException:
        try:await asyncio.shield(asyncio.wait_for(arm.stop(timeout=3),3))
        except Exception as error:print(f'STOP FAILED (arm): {error}',flush=True)
        raise
