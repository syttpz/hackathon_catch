"""Interpolate locally sampled Viam camera poses near image timestamps.

Feedback has no exposure timestamp. RPC midpoint sampling is an approximation;
long requests, missing brackets, and wide sample gaps are rejected.
"""
import asyncio
import math
import time
from collections import deque
import numpy as np
from viam.proto.common import Pose, PoseInFrame


def pose_matrix(pose):
    direction=np.array([pose.o_x,pose.o_y,pose.o_z],float)
    values=np.array([pose.x,pose.y,pose.z,pose.theta,*direction])
    if not np.isfinite(values).all() or np.linalg.norm(direction)<1e-9:
        raise ValueError('Invalid orientation/position feedback')
    direction/=np.linalg.norm(direction)
    lat=math.acos(np.clip(direction[2],-1,1))
    lon=math.atan2(direction[1],direction[0]) if 1-abs(direction[2])>1e-9 else 0
    theta=math.radians(pose.theta)
    def rz(a):return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
    ry=np.array([[math.cos(lat),0,math.sin(lat)],[0,1,0],[-math.sin(lat),0,math.cos(lat)]])
    # Viam orientation vectors use Z-Y-Z, not axis-angle.
    return np.array([pose.x,pose.y,pose.z],float),rz(lon)@ry@rz(theta)


def matrix_pose(position,rotation):
    direction=rotation[:,2]
    if 1-abs(direction[2])>1e-9:
        theta=math.atan2(rotation[2,1],-rotation[2,0])
    elif direction[2]>0:
        theta=math.atan2(rotation[1,0],rotation[0,0])
    else:
        theta=math.atan2(rotation[0,1],rotation[1,1])
    return Pose(x=float(position[0]),y=float(position[1]),z=float(position[2]),
        o_x=float(direction[0]),o_y=float(direction[1]),o_z=float(direction[2]),theta=math.degrees(theta))


def mix_rotation(a,b,fraction):
    u,_,vh=np.linalg.svd((1-fraction)*a+fraction*b)
    return u@np.diag([1,1,np.linalg.det(u@vh)])@vh


class PoseBuffer:
    def __init__(self):self.samples=deque(maxlen=200)
    def add(self,start,end,pose):
        if not 0<=end-start<=.04: return
        position,rotation=pose_matrix(pose)
        self.samples.append(((start+end)/2,position,rotation))
    def at(self,timestamp):
        for a,b in zip(self.samples,list(self.samples)[1:]):
            if a[0]<=timestamp<=b[0] and 0<b[0]-a[0]<=.08:
                f=(timestamp-a[0])/(b[0]-a[0])
                return a[1]+f*(b[1]-a[1]),mix_rotation(a[2],b[2],f)
        raise ValueError('No fresh camera-pose bracket for this frame')


async def sample_camera(robot,camera,world,buffer):
    source=PoseInFrame(reference_frame=camera,pose=Pose(o_z=1))
    while True:
        started=time.time()
        transformed=await asyncio.wait_for(robot.transform_pose(source,world),.3)
        finished=time.time()
        buffer.add(started,finished,transformed.pose)
        await asyncio.sleep(.008)
