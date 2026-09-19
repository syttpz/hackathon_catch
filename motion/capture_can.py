"""Follow a rolling can with fresh wrist-camera poses; --execute moves hardware.

This is a feedback-based experimental follower, not a 60 Hz joint servo. Pose
samples are RPC-midpoint estimates, not hardware-synchronized exposure poses.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient
from viam.robot.client import RobotClient
from viam.proto.common import Pose,PoseInFrame
from motion.trajectory_local import credentials,decode_color,positive
from motion.rolling_preview import table_plane,locate_object,camera_transform
from motion.rolling_intercept import RollingFit
from motion.rolling_catch import prepare_gripper
from motion.live_camera_pose import PoseBuffer,sample_camera,pose_matrix
from motion.can_follower import LatestMove,next_target,can_grab,start_pose,move_to_start


async def read_flange(motion,config):
    p=await asyncio.wait_for(motion.get_pose(config['arm'],config['world_frame'],timeout=.3),.3)
    if p.reference_frame!=config['world_frame']:raise ValueError('Unexpected flange frame')
    return pose_matrix(p.pose)


async def run(args):
    config=json.loads(args.config.read_text())
    print('TRACKER can-follow-v6: continuous target updates; no catch line or arrival-time gate.',flush=True)
    kid,key=credentials(args.machine_config)
    options=RobotClient.Options.with_api_key(api_key=key,api_key_id=kid)
    options.dial_options.disable_webrtc=True
    async with await RobotClient.at_address('127.0.0.1:8080',options) as robot:
        arm=Arm.from_robot(robot,config['arm']);cam=Camera.from_robot(robot,config['camera'])
        motion=MotionClient.from_robot(robot,config['motion'])
        gripper=Gripper.from_robot(robot,config['gripper']) if args.execute else None
        if await arm.is_moving(timeout=3):raise ValueError('Start with arm stationary')
        before=np.array((await arm.get_joint_positions(timeout=3)).values)
        flange=await read_flange(motion,config)
        bounds=np.asarray(config['workspace_mm'])
        if args.goto_start:
            goal=start_pose(config)
            print(f'FLANGE NOW ({flange[0][0]:.1f}, {flange[0][1]:.1f}, {flange[0][2]:.1f}) mm',flush=True)
            print(f'START POSE ({goal.x:.1f}, {goal.y:.1f}, {goal.z:.1f}) mm | orientation '
                  f'({goal.o_x:.3f}, {goal.o_y:.3f}, {goal.o_z:.3f}) theta {goal.theta:.1f} deg',flush=True)
            if not args.execute:
                print('PREVIEW: no motion commanded. Add --execute to move the arm there.',flush=True)
                return False
            offset,angle=await move_to_start(arm,motion,config,goal)
            print(f'AT START: within {offset:.1f} mm and {angle:.1f} deg of the taught tracking pose.',flush=True)
            print('Gripper untouched. Run ./capture-can.sh (optionally --execute) to follow.',flush=True)
            return True
        if not ((flange[0]>=bounds[0]).all() and (flange[0]<=bounds[1]).all()):
            raise ValueError('Starting pose is outside the configured follow workspace; '
                             'run with --goto-start --execute to move to the taught tracking pose')
        cam_pose=await asyncio.wait_for(robot.transform_pose(PoseInFrame(reference_frame=config['camera'],pose=Pose(o_z=1)),config['world_frame']),3)
        origin,rotation=pose_matrix(cam_pose.pose)
        # Verify the orientation conversion against four independent transformed
        # points while stationary, before it is used on moving-camera feedback.
        check_origin,check_rotation=await camera_transform(robot,config['camera'],config['world_frame'])
        if not np.allclose(origin,check_origin,atol=1) or not np.allclose(rotation,check_rotation,atol=.005):
            raise ValueError('Camera pose rotation conversion did not match frame-system calibration')
        mount=(flange[1].T@(origin-flange[0]),flange[1].T@rotation)
        cloud,_=await cam.get_point_cloud(timeout=5)
        plane=table_plane(cloud,origin,rotation)
        properties=await cam.get_properties(timeout=3);p=properties.intrinsic_parameters
        k=SimpleNamespace(fx=p.focal_x_px,fy=p.focal_y_px,cx=p.center_x_px,cy=p.center_y_px,width=p.width_px,height=p.height_px)
        if min(k.fx,k.fy,k.width,k.height)<=0:raise ValueError('Invalid camera intrinsics')
        after=np.array((await arm.get_joint_positions(timeout=3)).values)
        if before.shape!=after.shape or not before.size or not np.isfinite(after).all() or np.max(abs(after-before))>.1:
            raise ValueError('Arm moved during startup calibration')
        if args.execute:await prepare_gripper(gripper,config)
        print('READY: roll the can. '+('ARM WILL FOLLOW.' if args.execute else 'PREVIEW: no arm/gripper commands.'),flush=True)
        print('Fresh pose feedback required; lost tracking stops moves. View limits may prevent final contact.',flush=True)
        buffer=PoseBuffer();sampler=asyncio.create_task(sample_camera(robot,config['camera'],config['world_frame'],buffer))
        mover=LatestMove(arm,motion,config['arm'],config['world_frame'])
        fit=RollingFit();last_timestamp=None;last_seen=None;previous_point=None;captured=False
        started=last_report=time.monotonic();last_print=0;frames=detected=0;consecutive=0;near=0
        try:
            while not args.duration or time.monotonic()-started<args.duration:
                if sampler.done():await sampler
                if args.execute:await mover.check()
                images,meta=await cam.get_images(timeout=.3)
                now=time.monotonic();stamp=meta.captured_at.seconds+meta.captured_at.nanos/1e9
                point=None;status=None;fresh=False
                if not 0<=time.time()-stamp<=.12:
                    status='WAIT: camera frame stale';fit.reset();consecutive=0;near=0
                elif last_timestamp is None or stamp>last_timestamp:
                    fresh=True;last_timestamp=stamp;frames+=1
                    # Wait briefly for the newer half of the interpolation bracket.
                    for _ in range(5):
                        try:
                            origin,rotation=buffer.at(stamp);break
                        except ValueError:
                            await asyncio.sleep(.005)
                    else:
                        origin=None
                    if origin is None:
                        status='WAIT: no fresh camera-pose bracket';fit.reset();consecutive=0;near=0
                    else:
                        bgr=decode_color(next(im for im in images if im.name=='color'))
                        if bgr.shape[:2]!=(k.height,k.width):raise ValueError('Camera resolution changed')
                        point,reason=locate_object(bgr,k,origin,rotation,plane,config)
                        if point is None:
                            status=reason;fit.reset();consecutive=0;near=0
                        else:
                            detected+=1
                            if previous_point is not None and np.linalg.norm(point-previous_point)>100:
                                consecutive=0;near=0;fit.reset()
                            previous_point=point.copy();consecutive+=1
                            result=fit.add(stamp,point)
                            velocity=result[1] if result is not None else np.zeros(3)
                            flange=await read_flange(motion,config)
                            if time.time()-stamp>.12:
                                point=None;status='WAIT: observation expired during pose lookup'
                            else:
                                last_seen=time.monotonic()
                                target=next_target(point,velocity,flange,mount,k,config)
                                status=f'CAN XY=({point[0]:.0f}, {point[1]:.0f}) mm'
                                if target is None:
                                    status+=' | HOLD: view limit or already positioned'
                                    if args.execute and mover.task is not None:await mover.stop()
                                elif consecutive>=3:
                                    if args.execute:
                                        if await mover.update(target):
                                            status+=f' | ADJUST -> ({target.x:.0f}, {target.y:.0f}, {target.z:.0f}) mm'
                                    else:status+=f' | WOULD ADJUST -> ({target.x:.0f}, {target.y:.0f}, {target.z:.0f}) mm'
                                if can_grab(point,flange,config):near+=1
                                else:near=0
                                if args.execute and near>=3:
                                    await mover.stop()
                                    # Stop acknowledgements and a fresh post-stop
                                    # observation are required before closure.
                                    await asyncio.sleep(.03)
                                    check_images,check_meta=await cam.get_images(timeout=.3)
                                    check_t=check_meta.captured_at.seconds+check_meta.captured_at.nanos/1e9
                                    if check_t<=stamp or not 0<=time.time()-check_t<=.1:
                                        near=0;continue
                                    await asyncio.sleep(.02)
                                    try:co,cr=buffer.at(check_t)
                                    except ValueError:near=0;continue
                                    check_bgr=decode_color(next(im for im in check_images if im.name=='color'))
                                    check_point,_=locate_object(check_bgr,k,co,cr,plane,config)
                                    actual=await read_flange(motion,config)
                                    if check_point is None or time.time()-check_t>.12 or not can_grab(check_point,actual,config):
                                        near=0;continue
                                    print('CAPTURE: can observed at gripper; closing.',flush=True)
                                    grabbed=bool(await asyncio.wait_for(gripper.grab(timeout=3),3))
                                    print('GRIPPER REPORTS OBJECT' if grabbed else 'MISS: no object reported',flush=True)
                                    captured=grabbed
                                    return grabbed
                if last_seen is None or now-last_seen>config['lost_timeout_s']:
                    if args.execute and mover.commanded:
                        await mover.stop();fit.reset();consecutive=0;near=0
                        print('STOP: can tracking lost; waiting to reacquire.',flush=True)
                if status and now-last_print>=.3:print(status,flush=True);last_print=now
                if now-last_report>=2:
                    print(f'Fresh frames {frames/(now-last_report):.1f} Hz | can detections {detected/(now-last_report):.1f} Hz',flush=True)
                    frames=detected=0;last_report=now
                await asyncio.sleep(.002)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler,return_exceptions=True)
            if args.execute:
                try:await asyncio.shield(mover.stop())
                finally:
                    if not captured:await asyncio.shield(asyncio.wait_for(gripper.stop(timeout=3),3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parents[1]/'can_follow.config.json')
    parser.add_argument('--machine-config',type=Path)
    parser.add_argument('--execute',action='store_true',help='Continuously adjust the physical arm and attempt closure near the can')
    parser.add_argument('--goto-start',action='store_true',help='Move the flange to the taught tracking pose and exit; needs --execute to actually move')
    parser.add_argument('--duration',type=positive)
    args=parser.parse_args()
    try:asyncio.run(run(args))
    except KeyboardInterrupt:print('\nFollower stopped.')
    except Exception as error:parser.exit(2,f'Follower stopped: {error}\n')

if __name__=='__main__':main()
