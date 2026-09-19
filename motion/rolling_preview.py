"""Preview rolling-ball crossings; --execute enables one physical catch attempt.

Estimates a near-horizontal table plane from one point cloud, then projects RGB rays
onto the ball-center plane. This is approximate and must be physically checked.
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
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient

from motion.can_tracking import red_can_candidates, can_center_height
from motion.trajectory_local import credentials, decode_color, positive, red_candidates
from motion.rolling_intercept import RollingFit, line_crossing
from motion.table_plane import fit_table_plane, point_on_offset_plane, InvalidProjection
from motion.rolling_catch import PredictionGate, execute_catch, minimum_lead, validate_catch_config, prepare_gripper, move_to_prediction


async def camera_transform(robot, camera_name, world):
    points=[]
    for xyz in [(0,0,0),(100,0,0),(0,100,0),(0,0,100)]:
        p=await asyncio.wait_for(robot.transform_pose(
            PoseInFrame(reference_frame=camera_name,pose=Pose(x=xyz[0],y=xyz[1],z=xyz[2],o_z=1)),world),5)
        points.append([p.pose.x,p.pose.y,p.pose.z])
    origin=np.array(points[0])
    rotation=(np.array(points[1:])-origin).T/100
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=.001):
        raise ValueError('Camera transform changed during calibration')
    return origin,rotation


def table_plane(cloud, origin, rotation):
    end=cloud.index(b'\n',cloud.index(b'DATA '))+1
    header=cloud[:end].decode()
    fields={s.split()[0]:s.split()[1:] for s in header.splitlines() if s and not s.startswith('#')}
    if fields['DATA']!=['binary'] or fields.get('COUNT')!=['1']*len(fields['FIELDS']):
        raise ValueError('Expected uncompressed scalar binary PCD')
    types={('F','4'):'<f4',('F','8'):'<f8',('U','4'):'<u4',('I','4'):'<i4'}
    dtype=np.dtype([(n,types[(t,s)]) for n,t,s in zip(fields['FIELDS'],fields['TYPE'],fields['SIZE'])])
    data=np.frombuffer(cloud[end:],dtype=dtype)
    xyz=np.column_stack([data[k] for k in ('x','y','z')])[::20]*1000
    xyz=xyz[np.isfinite(xyz).all(axis=1)&(xyz[:,2]>0)]
    world=xyz@rotation.T+origin
    return fit_table_plane(world)


def locate_ball(bgr, intrinsics, origin, rotation, plane, radius_mm):
    """Filter distractors before requiring one candidate; never pick arbitrarily."""
    candidates = red_candidates(bgr)
    if not candidates:
        return None, 'BALL NOT VISIBLE: no clear saturated-red circle'
    valid = []
    for pixel, pixel_radius in candidates:
        try:
            point = point_on_offset_plane(pixel,intrinsics,origin,rotation,plane,radius_mm)
        except InvalidProjection:
            continue
        camera_point = rotation.T @ (point-origin)
        if camera_point[2] <= radius_mm:
            continue
        expected_radius = (intrinsics.fx+intrinsics.fy)*.5*radius_mm/camera_point[2]
        if .55*expected_radius <= pixel_radius <= 1.6*expected_radius:
            valid.append(point)
    if len(valid) == 1:
        return valid[0], None
    if not valid:
        return None, 'BALL REJECTED: red candidates outside table or wrong physical size'
    return None, f'BALL AMBIGUOUS: {len(valid)} plausible red balls'


def locate_object(bgr, intrinsics, origin, rotation, plane, config):
    if config.get('object_type','ball') != 'can':
        return locate_ball(bgr,intrinsics,origin,rotation,plane,config['ball_radius_mm'])
    height=can_center_height(config['can_diameter_mm'],config['can_length_mm'],on_side=config['can_on_side'])
    candidates=red_can_candidates(bgr)
    if not candidates: return None, 'CAN NOT VISIBLE: no clear red label'
    valid=[]
    for candidate in candidates:
        try:
            point=point_on_offset_plane(candidate.center_px,intrinsics,origin,rotation,plane,height)
        except InvalidProjection:
            continue
        camera_point=rotation.T@(point-origin)
        if camera_point[2]<=height: continue
        scale=(intrinsics.fx+intrinsics.fy)*.5/camera_point[2]
        diameter=scale*config['can_diameter_mm']
        length=scale*config['can_length_mm']
        # Visible label excludes silver ends and can be foreshortened. These
        # broad bounds reject gross distractors; they are not pose estimation.
        if (.35*diameter<=candidate.short_side_px<=1.6*diameter
            and .5*diameter<=candidate.long_side_px<=1.6*length):
            valid.append(point)
    if len(valid)==1: return valid[0],None
    return None, ('CAN REJECTED: label outside table or wrong size' if not valid
                  else f'CAN AMBIGUOUS: {len(valid)} plausible labels')


async def run(args):
    print(f'TRACKER can-v5 | {Path(__file__).resolve()}',flush=True)
    config=json.loads(args.config.read_text())
    for arg,key in [('horizon','max_prediction_s'),('move_budget','move_budget_s'),('close_lead','close_lead_s')]:
        value=getattr(args,arg,None)
        if value is not None: config[key]=value
    if config.get('object_type')=='can':
        config['ball_radius_mm']=can_center_height(config['can_diameter_mm'],config['can_length_mm'],on_side=config['can_on_side'])
    object_label=config.get('object_type','ball').upper()
    validate_catch_config(config)
    kid,key=credentials(args.machine_config)
    options=RobotClient.Options.with_api_key(api_key=key,api_key_id=kid)
    options.dial_options.disable_webrtc=True
    async with await RobotClient.at_address('127.0.0.1:8080',options) as robot:
        arm=Arm.from_robot(robot,config['arm'])
        cam=Camera.from_robot(robot,config['camera'])
        gripper=Gripper.from_robot(robot,config['gripper']) if args.execute else None
        motion=MotionClient.from_robot(robot,config['motion']) if args.execute else None
        if await arm.is_moving(timeout=3): raise ValueError('Stop the arm before preview')
        baseline=np.array((await arm.get_joint_positions(timeout=3)).values)
        if not len(baseline) or not np.isfinite(baseline).all(): raise ValueError('Invalid joints')
        origin,rotation=await camera_transform(robot,config['camera'],config['world_frame'])
        cloud,_=await cam.get_point_cloud(timeout=5)
        plane=table_plane(cloud,origin,rotation)
        p=(await cam.get_properties(timeout=3)).intrinsic_parameters
        intrinsics=SimpleNamespace(fx=p.focal_x_px,fy=p.focal_y_px,cx=p.center_x_px,cy=p.center_y_px)
        if min(intrinsics.fx,intrinsics.fy)<=0: raise ValueError('Invalid camera intrinsics')
        async def check_arm():
            joints=np.array((await arm.get_joint_positions(timeout=3)).values)
            if joints.shape!=baseline.shape or not np.isfinite(joints).all() or np.max(abs(joints-baseline))>.1 or await arm.is_moving(timeout=3):
                raise ValueError('Arm moved: cached camera transform invalid; restart preview')
        await check_arm()
        if args.execute:
            await prepare_gripper(gripper,config)
            await check_arm()
            print(f'READY: gripper open. Roll the {object_label.lower()} toward the catch line.',flush=True)
        async def watch_arm():
            while True:
                await check_arm()
                await asyncio.sleep(.2)
        monitor=asyncio.create_task(watch_arm())
        mode=('EXECUTE: move to first prediction; no timed grasp.' if args.move_on_prediction else 'EXECUTE: one physical catch attempt.') if args.execute else 'PREVIEW ONLY.'
        print(f'{mode} Table plane: tilt={np.rad2deg(np.arccos(plane.normal[2])):.1f} degrees, '
              f'support={plane.support:.0%}, residual={plane.residual_mm:.1f} mm; '
              f'{object_label} center height above plane={config["ball_radius_mm"]:g} mm.',flush=True)
        print('Approximate world XY from table-plane projection. Keep arm stationary.',flush=True)
        print('Catch segment:',config['catch_line_end_effector_mm'],flush=True)
        fit=RollingFit()
        gate=PredictionGate()
        if args.move_on_prediction:
            print('FIRST TARGET MODE: no stability or arrival-time gate; planner still checks the move.',flush=True)
        else:
            print(f'Catch admission needs {minimum_lead(config, gripper_preopened=True):.2f}s lead; horizon={config["max_prediction_s"]:g}s.',flush=True)
        last_stamp=None
        start=report=time.monotonic()
        last_print=0
        frames=detected=0
        try:
            while not args.duration or time.monotonic()-start<args.duration:
                if monitor.done(): await monitor
                images,meta=await cam.get_images(timeout=3)
                if monitor.done(): await monitor
                timestamp=meta.captured_at.seconds+meta.captured_at.nanos/1e9
                now=time.monotonic()
                status=None
                ready=None
                valid_prediction=False
                if not 0<=time.time()-timestamp<=.25:
                    fit.reset();status='STALE CAMERA FRAME'
                elif last_stamp is None or timestamp>last_stamp:
                    last_stamp=timestamp;frames+=1
                    color=next(im for im in images if im.name=='color')
                    bgr=decode_color(color)
                    if bgr.shape[:2]!=(p.height_px,p.width_px): raise ValueError('Intrinsics/image resolution mismatch')
                    point, reason = locate_object(bgr,intrinsics,origin,rotation,plane,config)
                    if point is None:
                        fit.reset();status=reason
                    else:
                        detected+=1
                        result=fit.add(timestamp,point)
                        status=f'{object_label} XY=({point[0]:.1f}, {point[1]:.1f}) mm | collecting track'
                        if result is not None:
                            position,velocity,residual=result
                            speed=np.linalg.norm(velocity[:2])
                            status=f'{object_label} XY=({position[0]:.1f}, {position[1]:.1f}) mm | speed={speed:.0f} mm/s'
                            if speed<30: status+=' | AT REST / TOO SLOW'
                            else:
                                hit=line_crossing(position,velocity,config['catch_line_end_effector_mm'],config['max_prediction_s'])
                                if hit is None: status+=f' | no segment crossing within {config["max_prediction_s"]:g} s'
                                else:
                                    seconds,_,target=hit
                                    remaining=seconds-(time.time()-timestamp)
                                    if remaining<=0: status+=' | crossing already passed'
                                    else:
                                        status+=f' | CROSSING ({target[0]:.1f}, {target[1]:.1f}), flange Z=175 mm in {remaining:.2f} s'
                                        arrival=time.monotonic()+remaining
                                        valid_prediction=True
                                        stable=gate.add(now,target,arrival)
                                        if args.move_on_prediction:
                                            status+=' | MOVE TO FIRST PREDICTED TARGET'
                                            ready=(target,arrival)
                                        elif remaining<minimum_lead(config, gripper_preopened=True): status+=' | TOO LATE TO COMMIT'
                                        elif not stable: status+=' | confirming prediction'
                                        else:
                                            status+=' | READY TO CATCH' if args.execute else ' | WOULD CATCH (--execute to move)'
                                            ready=(target,arrival)
                if status is not None and not valid_prediction:
                    gate.reset()
                if ready is not None and args.execute:
                    await check_arm()
                    if monitor.done(): await monitor
                    monitor.cancel()
                    await asyncio.gather(monitor,return_exceptions=True)
                    # This pose came from get_end_position(): the taught line is
                    # in arm-base axes. Verify they still coincide with world.
                    local=await arm.get_end_position(timeout=3)
                    world=await motion.get_pose(config['arm'],config['world_frame'],timeout=3)
                    lp=np.array([local.x,local.y,local.z,local.o_x,local.o_y,local.o_z,local.theta])
                    wp=np.array([world.pose.x,world.pose.y,world.pose.z,world.pose.o_x,world.pose.o_y,world.pose.o_z,world.pose.theta])
                    if (world.reference_frame!=config['world_frame'] or not np.isfinite(lp).all()
                        or not np.isfinite(wp).all() or np.linalg.norm(lp[:3]-wp[:3])>1
                        or np.linalg.norm(lp[3:6]-wp[3:6])>.01
                        or abs((lp[6]-wp[6]+180)%360-180)>.5):
                        raise ValueError('Arm-base/world frame mismatch; refusing catch')
                    if args.move_on_prediction:
                        return await move_to_prediction(arm,gripper,motion,config,ready[0])
                    caught=await execute_catch(arm,gripper,motion,config,*ready,gripper_preopened=True)
                    return caught
                if status and now-last_print>=.5:
                    print(status,flush=True);last_print=now
                if now-report>=2:
                    print(f'Fresh frames {frames/(now-report):.1f} Hz | object detections {detected/(now-report):.1f} Hz',flush=True)
                    frames=detected=0;report=now
                await asyncio.sleep(.001)
        finally:
            monitor.cancel()
            await asyncio.gather(monitor,return_exceptions=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine-config',type=Path)
    parser.add_argument('--config',type=Path,default=Path(__file__).parents[1]/'rolling_catch.config.json')
    parser.add_argument('--duration',type=positive)
    parser.add_argument('--execute',action='store_true',help='Open gripper, move arm to line and attempt ONE catch')
    parser.add_argument('--horizon',type=positive,help='Maximum predicted crossing time in seconds (default 5)')
    parser.add_argument('--move-budget',type=positive,help='Minimum remaining seconds allocated for planning/movement')
    parser.add_argument('--close-lead',type=positive,help='Seconds before predicted crossing to start gripper closure')
    parser.add_argument('--move-on-prediction',action='store_true',help='With --execute: move to first valid crossing, leave gripper open; no timed catch')
    args=parser.parse_args()
    if args.move_on_prediction and not args.execute:
        parser.error('--move-on-prediction requires --execute')
    try:
        caught=asyncio.run(run(args))
        if args.execute and caught is not True:
            parser.exit(2,'No catch confirmed.\n')
    except KeyboardInterrupt: print('\nPreview stopped.')
    except Exception as error: parser.exit(2,f'Preview failed: {error}\n')

if __name__=='__main__': main()
