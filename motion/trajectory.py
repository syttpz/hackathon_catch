import asyncio

from connection import connect
from viam.components.arm import Arm
from viam.proto.common import Pose
from collections import deque
# camera input 
# [x, y, z, timestamp]
# assume xyz values in the same coordinate frame as well

'''
To detect release we want a velocity threshold s.t. 
once v_z passes the threshold, its a throw
'''
launch_threshold = 0.5 # m/s

# check pose and see if the vertical velocity passes the threshold



class BallPredictor:
    def __init__(self):
        self.samples = deque(maxlen=10) #lastest 10

    def add_observation(self, timestamp, x, y, z):
        if self.samples and timestamp <= self.samples[-1][0]:
            return False

        self.samples.append((timestamp, x, y, z))
        return True

    def estimate_velocity(self):
        # at least two images to calculate velocity
        if len(self.samples) < 2:
            return None
        
        t1, x1, y1, z1 = self.samples[-2]
        t2, x2, y2, z2 = self.samples[-1]

        dt = t2 - t1

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        vz = (z2 - z1) / dt

        return vx, vy, vz

if __name__ == "__main__":
    predictor = BallPredictor()

    # camera input here, fake data for the moment
    predictor.add_observation(0.00, 0.30, 0.10, 0.80)
    predictor.add_observation(0.05, 0.32, 0.10, 0.85)

    velocity = predictor.estimate_velocity()

    if velocity is not None:
        vx, vy, vz = velocity

    print(f"vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f} m/s")

    if vz > launch_threshold:
        print("Possible throw detected")