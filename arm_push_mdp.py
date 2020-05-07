import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np

from sim_objects import Bottle, Arm
from environment import Environment

def main():
    # initialize simulator environment
    VISUALIZE = True
    LOGGING = False
    GRAVITY = -9.81
    if VISUALIZE: p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    else: p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,GRAVITY)
    planeId = p.loadURDF(Environment.plane_urdf_filepath)
    kukaId = p.loadURDF(Environment.arm_filepath, [0, 0, 0])
    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "sim_run.mp4")

    # starting end-effector pos, not base pos
    EE_start_pos = np.array([0.2, 0.2, 0.3]).astype(float)
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    arm = Arm(EE_start_pos=EE_start_pos, start_ori=base_start_ori, 
        kukaId=kukaId)

    # bottle
    bottle_start_pos = np.array([0.7, 0.6, 0.1]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)
    
    N = 700
    env = Environment(arm, bottle, is_viz=VISUALIZE, N=N)

    # Action space
    dh = 5
    contact_heights = np.arange(
        start=bottle.height/dh, 
        stop=bottle.height + bottle.height/dh, 
        step=bottle.height/dh)
    main_angle = math.atan2(bottle_start_pos[1], bottle_start_pos[0])
    da = math.pi/80
    angles = np.arange(start=main_angle-3*da, stop=main_angle+4*da, step=da)
    velocities = np.arange(start=0.1, stop=1, step=0.1)

    # run through all possible actions at a given state
    for angle in angles:
        action = (angle, velocities[0], contact_heights[0])
        expected_cost = env.run_sim(action)
        # for h in contact_heights:
        #     for vel in velocities:
        #         action = (angle, vel, h)
                
                # expected_cost = env.run_sim_stochastic(action)

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

if __name__=='__main__':
    main()