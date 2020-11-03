import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sim_objects import Bottle, Arm
from environment import Environment, Action


def main():
    # initialize simulator environment
    VISUALIZE = True
    LOGGING = False
    GRAVITY = -9.81
    RUN_FULL_MDP = False
    if VISUALIZE: p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    else: p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,GRAVITY)
    planeId = p.loadURDF(Environment.plane_urdf_filepath)
    kukaId = p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0])
    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "fully_functional.mp4")

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
    cost_based = False
    env = Environment(arm, bottle, is_viz=VISUALIZE, N=N, 
        run_full_mdp=RUN_FULL_MDP, cost_based=cost_based)

    dx = dy = 0.1
    xmin, xmax = 0, 0.8
    ymin, ymax = 0, 0.8
    X = np.arange(
        start=xmin, 
        stop=xmax+dx, 
        step=dx)
    Y = np.arange(
        start=ymin, 
        stop=ymax+dy, 
        step=dy)

    # explore the vertical reach of the arm
    hstart = 0
    hend = env.bottle.height
    env.arm.set_general_max_reach([hstart, hend])

    valid_states = []
    for x in X:
        for y in Y:
            # dist_from_base = np.linalg.norm(
            #     np.array([x,y]) - env.arm.base_pos[:2])
            # too_close = (env.init_reach_p * dist_from_base < 
            #     env.arm.min_dist)
            # too_far = dist_from_base > env.arm.MAX_REACH
            too_close = False
            too_far = False
            if not (too_close or too_far):
                valid_states.append((x,y))

    for (x,y) in valid_states:
        env.change_bottle_pos(
            new_pos=[x, y, 0.1])
        print("X, Y: %.2f, %.2f" % (x,y))
        num_iters = 100
        start_pos = np.array([x,y,hstart])
        end_pos = np.array([x,y,hend])
        traj = np.linspace(start_pos, end_pos, num=num_iters)
        angle = math.atan2(y,x)
        _, _, error = env.simulate_plan(traj, angle)
        print(error)

if __name__ == "__main__":
    main()