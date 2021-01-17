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
import heapq

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace

VISUALIZE = False
LOGGING = False
GRAVITY = -9.81
if VISUALIZE:
    p.connect(p.GUI)  # or p.DIRECT for nongraphical version
else:
    p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, GRAVITY)
planeId = p.loadURDF(Environment.plane_urdf_filepath)
kukaId = p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0])
if LOGGING and VISUALIZE:
    log_id = p.startStateLogging(
        p.STATE_LOGGING_VIDEO_MP4, "fully_functional.mp4")
offset = -np.array([0.05, 0, 0])
bottle_start_pos = np.array([0.5, 0.4, 0.1]).astype(float)
EE_start_pos = bottle_start_pos + offset
base_start_ori = np.array([0, 0, 0, 1]).astype(float)
arm = Arm(kuka_id=kukaId, ee_start_pos=EE_start_pos, start_ori=base_start_ori)
