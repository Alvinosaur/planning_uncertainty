import pybullet as p
import time
import pybullet_data
import math
import numpy as np
# from scipy.optimize import minimize

import helpers
from sim_objects import Bottle, Arm


VISUALIZE = False
GRAVITY = -9.81
BASE_ID = 0
SIM_RUNTIME = 2000  # iters for each test of a parameter
SIM_VIZ_FREQ = 1./240.
NUM_JOINTS = 7
END_EFFECTOR_ID = 6
LOGGING = False
MAX_VOLUME = 16.9      # fl-oz
WATER_DENSITY = 997    # kg/m³
BOTTLE_H = 0.1905      # m
BOTTLE_R = 0.03175     # m 
VOL_TO_MASS = 0.0296   # fl-oz to kg
FULL_WATER_MASS = MAX_VOLUME * VOL_TO_MASS
PLASTIC_MASS = 0.0127  # kg
# Source: https://github.com/bulletphysics/bullet3/blob/master/data/kuka_lwr/kuka.urdf
BASE_LINK_L = 0.35
FINAL_ARM_POS = (10 * math.pi / 180)
M_TO_CM = 100
MAX_DIST = 44  # for graphing distribution
L1 = 0
L2 = 0  # sum of the rest of arm length
non_base_links = 0
table_height = 0
log_id = -1

# pybullet_data built-in models
plane_urdf_filepath = "plane.urdf"
arm_filepath = "kuka_iiwa/model.urdf"
table_filepath = "table/table.urdf"
gripper_path = "kuka_iiwa/kuka_with_gripper.sdf"

# p.setTimeOut(max_timeout_sec)

def reset_arm(arm_id, pos, ori):
    # reset arm
    p.resetBasePositionAndOrientation(bodyUniqueId=arm_id, 
        posObj=pos,
        ornObj=ori)
    
def run_sim(bottle, arm, duration=SIM_RUNTIME):
    t = 0
    p.setJointMotorControl2(bodyUniqueId=arm.arm_id, 
            jointIndex=BASE_ID, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-arm.rot_vel,
            force = arm.max_force)
    move_arm = True
    prev_pos = None
    for i in range(duration):
        p.stepSimulation()
        base_joint_pos = p.getJointState(bodyUniqueId=arm.arm_id, jointIndex=BASE_ID)

        # stop arm at final position
        if move_arm and (base_joint_pos[0] <= FINAL_ARM_POS):
            move_arm = False
            p.setJointMotorControl2(bodyUniqueId=arm.arm_id, 
                jointIndex=BASE_ID, 
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force = arm.max_force)

        # stop simulation if bottle and arm stopped moving
        bottle_pos, bottle_ori = p.getBasePositionAndOrientation(bottle.bottle_id)
        if not move_arm and prev_pos is not None:
            is_fallen = helpers.check_is_fallen(bottle_ori)
            bottle_vert_stopped = math.isclose(bottle_pos[2] - prev_pos[2], 0.0, abs_tol=1e-05)
            # print(bottle_vert_stopped)
            bottle_horiz_stopped = math.isclose(helpers.euc_dist_horiz(bottle_pos, prev_pos), 0.0, abs_tol=1e-05)
            # if is_fallen:
            if bottle_horiz_stopped or is_fallen:
                break
        prev_pos = bottle_pos
        if VISUALIZE: time.sleep(SIM_VIZ_FREQ)

    
def com_from_fill(bottle, fill_p):
    # calculate center of mass of water bottle
    water_height = bottle.height * fill_p
    if water_height == 0: 
        # if bottle empty, com is just center of cylinder
        return [0, 0, bottle.height / 2.]
    else:
        return [0, 0, water_height / 2.]


def get_arm_dimensions():
    global L1, L2, non_base_links
    arm_start_pos = np.array([0, 0, 0])
    arm_start_ori = p.getQuaternionFromEuler([0, 0, 0])
    init_arm_id = p.loadURDF(arm_filepath, arm_start_pos, arm_start_ori)
    init_ang = [math.pi, 0, 0, (-90)*math.pi/180, 0, 0, 0]
    for i in range(NUM_JOINTS):
        p.resetJointState(init_arm_id, i, init_ang[i])

    state = p.getLinkState(init_arm_id, END_EFFECTOR_ID)
    EE_pos = state[0]  # world frame

    L1 = EE_pos[2] - BASE_LINK_L
    L2 = (EE_pos[0]**2 + EE_pos[1]**2)**0.5

    non_base_links = (L1**2 + L2**2)**0.5
    p.removeBody(init_arm_id)


def simulate_action(bottle, arm, action):
    lat_fric = 0.25
    fill_prop = 0.5
    bottle_mass = PLASTIC_MASS + (fill_prop * MAX_VOLUME * VOL_TO_MASS)
    center_of_mass = com_from_fill(bottle, fill_prop)

    # have arm hit center of bottle
    joint_pos = action

    bottle.bottle_id = p.createMultiBody(
        baseMass=bottle_mass,
        baseInertialFramePosition=center_of_mass,
        baseCollisionShapeIndex=bottle.col_id,
        basePosition=bottle.start_pos)
    p.changeDynamics(
        bodyUniqueId=bottle.bottle_id, 
        linkIndex=-1,  # no links, -1 refers to bottle base
        lateralFriction=lat_fric
    )

    arm.arm_id = p.loadURDF(arm_filepath, arm.start_pos, arm.start_ori)
    for joint_i in range(NUM_JOINTS):
        p.resetJointState(arm.arm_id, joint_i, joint_pos[joint_i])

    run_sim(bottle, arm)

    bottle_pos, bottle_ori = p.getBasePositionAndOrientation(bottle.bottle_id)
    is_fallen = helpers.check_is_fallen(bottle_ori)

    p.removeBody(bottle.bottle_id)
    p.removeBody(arm.arm_id)
    
    return bottle_pos, is_fallen


def get_fall_prob(bottle, arm, action):
    # lat_frics = np.arange(start=0.1, stop=(0.4+0.01), step=0.1)
    # fill_props = np.arange(start=0, stop=(1+0.1), step=0.25)
    lat_frics = np.arange(start=0.1, stop=(0.4+0.01), step=0.3)
    fill_props = np.arange(start=0, stop=(1+0.1), step=0.5)
    bottle_masses = PLASTIC_MASS + (fill_props * MAX_VOLUME * VOL_TO_MASS)

    # have arm hit center of bottle
    joint_pos = action

    fall_count = 0

    # for each fill proportion, test lateral friction and arm velocity separately
    sim_iter = 0
    total_iters = float(len(bottle_masses) * len(lat_frics))
    for fill_pi, bottle_mass in enumerate(bottle_masses):
        center_of_mass = com_from_fill(bottle, fill_props[fill_pi])
        for lat_fric in lat_frics:
            sim_iter += 1
            bottle.bottle_id = p.createMultiBody(
                baseMass=bottle_mass,
                baseInertialFramePosition=center_of_mass,
                baseCollisionShapeIndex=bottle.col_id,
                basePosition=bottle.start_pos)
            p.changeDynamics(
                bodyUniqueId=bottle.bottle_id, 
                linkIndex=-1,  # no links, -1 refers to bottle base
                lateralFriction=lat_fric
            )

            arm.arm_id = p.loadURDF(arm_filepath, arm.start_pos, arm.start_ori)
            for joint_i in range(NUM_JOINTS):
                p.resetJointState(arm.arm_id, joint_i, joint_pos[joint_i])

            run_sim(bottle, arm)

            bottle_pos, bottle_ori = p.getBasePositionAndOrientation(bottle.bottle_id)
            is_fallen = helpers.check_is_fallen(bottle_ori)
            fall_count += int(is_fallen)

            p.removeBody(bottle.bottle_id)
            p.removeBody(arm.arm_id)
            print("Percent Complete: %d" % int(sim_iter * 100. / total_iters))

    return fall_count / float(total_iters)

def heuristic(target, cur):
    # euclidean distance from target
    return np.linalg.norm(target - cur)

def update(V, bottle):
    global A, target, FALL_COST, gamma, contact_heights
    lowest_cost = 0  # initial value doesn't matter
    best_action = None
    for ai, action in enumerate(A):
        prob_fall = get_fall_prob(bottle, arm, action)
        print("Prob of falling: %.3f" % prob_fall)
        new_pos, is_fallen = simulate_action(bottle, arm, action)

        trans_cost = heuristic(target, new_pos) + prob_fall*FALL_COST
        cost = gamma*V + trans_cost
        print("Cost of %.3f from height %.3f" % (cost, contact_heights[ai]))

        if cost < lowest_cost or best_action is None:
            lowest_cost = cost
            best_action = (action, ai)

    assert(best_action is not None)
    return lowest_cost, best_action


# MAIN:
if VISUALIZE: p.connect(p.GUI)  # or p.DIRECT for nongraphical version
else: p.connect(p.DIRECT)

# allows you to use pybullet_data package's existing URDF models w/out actually having them
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,GRAVITY)
planeId = p.loadURDF(plane_urdf_filepath)

# robot arm
arm = Arm(start_pos=np.array([-0.25, 0, 0]),
    start_ori=p.getQuaternionFromEuler([0, 0, 0]))
get_arm_dimensions()

# bottle
orig_bottle = Bottle(bottle_r=BOTTLE_R, bottle_h=BOTTLE_H)
orig_bottle.col_id = p.createCollisionShape(
    shapeType=p.GEOM_CYLINDER,
    radius=orig_bottle.radius, 
    height=orig_bottle.height)

# grid of 2 x 2 with dx and dy = 0.1
target = np.array(orig_bottle.start_pos) + np.array([-1, 0, 0])
max_iters = 100

# for only binary state of fallen or not
V = 0

# Action space is contact heights, which corresponds to set joint poses
# dh = 20
dh = 10
contact_heights = np.arange(
    start=0, 
    stop=orig_bottle.height + orig_bottle.height/dh, 
    step=orig_bottle.height/dh)
A = helpers.get_target_joint_pos(arm, contact_heights, L1, L2, BASE_LINK_L)

gamma = 0.9
max_iters = 100
policy = None
is_converged = False
min_change = 0.5
FALL_COST = 100
NON_STATES = set()  # obstacles or goal locations
for iter in range(max_iters):
    newV, (policy, policy_i) = update(V, orig_bottle)
    total_change = abs(newV - V)
    V = newV
    if iter > 2 and total_change < min_change: break

p.disconnect()

print("Final V and optimal contact height: %.3f, %.3f" % (
    V, contact_heights[policy_i]))

# Design Questions:
# if bottle falls, should we even consider its position valid?
# since discretizing state space, what happens if bottle falls outside grid space?
# --> do we just round to nearest grid space location? 
# need to better define action space rather than just contact heights because 
# depending on bottle position, arm currently may not even reach bottle since
# it's set to stop at 90deg
# Remember for final version to change dh for contact height discretization
# also the range of values for lateral friction and bottle fill prop
