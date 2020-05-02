import pybullet as p
import time
import pybullet_data
import math
import numpy as np
# from scipy.optimize import minimize

import helpers
from sim_objects import Bottle, Arm

TEST_ID = 3  # {0: contact height v.s topple frequency, 1: arm speed v.s dist, 2: friction v.s dist}
# 3: bottle shape v.s dist
VISUALIZE = True
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
FINAL_ARM_POS = (5 * math.pi / 180)
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


def test_contact_height_fill_proportion(bottle, arm):
    contact_heights = np.arange(0, bottle.height + bottle.height/20, bottle.height/20)
    joint_poses = helpers.get_target_joint_pos(arm, contact_heights, L1, L2, BASE_LINK_L)
    fill_props = np.arange(start=0, stop=(1+0.1), step=0.1)
    bottle_masses = PLASTIC_MASS + (fill_props * MAX_VOLUME * VOL_TO_MASS)

    # Store results
    horiz_bins = (contact_heights / bottle.height) * 100  # percentage of height
    fall_counts = [0] * len(horiz_bins)

    # for each fill proportion, test lateral friction and arm velocity separately
    sim_iter = 0
    total_iters = float(len(bottle_masses) * len(joint_poses))
    for fill_pi, bottle_mass in enumerate(bottle_masses):
        center_of_mass = com_from_fill(bottle, fill_props[fill_pi])
        for joint_test_i, joint_pos in enumerate(joint_poses):
            sim_iter += 1
            bottle.bottle_id = p.createMultiBody(
                baseMass=bottle_mass,
                baseInertialFramePosition=center_of_mass,
                baseCollisionShapeIndex=bottle.col_id,
                basePosition=bottle.start_pos)
            p.changeDynamics(
                bodyUniqueId=bottle.bottle_id, 
                linkIndex=-1,  # no links, -1 refers to bottle base
                lateralFriction=bottle.default_fric,
                # rollingFriction=0.01,
                # spinningFriction=bottle.default_fric
            )

            arm.arm_id = p.loadURDF(arm_filepath, arm.start_pos, arm.start_ori)
            for joint_i in range(NUM_JOINTS):
                p.resetJointState(arm.arm_id, joint_i, joint_pos[joint_i])

            run_sim(bottle, arm)

            bottle_pos, bottle_ori = p.getBasePositionAndOrientation(bottle.bottle_id)
            is_fallen = helpers.check_is_fallen(bottle_ori)
            fall_counts[joint_test_i] += is_fallen

            p.removeBody(bottle.bottle_id)
            p.removeBody(arm.arm_id)
            print("Percent Complete: %d" % int(sim_iter * 100. / total_iters))

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

    helpers.plot_distrib(horiz_bins=horiz_bins, vert_bins=fall_counts,
        xlabel='Contact Height Percent of Total Height', ylabel='Fall Count',
        title='Number of Times bottle fell v.s contact height and fill proportions')


def test_arm_speed_fill_proportion(bottle, arm):
    target = 22.5  # deg/s
    tolerance = 5  # deg/s
    speed_bounds = np.array([target - tolerance, target + tolerance]) * math.pi / 180.
    arm_rot_vels = np.arange(start=speed_bounds[0], stop=speed_bounds[1], step=math.pi/180.)
    fill_props = np.arange(start=0, stop=(1+0.1), step=0.1)
    bottle_masses = PLASTIC_MASS + (fill_props * MAX_VOLUME * VOL_TO_MASS)

    # have arm hit center of bottle
    default_joint_pos = helpers.get_target_joint_pos(arm, [bottle.height/2], L1, L2, BASE_LINK_L)[0]

    # Store results
    dist_bins = np.arange(start=0, stop=16, step=1)  # cm
    dist_counts = [0] * len(dist_bins)

    # for each fill proportion, test lateral friction and arm velocity separately
    sim_iter = 0
    total_iters = float(len(bottle_masses) * len(arm_rot_vels))
    for fill_pi, bottle_mass in enumerate(bottle_masses):
        center_of_mass = com_from_fill(bottle, fill_props[fill_pi])
        for rot_vel in arm_rot_vels:
            sim_iter += 1
            bottle.bottle_id = p.createMultiBody(
                baseMass=bottle_mass,
                baseInertialFramePosition=center_of_mass,
                baseCollisionShapeIndex=bottle.col_id,
                basePosition=bottle.start_pos)
            p.changeDynamics(
                bodyUniqueId=bottle.bottle_id, 
                linkIndex=-1,  # no links, -1 refers to bottle base
                lateralFriction=bottle.default_fric,
                # rollingFriction=bottle.default_fric,
                # spinningFriction=0.5
            )

            arm.arm_id = p.loadURDF(arm_filepath, arm.start_pos, arm.start_ori)
            for joint_i in range(NUM_JOINTS):
                p.resetJointState(arm.arm_id, joint_i, default_joint_pos[joint_i])

            arm.rot_vel = rot_vel
            run_sim(bottle, arm)

            bottle_pos, bottle_ori = p.getBasePositionAndOrientation(bottle.bottle_id)
            is_fallen = helpers.check_is_fallen(bottle_ori)
            if not is_fallen:
                dist = helpers.euc_dist_horiz(bottle_pos, bottle.start_pos) * M_TO_CM
                nearest_bin =helpers.find_nearest_bin(dist, dist_bins)
                dist_counts[nearest_bin] += 1

            p.removeBody(bottle.bottle_id)
            p.removeBody(arm.arm_id)
            print("Percent Complete: %d" % int(sim_iter * 100. / total_iters))

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

    helpers.plot_distrib(horiz_bins=horiz_bins, vert_bins=fall_counts,
        xlabel='Contact Height Percent of Total Height', ylabel='Fall Count',
        title='Number of Times bottle fell v.s contact height and fill proportions')


def test_friction_fill_proportion(bottle, arm):
    lat_frics = np.arange(start=0.1, stop=(0.4+0.01), step=0.01)
    fill_props = np.arange(start=0, stop=(1+0.1), step=0.1)
    bottle_masses = PLASTIC_MASS + (fill_props * MAX_VOLUME * VOL_TO_MASS)

    # have arm hit center of bottle
    default_joint_pos = helpers.get_target_joint_pos(arm, [bottle.height/2], L1, L2, BASE_LINK_L)[0]

    # Store results
    dist_bins = np.arange(start=0, stop=16, step=1)  # cm
    dist_counts = [0] * len(dist_bins)

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
                p.resetJointState(arm.arm_id, joint_i, default_joint_pos[joint_i])

            run_sim(bottle, arm)

            bottle_pos, bottle_ori = p.getBasePositionAndOrientation(bottle.bottle_id)
            is_fallen = helpers.check_is_fallen(bottle_ori)
            if not is_fallen:
                dist = helpers.euc_dist_horiz(bottle_pos, bottle.start_pos) * M_TO_CM
                nearest_bin = helpers.find_nearest_bin(dist, dist_bins)
                dist_counts[nearest_bin] += 1

            p.removeBody(bottle.bottle_id)
            p.removeBody(arm.arm_id)
            print("Percent Complete: %d" % int(sim_iter * 100. / total_iters))

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

    helpers.plot_distrib(horiz_bins=dist_bins, vert_bins=dist_counts,
        xlabel='Distance moved (cm)', ylabel='Tally of Occurences',
        title='Distance moved v.s friction and fill proportions')


def test_bottle_shape(bottle, arm):
    tolerance_r = BOTTLE_R * .2 # m
    radius_bounds = np.array([BOTTLE_R - tolerance_r, BOTTLE_R + tolerance_r])
    bottle_radiuses = np.arange(start=radius_bounds[0], stop=radius_bounds[1], step=0.001)

    tolerance_h = BOTTLE_H * .2 # m
    height_bounds = np.array([BOTTLE_H - tolerance_h, BOTTLE_H + tolerance_h])
    bottle_heights = np.arange(start=height_bounds[0], stop=height_bounds[1], step=0.01)

    # have arm hit center of bottle
    default_joint_pos = helpers.get_target_joint_pos(arm, [bottle.height/2], L1, L2, BASE_LINK_L)[0]
    center_of_mass = com_from_fill(bottle, 1)  # default to full bottle

    # Store results
    horiz_bins = (contact_heights / bottle.height) * 100  # percentage of height
    fall_counts = [0] * len(horiz_bins)


    # for each fill proportion, test lateral friction and arm velocity separately
    sim_iter = 0
    total_iters = float(len(bottle_heights) * len(bottle_radiuses))
    for bottle_h in bottle_heights:
        for bottle_r in bottle_radiuses:
            sim_iter += 1

            new_col_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=bottle_r, 
                height=bottle_h)

            bottle.bottle_id = p.createMultiBody(
                baseMass=bottle.mass,
                baseInertialFramePosition=center_of_mass,
                baseCollisionShapeIndex=new_col_id,
                basePosition=bottle.start_pos)
            p.changeDynamics(
                bodyUniqueId=bottle.bottle_id, 
                linkIndex=-1,  # no links, -1 refers to bottle base
                lateralFriction=bottle.default_fric,
                # rollingFriction=bottle.default_fric,
                # spinningFriction=0.5
            )

            arm.arm_id = p.loadURDF(arm_filepath, arm.start_pos, arm.start_ori)
            for joint_i in range(NUM_JOINTS):
                p.resetJointState(arm.arm_id, joint_i, default_joint_pos[joint_i])

            run_sim(bottle, arm)

            bottle_pos, bottle_ori = p.getBasePositionAndOrientation(bottle.bottle_id)
            is_fallen = helpers.check_is_fallen(bottle_ori)
            fall_counts[joint_test_i] += is_fallen

            p.removeBody(bottle.bottle_id)
            p.removeBody(arm.arm_id)
            print("Percent Complete: %d" % int(sim_iter * 100. / total_iters))

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

    helpers.plot_distrib(horiz_bins=dist_bins, vert_bins=dist_counts,
        xlabel='Distance moved (cm)', ylabel='Tally of Occurences',
        title='Distance moved v.s arm rotational velocity and fill proportions')


def test_diff_factors():
    global table_height, log_id
    if VISUALIZE: p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    else: p.connect(p.DIRECT)

    # allows you to use pybullet_data package's existing URDF models w/out actually having them
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,GRAVITY)
    planeId = p.loadURDF(plane_urdf_filepath)

    # table model
    # table_start_pos = [0, 0, 0]
    # table_id = p.loadURDF(table_filepath, table_start_pos, useFixedBase=True)
    # min_table_bounds, max_table_bounds = p.getAABB(table_id)
    # table_height = max_table_bounds[2]

    # robot arm
    arm = Arm(start_pos=np.array([-0.25, 0, 0]),
        start_ori=p.getQuaternionFromEuler([0, 0, 0]))
    get_arm_dimensions()

    # bottle
    bottle = Bottle(0, BOTTLE_R, BOTTLE_H)
    bottle.col_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=bottle.radius, 
        height=bottle.height)

    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "result_test_{}.mp4".format(TEST_ID))

    # NOTE: DO NOT RUN MULTIPLE TESTS IN ONE SCRIPT, PLOTS GET MESSED UP
    if TEST_ID == 0:
        # topple frequency, contact height, mass
        test_contact_height_fill_proportion(bottle, arm)
    
    elif TEST_ID == 1:
        # distance moved, arm rotation velocity, mass
        test_arm_speed_fill_proportion(bottle, arm)

    elif TEST_ID == 2:  # TEST_ID == 2
        # distance moved, friction, bottle mass
        test_friction_fill_proportion(bottle, arm)
    elif TEST_ID == 3:
        test_bottle_shape(bottle, arm)

    p.disconnect()