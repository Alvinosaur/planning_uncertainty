import pybullet as p
import time
import pybullet_data
import math
import numpy as np
# from scipy.optimize import minimize

import helpers
from sim_objects import Bottle, Arm


VISUALIZE = True
GRAVITY = -9.81
BASE_ID = 0
SIM_RUNTIME = 2000  # iters for each test of a parameter
SIM_VIZ_FREQ = 1./240.
NUM_JOINTS = 7
END_EFFECTOR_ID = 6
LOGGING = False

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
    lat_frics = np.arange(start=0.1, stop=(0.4+0.01), step=0.1)
    fill_props = np.arange(start=0, stop=(1+0.1), step=0.25)
    # lat_frics = np.arange(start=0.1, stop=(0.4+0.01), step=0.3)
    # fill_props = np.arange(start=0, stop=(1+0.1), step=0.5)
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
            # print("Percent Complete: %d" % int(sim_iter * 100. / total_iters))

    return fall_count / float(total_iters)


def state_to_idx(y, x):
    global dx, dy, maxX, maxY
    # ensure x, y in grid bounds
    # offset by max so [-max, max] -> [0, 2*max]
    x = min(max(x, -maxX), maxX) + maxX
    y = min(max(y, -maxY), maxY) + maxY
    xi = int(np.rint(x / dx))
    yi = int(np.rint(y / dy))
    return yi, xi

def get_value(pos, V):
    global gamma, dx, dy, maxX, maxY
    [x, y, z] = pos
    yi, xi = state_to_idx(y, x)
    return V[yi][xi]

def heuristic(target, cur):
    # euclidean distance from target
    return np.linalg.norm(target - cur)

def update(V, bottle):
    global A, target, FALL_COST, gamma
    lowest_cost = 0  # initial value doesn't matter
    best_action = None
    for action in A:
        prob_fall = get_fall_prob(bottle, arm, action)
        print(prob_fall)
        new_pos, is_fallen = simulate_action(bottle, arm, action)

        trans_cost = heuristic(target, new_pos) + prob_fall*FALL_COST
        cost = gamma*get_value(new_pos, V) + trans_cost

        if cost < lowest_cost or best_action is None:
            lowest_cost = cost
            best_action = action

    assert(best_action is not None)
    return lowest_cost, best_action

class Environment(object):
    # pybullet_data built-in models
    plane_urdf_filepath = "plane.urdf"
    arm_filepath = "kuka_iiwa/model.urdf"
    table_filepath = "table/table.urdf"
    gripper_path = "kuka_iiwa/kuka_with_gripper.sdf"
    def __init__(self, arm, bottle, is_viz=True, gravity=-9.81, N=5):
        self.is_viz = is_viz
        self.gravity = gravity
        self.arm = arm
        self.bottle = bottle
        self.N = N  # number of sim steps(240Hz update) per action

    def create_sim(self):
        if self.is_viz: p.connect(p.GUI)  # or p.DIRECT for nongraphical version
        else: p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,self.gravity)
        planeId = p.loadURDF(Environment.plane_urdf_filepath)

    def restart_sim(self, is_viz=None):
        # restart simulator with option of changing visualization mode
        try:
            p.disconnect()
        except p.error:
            print("No sim to disconnect from, creating new one as normal...")
        if is_viz is not None: self.is_viz = is_viz
        self.create_sim()

    def step(self, target_EE_pos):
        """[summary]

        Arguments:
            action {Action} -- target position, force
        """
        joint_poses = self.arm.get_target_joints(target_EE_pos)
        for i in range(self.arm.num_joints):
            p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_poses[i],
                                    targetVelocity=self.target_velocity,
                                    force=self.arm.force,
                                    positionGain=self.arm.position_gain,
                                    velocityGain=self.arm.velocity_gain)

        # simulate this action for N steps
        for i in range(self.N): 
            p.stepSimulation()
            if self.is_viz: time.sleep(SIM_VIZ_FREQ)

        # stop simulation if bottle and arm stopped moving
        bottle_pos, bottle_ori = p.getBasePositionAndOrientation(bottle.bottle_id)
        is_fallen = helpers.check_is_fallen(bottle_ori)
        # bottle_vert_stopped = math.isclose(bottle_pos[2] - prev_pos[2], 0.0, abs_tol=1e-05)
        # # print(bottle_vert_stopped)
        # bottle_horiz_stopped = math.isclose(helpers.euc_dist_horiz(bottle_pos, prev_pos), 0.0, abs_tol=1e-05)

        return is_fallen
        
# action space
# move in a given direction vector for N time steps, do this for various starting positions of the arm EE pos and botle pos
# state space: bottle (x,y) position, arm EE (x,y,z) position
# expected value over all possible actions: directions and velocities
# Cost: smallalpha*dist from botle (must be very small alpha), don't want arm to always shoot forward towards bottle in shortest path
# + large alpha*bottle dist from target + HUGE_FALL_COST*is_fallen
# do expected cost over this, summing over unknown friction/mass of bottle

def main():
    #arm 
    arm_start_pos = np.array([0, 0, 0]).astype(float)
    arm_start_ori = np.array([0, 0, 0, 1]).astype(float)
    arm = Arm(start_pos=arm_start_pos, start_ori=arm_start_ori)

    # bottle
    bottle_start_pos = np.array([0.8, 0, 0.1]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    orig_bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)
    orig_bottle.col_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=orig_bottle.radius, 
        height=orig_bottle.height)

    # grid of 2 x 2 with dx and dy = 0.1
    dx = dy = 0.05
    maxX = maxY = 0.2
    target = np.array(orig_bottle.start_pos) + np.array([-1, 0, 0])
    max_iters = 100

    V = np.zeros((int(maxY*2/dy)+1, int(maxX*2/dx)+1))
    H, W = V.shape

    # Action space is contact heights, which corresponds to set joint poses
    # dh = 20
    # angle of pushing 
    # and force exerted
    dh = 5
    contact_heights = np.arange(
        start=0, 
        stop=orig_bottle.height + orig_bottle.height/dh, 
        step=orig_bottle.height/dh)
    A = helpers.get_target_joint_pos(arm, contact_heights, L1, L2, BASE_LINK_L)

    gamma = 0.9
    max_iters = 100
    policy = [[None]*W for h in range(H)]
    is_converged = False
    min_change = 0.5
    FALL_COST = 100
    NON_STATES = set()  # obstacles or goal locations
    for iter in range(max_iters):
        total_change = 0
        # in normal value iteration, don't update on the fly
        oldV = np.copy(V)

        # loop through all states to update their V-value
        for y in np.arange(start=-maxY, stop=maxY+dy, step=dy):
            for x in np.arange(start=-maxX, stop=maxX+dx, step=dx):
                state = np.array([y,x,0]) + np.array(orig_bottle.start_pos)
                # don't update values for obstacles or set states
                if (y, x) in NON_STATES: continue
                new_bottle = Bottle(bottle_r=BOTTLE_R, bottle_h=BOTTLE_H)
                new_bottle.start_pos = state.tolist()
                new_bottle.col_id = orig_bottle.col_id
                yi, xi = state_to_idx(y, x)
                V[yi][xi], policy[yi][xi] = update(oldV, new_bottle)
                total_change += abs(V[yi][xi] - oldV[yi][xi])
        if iter > 2 and total_change < min_change: break

    p.disconnect()

# Design Questions:
# if bottle falls, should we even consider its position valid?
# since discretizing state space, what happens if bottle falls outside grid space?
# --> do we just round to nearest grid space location? 
# need to better define action space rather than just contact heights because 
# depending on bottle position, arm currently may not even reach bottle since
# it's set to stop at 90deg
# Remember for final version to change dh for contact height discretization
# also the range of values for lateral friction and bottle fill prop
