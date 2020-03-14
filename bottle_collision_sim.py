import pybullet as p
import time
import pybullet_data
import math
import numpy as np
# from scipy.optimize import minimize

GRAVITY = -9.81
BASE_ID = -1
N = 10000  # total simulation iterations
test_N = 1200  # iters for each test of a parameter
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
BASE_LINK_L = 0.130
L1 = 0
L2 = 0  # sum of the rest of arm length
non_base_links = 0

# pybullet_data built-in models
plane_urdf_filepath = "plane.urdf"
arm_filepath = "kuka_iiwa/model.urdf"
table_filepath = "table/table.urdf"

# p.setTimeOut(max_timeout_sec)

# water bottle 
class Bottle:
    def __init__(self, table_height):
        self.radius = BOTTLE_R  # 0.03175  # m
        self.height = BOTTLE_H  # 0.1905   # m
        self.mass = 0.5        # kg
        self.start_pos = [0.5, 0, table_height+.3]
        self.start_ori = [0, 0, 0, 1]
        self.col_id = None

class Arm:
    def __init__(self, start_pos, start_ori):
        # NOTE: taken from example: 
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        #lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        # self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        self.rp = [math.pi, (90 + 15)*math.pi/180, 0, 0, 0, 0, 0]
        #joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.max_force = 20
        self.max_vel = 20
        self.rot_vel = 20

        self.start_pos = start_pos
        self.start_ori = start_ori

class Test:
    def __init__(self, bottle_mass, lat_fric, bounce, 
        contact_stiffness, contact_dampness, bottle_inertia):
        self.bottle_mass = bottle_mass
        self.lat_fric = lat_fric
        self.bounce = bounce 
        self.contact_stiffness = contact_stiffness
        self.contact_dampness = contact_dampness
        self.bottle_inertia = bottle_inertia

def reset_arm(arm_id, pos, ori):
    # reset arm
    p.resetBasePositionAndOrientation(bodyUniqueId=arm_id, 
        posObj=pos,
        ornObj=ori)
    
def run_sim(arm):
    t = 0
    p.setJointMotorControl2(bodyUniqueId=arm.arm_id, 
            jointIndex=0, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity = -arm.rot_vel,
            force = arm.max_force,
            maxVelocity=arm.max_vel)
    for i in range(test_N):
        p.stepSimulation()
        time.sleep(1./240.)
        t = t + 0.01


def calc_joints_from_pos(L1, L2, goal_x, goal_y):
    """
    Geometric solution to 2-DOF robot arm inverse kinematics.
    NOTE: goal_x and goal_y must be definied WITH RESPECT TO BASE OF ARM, so
    provide something like (arm_L, goal_x - base_x, goal_y - base_y)
    :param arm_L: length of robot arm (in our case, both links same length)
    :type: float
    :param goal_x: target x-position of end_effector
    :type: float
    :param goal_y: target y-position
    :type: float
    :returns (theta0, theta1): two joint angles required for goal position
    :type: tuple(float, float)
    """
    # while desired x, y is out of reach of arm
    # check if hypotenuse of triangle formed by x and y is > combined arm length
    theta1 = math.acos((goal_x**2 + goal_y**2 - L1**2 - L2**2) /
                       (2*L1*L2))
    theta0 = math.atan2(goal_y, goal_x) - (
                math.atan2(L2*math.sin(theta1),
                           L1 + L2*math.cos(theta1)))
    # simply invert conversion to get radians to degree
    return (theta0, theta1)


def get_arm_dimensions():
    global L1, L2, non_base_links
    arm_start_pos = np.array([0, 0, 0])
    arm_start_ori = p.getQuaternionFromEuler([0, 0, 0])
    init_arm_id = p.loadURDF(arm_filepath, arm_start_pos, arm_start_ori)
    init_ang = [math.pi, 0, 0, (-90)*math.pi/180, 0, 0, 0]
    for i in range(NUM_JOINTS):
        p.resetJointState(init_arm_id, i, init_ang[i])

    state = p.getLinkState(init_arm_id, END_EFFECTOR_ID)
    EE_pos = state[0]

    L1 = EE_pos[2] - BASE_LINK_L
    L2 = (EE_pos[0]**2 + EE_pos[1]**2)**0.5
    print("Link lengths: ", L1, L2)
    non_base_links = (L1**2 + L2**2)**0.5
    p.removeBody(init_arm_id)

# maximize x distance, or equivalently minimize negative x
# def min_neg_x(theta2, y):
#     global L1, L2
#     return -1*L1*math.cos(math.asin((y - L2*math.sin(theta2)/L1)))


def test_diff_factors():
    physics_client = p.connect(p.GUI)  # or p.DIRECT for nongraphical version

    # allows you to use pybullet_data package's existing URDF models w/out actually having them
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,GRAVITY)
    planeId = p.loadURDF(plane_urdf_filepath)
    log_id = -1
    if LOGGING:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "result.mp4")

    # table model
    table_start_pos = [0, 0, 0]
    table_id = p.loadURDF(table_filepath, table_start_pos, useFixedBase=True)
    min_table_bounds, max_table_bounds = p.getAABB(table_id)
    table_height = max_table_bounds[2]

    # robot arm
    arm = Arm(start_pos=np.array([-0.25, 0, table_height]),
        start_ori=p.getQuaternionFromEuler([0, 0, 0]))

    get_arm_dimensions()
    
    #trailDuration is duration (in seconds) after debug lines will be removed automatically
    trailDuration = 5

    # bottle
    bottle = Bottle(table_height)
    bottle.col_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=bottle.radius, 
        height=bottle.height)

    arm_rot_vels = np.arange(0, 20+1, 2)  # 0, 2, 4, ... 20 m/s
    contact_heights = np.arange(0, bottle.height, bottle.height/5)  # 5 different contact heights evenly spread
    fill_props = np.arange(0, 1+0.1, 0.25)  # maps to mass of bottle
    lat_frics = np.arange(0.25, 0.4, 0.05)
    # restitution = [0.5, 0.75]
    # contact_width

    # distance moved
    test_distance_moved(bottle, arm)

    p.disconnect()


def test_distance_moved(bottle, arm):
    lat_frics = np.arange(start=0.25, stop=(0.4+0.05), step=0.05)
    fill_props = np.arange(start=0, stop=(1+0.25), step=0.25)
    arm_rot_vels = np.arange(start=0, stop=(20+2), step=2)  # 0, 2, 4, ... 20 m/s
    contact_heights = np.arange(0, bottle.height + bottle.height/5, bottle.height/5)
    full_bottle_mass = PLASTIC_MASS + MAX_VOLUME * VOL_TO_MASS

    # for each fill proportion, test lateral friction and arm velocity separately
    for fill_i in range(len(fill_props)):
        fill_p = fill_props[fill_i]
        bottle_mass = PLASTIC_MASS + (FULL_WATER_MASS * fill_p)

        for fric_i in range(len(lat_frics)):
            lat_f = lat_frics[fric_i]

            bottle_id = p.createMultiBody(
                baseMass=bottle_mass, 
                baseCollisionShapeIndex=bottle.col_id,
                basePosition=bottle.start_pos)
            p.changeDynamics(
                bodyUniqueId=bottle_id, 
                linkIndex=BASE_ID, 
                lateralFriction=lat_f
            )

            arm.arm_id = p.loadURDF(arm_filepath, arm.start_pos, arm.start_ori)
            for joint_i in range(NUM_JOINTS):
                p.resetJointState(arm.arm_id, joint_i, arm.rp[joint_i])

            run_sim(arm)

            p.removeBody(bottle_id)
            p.removeBody(arm.arm_id)
                
            # reset other components of simulation
            # reset_arm(arm_id=arm.arm_id, pos=arm_start_pos, ori=arm_start_ori)
            if LOGGING:
                p.stopStateLogging(log_id)


def get_target_joint_pos(contact_heights):
    joint_poses = []
    for contact_height in contact_heights:
        target_z = contact_height - BASE_LINK_L # subtracted base link length
        theta1, theta2 = None, None
        possible_y = np.arange(start=(L1 + L2), stop=0, step=-1*(L1+L2)/20)
        for target_y in possible_y:
            try:
                theta1, theta2 = calc_joints_from_pos(L1, L2, goal_x=target_y, goal_y=target_z)
                break
            except Exception:
                continue
        new_pos = [math.pi, math.pi/2 - theta1, 0, theta2, 0, 0, 0]
        joint_poses.append(new_pos)

    return joint_poses


def main():
    # run_all_tests()
    test_diff_factors()

if __name__ == '__main__':
    main()



    # cpu parallell pybullet
    # ODE physics engine
    # Vrep
    # mujoco- fast, not accurate
    # parallel Flex NVidia