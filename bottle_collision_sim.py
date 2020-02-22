import pybullet as p
import time
import pybullet_data
import math

GRAVITY = -9.81
BASE_ID = -1
N = 10000  # total simulation iterations
test_N = 20000  # iters for each test of a parameter
NUM_JOINTS = 7
END_EFFECTOR_ID = 6
LOGGING = True

# pybullet_data built-in models
plane_urdf_filepath = "plane.urdf"
arm_filepath = "kuka_iiwa/model.urdf"
table_filepath = "table/table.urdf"

# p.setTimeOut(max_timeout_sec)

# water bottle 
class Bottle:
    def __init__(self, table_height):
        self.radius = 0.05
        self.height = 0.5
        self.mass = 1
        self.start_pos = [0.5, 0, table_height+.3]
        self.start_ori = [0, 0, 0, 1]

class Arm:
    def __init__(self):
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
        self.rp = [math.pi, math.pi/2, 0, 0, 0, 0, 0]
        #joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.max_force = 20
        self.max_vel = 20

class Test:
    def __init__(self, bottle_mass, lat_fric, roll_fric, spin_fric, bounce, 
        contact_stiffness, contact_dampness, bottle_inertia):
        self.bottle_mass = bottle_mass
        self.lat_fric = lat_fric
        self.roll_fric = roll_fric
        self.spin_fric = spin_fric
        self.bounce = bounce 
        self.contact_stiffness = contact_stiffness
        self.contact_dampness = contact_dampness
        self.bottle_inertia = bottle_inertia

def reset_arm(arm_id, pos, ori):
    # reset arm
    p.resetBasePositionAndOrientation(bodyUniqueId=arm_id, 
        posObj=pos,
        ornObj=ori)


def rotate_arm():
    return


def change_arm_params():
    return


def run_sim(arm):
    t = 0
    p.setJointMotorControl2(bodyUniqueId=arm.arm_id, 
            jointIndex=0, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity = -20,
            force = arm.max_force,
            maxVelocity=arm.max_vel)
    for i in range(test_N):
        p.stepSimulation()
        time.sleep(1./240.)
        t = t + 0.01


# def run_all_tests():
#     physics_client = p.connect(p.GUI)  # or p.DIRECT for nongraphical version

#     # allows you to use pybullet_data package's existing URDF models w/out actually having them
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0,0,GRAVITY)
#     planeId = p.loadURDF(plane_urdf_filepath)

#     # table model
#     table_start_pos = [0, 0, 0]
#     table_id = p.loadURDF(table_filepath, table_start_pos, useFixedBase=True)
#     min_table_bounds, max_table_bounds = p.getAABB(table_id)
#     table_height = max_table_bounds[2]

#     # robot arm
#     arm = Arm()
#     arm_start_pos = [0, 0, table_height]
#     arm_start_ori = p.getQuaternionFromEuler([0, 0, 0])
#     arm.arm_id = p.loadURDF(arm_filepath, arm_start_pos, arm_start_ori)
#     if (p.getNumJoints(arm.arm_id) != NUM_JOINTS):
#         print("Invalid number of joints. Expected: %d, Actual: %d", NUM_JOINTS, num_joints)
#         exit()  # undefined
#     for i in range(numJoints):
#         p.resetJointState(arm.arm_id, i, [0]*NUM_JOINTS)
    
#     # arm sim params
#     #trailDuration is duration (in seconds) after debug lines will be removed automatically
#     trailDuration = 5

#     # jointPoses = p.calculateInverseKinematics(
#     #     arm_id, END_EFFECTOR_ID, pos, orn, ll, ul, jr, rp)

#     # bottle
#     bottle = Bottle(table_height)
#     bottle_col_id = p.createCollisionShape(
#         shapeType=p.GEOM_CYLINDER,
#         radius=bottle.radius, 
#         height=bottle.height)

#     mass_tests = []
#     lat_friction_tests = []
#     roll_friction_tests = []
#     spinning_friction_tests = []
#     bounciness_tests = [] 
#     contact_stiffness_tests = []
#     contact_damping_tests = []
#     inertia_tests = []  # changes center of mass
    
#     all_tests = []  # all test combos

#     running = True

#     for test in all_tests:
#         bottle_id = p.createMultiBody(
#             baseMass=bottle.mass, 
#             baseInertialFramePosition=test.bottle_inertia,
#             baseCollisionShapeIndex=bottle_col_id,
#             basePosition=bottle.start_pos)
#         p.changeDynamics(
#             bodyUniqueId=bottle_id, 
#             linkIndex=BASE_ID, 
#             mass=test.bottle_mass,
#             lateralFriction=test.lat_fric,
#             spinningFriction=test.spin_fric,
#             rollingFriction=test.roll_fric,
#             restitution=test.bounce,
#             contactStiffness=test.contact_stiffness,
#             contactDamping=test.contact_dampness)

#         run_sim(arm)

#         # check if bottle fell over
#         cube_pos, cube_ori = p.getBasePositionAndOrientation(bottle_id)
#         is_fallen = check_is_fallen(cube_pos, cube_ori)
#         if is_fallen: 
#             # TODO: Mark in some table, true
#             print("fell over")

#         # remove bottle to not interfere with next test
#         p.removeBody(bottle_id)
        
#         # reset other components of simulation
#         reset_arm(arm_id=arm_id, pos=arm_start_pos, ori=arm_start_ori)

    
#     p.disconnect()


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
    arm = Arm()
    arm_start_pos = [0, 0, table_height]
    arm_start_ori = p.getQuaternionFromEuler([0, 0, 0])
    arm.arm_id = p.loadURDF(arm_filepath, arm_start_pos, arm_start_ori)
    if (p.getNumJoints(arm.arm_id) != NUM_JOINTS):
        print("Invalid number of joints. Expected: %d, Actual: %d", NUM_JOINTS, num_joints)
        exit()  # undefined
    for i in range(NUM_JOINTS):
        p.resetJointState(arm.arm_id, i, arm.rp[i])
    
    # arm sim params
    #trailDuration is duration (in seconds) after debug lines will be removed automatically
    trailDuration = 5

    # bottle
    bottle = Bottle(table_height)
    bottle_col_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=bottle.radius, 
        height=bottle.height)

    test = Test(
        bottle_mass = 1, 
        lat_fric = 0.1,  # static friction
        roll_fric = 0.05, 
        spin_fric = 0,  # shouldn't be any resistance in spinning in midair
        bounce = 1, 
        contact_stiffness = 1, 
        contact_dampness = 1, 
        bottle_inertia = [0, 0, 0]
    )

    bottle_id = p.createMultiBody(
        baseMass=bottle.mass, 
        baseInertialFramePosition=test.bottle_inertia,
        baseCollisionShapeIndex=bottle_col_id,
        basePosition=bottle.start_pos)
    p.changeDynamics(
        bodyUniqueId=bottle_id, 
        linkIndex=BASE_ID, 
        # mass=test.bottle_mass,
        lateralFriction=test.lat_fric,
        spinningFriction=test.spin_fric,
        rollingFriction=test.roll_fric,
        # restitution=test.bounce,
        # contactStiffness=test.contact_stiffness,
        # contactDamping=test.contact_dampness)
    )

    run_sim(arm)

    p.removeBody(bottle_id)
        
    # reset other components of simulation
    reset_arm(arm_id=arm_id, pos=arm_start_pos, ori=arm_start_ori)

    run_sim()

    p.stopStateLogging(log_id)

    p.disconnect()


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