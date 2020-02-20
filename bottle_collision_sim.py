import pybullet as p
import time
import pybullet_data

GRAVITY = -9.81
BASE_ID = -1
N = 10000  # total simulation iterations
test_N = 200  # iters for each test of a parameter

# pybullet_data built-in models
plane_urdf_filepath = "plane.urdf"
arm_filepath = "xarm/xarm6_robot.urdf"
table_filepath = "table/table.urdf"

# p.setTimeOut(max_timeout_sec)

# water bottle 
class Bottle:
    def __init__(self, table_height):
        self.radius = 0.05
        self.height = 0.3
        self.mass = 1
        self.start_pos = [0.5, 0, table_height+.1]
        self.start_ori = [0, 0, 0, 1]

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
    p.resetBasePositionAndOrientation(objectUniqueId=arm_id, 
        posObj=pos,
        ornObj=ori)


def rotate_arm():
    return


def change_arm_params():
    return


def run_sim():
    for _ in range(test_N):
        p.stepSimulation()
        time.sleep(1./240.)


def main():
    physics_client = p.connect(p.GUI)  # or p.DIRECT for nongraphical version

    # allows you to use pybullet_data package's existing URDF models w/out actually having them
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,GRAVITY)
    planeId = p.loadURDF(plane_urdf_filepath)

    # table model
    table_start_pos = [0, 0, 0]
    table_id = p.loadURDF(table_filepath, table_start_pos, useFixedBase=True)
    min_table_bounds, max_table_bounds = p.getAABB(table_id)
    table_height = max_table_bounds[2]

    # robot arm
    arm_start_pos = [0, 0, table_height]
    arm_start_ori = p.getQuaternionFromEuler([0, 0, 0])
    arm_id = p.loadURDF(arm_filepath, arm_start_pos, arm_start_ori)
    bottle = Bottle(table_height)
    bottle_col_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=bottle.radius, 
        height=bottle.height)

    mass_tests = []
    lat_friction_tests = []
    roll_friction_tests = []
    spinning_friction_tests = []
    bounciness_tests = [] 
    contact_stiffness_tests = []
    contact_damping_tests = []
    inertia_tests = []  # changes center of mass
    
    all_tests = []  # all test combos

    running = True

    for test in all_tests:
        bottle_id = p.createMultiBody(
            baseMass=bottle.mass, 
            baseInertialFramePosition=test.bottle_inertia,
            baseCollisionShapeIndex=bottle_col_id,
            basePosition=bottle.start_pos)
        p.changeDynamics(
            bodyUniqueId=bottle_id, 
            linkIndex=BASE_ID, 
            mass=test.bottle_mass,
            lateralFriction=test.lat_fric,
            spinningFriction=test.spin_fric,
            rollingFriction=test.roll_fric,
            restitution=test.bounce,
            contactStiffness=test.contact_stiffness,
            contactDamping=test.contact_dampness)

        run_sim()

        # check if bottle fell over
        cube_pos, cube_ori = p.getBasePositionAndOrientation(bottle_id)

        # remove bottle to not interfere with next test
        p.removeBody(bottle_id)
        
        # reset other components of simulation
        reset_arm(arm_id=arm_id, pos=arm_start_pos, ori=arm_start_ori)

    
    p.disconnect()


if __name__ == '__main__':
    main()