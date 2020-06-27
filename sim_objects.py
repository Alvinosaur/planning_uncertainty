import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

PI = math.pi
TWO_PI = 2*math.pi

# water bottle


class Bottle:
    WATER_DENSITY = 997    # kg/m³
    VOL_TO_MASS = 0.0296   # fl-oz to kg
    PLASTIC_MASS = 0.0127  # kg
    PLANE_OFFSET = 0.056
    INIT_PLANE_OFFSET = 0.03805010362200368  # found experimentally
    # PLANE_OFFSET = 0

    def __init__(self, start_pos, start_ori, fill_prop=0.5):
        self.start_pos = start_pos
        self.start_ori = start_ori
        self.col_id = None

        self.max_volume = 16.9      # fl-oz
        self.radius = 0.03175    # m
        self.height = 0.1905     # m
        self.mass = 0.5        # kg
        self.default_fric = 0.1  # plastic-wood dynamic friction
        self.lat_fric = self.default_fric
        self.min_fill = 0.3
        self.max_fill = 1.0

        # sets mass and center of mass
        self.bottle_mass = None
        self.inertial_shift = None
        self.center_of_mass = None
        self.default_com = np.array([0, 0, self.height/2])
        self.set_fill_proportion(fill_prop)

        self.col_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height)

    def set_fill_proportion(self, fill_prop):
        self.bottle_mass = self.mass_from_fill(fill_prop)
        self.center_of_mass = self.com_from_fill(fill_prop)
        self.inertial_shift = self.center_of_mass - self.default_com

    def mass_from_fill(self, fill_prop):
        return Bottle.PLASTIC_MASS + (
            fill_prop * self.max_volume * Bottle.VOL_TO_MASS)

    def com_from_fill(self, fill_prop):
        # calculate center of mass of water bottle
        fill_prop = np.clip(fill_prop, self.min_fill, self.max_fill)
        water_height = self.height * fill_prop
        if fill_prop <= self.min_fill:
            # if bottle empty, com is just center of cylinder
            # less than half of height
            return np.array([0, 0, self.height * 0.4])
        else:
            return np.array([0, 0, water_height * 0.4])

    def create_sim_bottle(self, pos=None, ori=None):
        if ori is None:
            ori = [0, 0, 0, 1]
        if pos is not None:
            x, y, z = pos
            pos = [x, y, z + self.PLANE_OFFSET]
            self.bottle_id = p.createMultiBody(
                baseMass=self.bottle_mass,
                baseInertialFramePosition=self.inertial_shift,
                baseCollisionShapeIndex=self.col_id,
                basePosition=pos,
                baseOrientation=ori)
        else:
            x, y, z = self.start_pos
            pos = [x, y, z + self.PLANE_OFFSET]
            self.bottle_id = p.createMultiBody(
                baseMass=self.bottle_mass,
                baseInertialFramePosition=self.inertial_shift,
                baseCollisionShapeIndex=self.col_id,
                basePosition=self.start_pos,
                baseOrientation=ori)
        p.changeDynamics(
            bodyUniqueId=self.bottle_id,
            linkIndex=-1,  # no links, -1 refers to bottle base
            lateralFriction=self.lat_fric
        )

    def check_is_fallen(self):
        bottle_pos, bottle_ori = p.getBasePositionAndOrientation(
            self.bottle_id)
        z_axis = np.array([0, 0, 1])
        rot_mat = R.from_quat(bottle_ori).as_matrix()
        new_z_axis = rot_mat @ z_axis
        angle = math.acos(z_axis @ new_z_axis /
                          (np.linalg.norm(z_axis) * np.linalg.norm(new_z_axis)))
        # when z-axis rotation is 90deg
        return abs(angle) > (45 * math.pi / 180)


class Arm:
    def __init__(self, EE_start_pos, start_ori, kukaId):
        self.EE_start_pos = EE_start_pos
        self.start_ori = start_ori
        self.kukaId = kukaId
        self.base_pos = np.array([0, 0, 0.1])
        self.min_dist = 0.3
        self.MAX_REACH = None  # need to set with set_general_max_reach()

        # NOTE: taken from example:
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        # lower limits for null space
        self.ll = np.array([-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05])
        # self.ll = [-PI, -PI, -PI, -PI, -PI, -PI, -PI]
        # upper limits for null space
        self.ul = np.array([.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05])
        # self.ul = [PI, PI, PI, PI, PI, PI, PI]
        # joint ranges for null space
        self.jr = np.array([5.8, 4, 5.8, 4, 5.8, 4, 6])
        # restposes for null space
        # self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        self.rp = np.array([math.pi/4, (90 + 15)*math.pi/180, 0, 0, 0, 0, 0])
        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.force = 500  # allow instantenous velocity = target velocity
        self.max_vel = 2*math.pi/16  # angular velocity
        self.rot_vel = self.max_vel

        # may possibly change the below params to be part of action space
        self.max_straight_pos = np.array([0.78751945, 0.57154083, 0.49517447])
        # length of base link
        self.L1 = self.max_straight_pos[2] - self.base_pos[2]
        self.LE = 0.081  # dist btwn EE frame and frame before EE
        self.rprime = np.linalg.norm(self.max_straight_pos[:2]) - self.LE
        self.target_velocity = 0
        self.force = 500
        self.position_gain = 0.1
        self.velocity_gain = 0.5

        self.EE_idx = 6
        self.num_joints = p.getNumJoints(self.kukaId)
        self.ikSolver = 0  # id of solver algorithm provided by pybullet

        self.init_joints = self.get_target_joints(EE_start_pos, angle=0)
        self.reset(joint_pose=self.init_joints)

    def update_joint_pose(self):
        joint_states = p.getJointStates(self.kukaId, range(self.num_joints))
        self.joint_pose = np.array([state[0] for state in joint_states])

    def get_link_positions(self):
        link_states = p.getLinkStates(self.kukaId, range(self.num_joints))
        return [state[4] for state in link_states]

    def reset(self, joint_pose):
        self.joint_pose = joint_pose
        for i in range(self.num_joints):
            p.resetJointState(self.kukaId, i, self.joint_pose[i])

    def resetEE(self, target_pos=None, angle=0):
        p.resetBasePositionAndOrientation(
            self.kukaId, self.base_pos, self.start_ori)

        # reset all joints in case current pose cannot be solved for with iK
        # if this isn't done, ik will attempt to solve for target pos with current pos rather than default pos, which might not be possible
        # for i in range(self.num_joints):
        #     p.resetJointState(self.kukaId, i, 0)

        if target_pos is None:
            self.joint_pose = self.init_joints
        else:
            self.joint_pose = self.get_target_joints(target_pos, angle=angle)
            # print(joints)

        self.reset(self.joint_pose)

    def calc_max_horiz_dist(self, contact_height):
        hprime = abs(self.L1 - contact_height)
        dprime = (self.rprime**2 - hprime**2)**0.5
        max_horiz_dist = dprime + self.LE
        return max_horiz_dist

    def set_general_max_reach(self, all_contact_heights):
        closest_h = None
        min_dist = 0  # dist from L1 joint, which has fixed height
        for h in all_contact_heights:
            dist = abs(h - self.L1)
            if dist < min_dist or closest_h is None:
                min_dist = dist
                closest_h = h

        self.MAX_REACH = self.calc_max_horiz_dist(closest_h)

    def get_target_joints(self, target_EE_pos, angle):
        """Given target EE position, runs pybullet's internal inverse
        kinematics to solve for joint pose that minimizes dist error
        of EE position. Ensures that returned joint pose lies within
        joint limits of arm, which prevents undefinend simulation behavior.

        Args:
            target_EE_pos ([type]): [description]
            angle ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert(len(self.ll) == len(self.ul) == len(self.jr) == len(self.rp))
        joint_poses = p.calculateInverseKinematics(
            self.kukaId,
            self.EE_idx,
            target_EE_pos,
            lowerLimits=self.ll.tolist(),
            upperLimits=self.ul.tolist(),
            jointRanges=self.jr.tolist(),
            restPoses=self.rp.tolist())
        # joint_poses = np.clip(joint_poses, self.ll, self.ul)
        orn = p.getQuaternionFromEuler([-math.pi/2, 0, angle-math.pi/2])
        # joint_poses = list(p.calculateInverseKinematics(
        #     self.kukaId,
        #     self.EE_idx,
        #     target_EE_pos,
        #     orn,
        #     lowerLimits=self.ll,
        #     upperLimits=self.ul,
        #     jointRanges=self.jr,
        #     restPoses=self.rp))
        # joint_poses = p.calculateInverseKinematics(
        #     self.kukaId,
        #     self.EE_idx,
        #     target_EE_pos,
        #     orn,
        #     jointDamping=self.jd,
        #     solver=self.ikSolver)
        # maxNumIterations=100,
        # residualThreshold=.01)

        return joint_poses
