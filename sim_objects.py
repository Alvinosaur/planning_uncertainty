import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

PI = math.pi
TWO_PI = 2 * PI

# water bottle


class Bottle:
    WATER_DENSITY = 997  # kg/m³
    PLASTIC_MASS = 0.0127  # kg
    PLANE_OFFSET = 0.095
    # height and diameter of bottle of water
    # (diameter / 2) * in_to_cm * cm_to_m
    DEFAULT_RAD = (2.8 / 2.0) * 2.54 * 0.01  # m
    DEFAULT_HEIGHT = 10.1 * 2.54 * 0.01  # m
    DEFAULT_FRIC = 0.1  # plastic-wood dynamic friction
    DEFAULT_FILL = 0.5

    def __init__(self, start_pos=(0, 0, 0), start_ori=(0, 0, 0, 1),
                 fill_prop=DEFAULT_FILL,
                 fric=DEFAULT_FRIC,
                 radius=DEFAULT_RAD,
                 height=DEFAULT_HEIGHT):
        self.start_pos = start_pos
        self.start_ori = start_ori
        self.pos = self.start_pos
        self.ori = self.start_ori
        self.col_id = None
        self.bottle_id = None

        self.max_volume = 16.9  # fl-oz
        self.radius = radius  # m
        self.height = height  # m
        self.max_volume = math.pi * self.radius ** 2 * self.height  # m^3
        self.lat_fric = fric
        self.min_fric = 0.05
        self.max_fric = 0.15
        self.min_fill = 0.2
        self.max_fill = 1.0

        # sets mass and center of mass
        self.bottle_mass = None
        self.inertial_shift = None
        self.center_of_mass = None
        self.default_center_of_mass = np.array([0, 0, self.height / 2])
        self.set_fill_proportion(fill_prop)

        self.col_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height)

    def set_fric(self, fric):
        assert self.min_fric <= fric <= self.max_fric, \
            f"{self.min_fric} <= {fric} <= {self.max_fric}"
        self.lat_fric = fric

    def set_fill_proportion(self, fill_prop):
        self.fill_prop = np.clip(fill_prop, self.min_fill, self.max_fill)
        self.bottle_mass = self.mass_from_fill(self.fill_prop)
        self.center_of_mass = self.center_of_mass_from_fill(self.fill_prop)
        self.inertial_shift = self.center_of_mass - self.default_center_of_mass

    def mass_from_fill(self, fill_prop):
        return Bottle.PLASTIC_MASS + (
                fill_prop * self.max_volume * self.WATER_DENSITY)

    def center_of_mass_from_fill(self, fill_prop):
        # calculate center of mass of water bottle
        water_height = self.height * fill_prop
        if fill_prop <= self.min_fill:
            # if bottle empty, center_of_mass is just center of cylinder
            # less than half of height
            return np.array([0, 0, self.height * 0.4])
        else:
            return np.array([0, 0, water_height * 0.4])

    def update_pose(self):
        self.pos, self.ori = p.getBasePositionAndOrientation(
            self.bottle_id)
        self.pos -= self.inertial_shift

    def create_sim_bottle(self, pos=None, ori=None):
        self.col_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height)
        # below line not needed because p.resetSimulation() already
        # deletes all objects
        # if self.bottle_id is not None:
        #     p.removeBody(self.bottle_id)
        if ori is None:
            ori = [0, 0, 0, 1]

        if pos is None:
            pos = self.start_pos

        self.pos = pos
        self.ori = ori
        self.bottle_id = p.createMultiBody(
            baseMass=self.bottle_mass,
            baseInertialFramePosition=self.inertial_shift,
            baseCollisionShapeIndex=self.col_id,
            basePosition=self.pos,
            baseOrientation=self.ori)

        p.changeDynamics(
            bodyUniqueId=self.bottle_id,
            linkIndex=-1,  # no links, -1 refers to bottle base
            lateralFriction=self.lat_fric
        )

    def calc_vert_angle(self, ori=None):
        self.update_pose()
        if ori is None:
            ori = self.ori
        z_axis = np.array([0, 0, 1])
        rot_mat = R.from_quat(ori).as_matrix()
        new_z_axis = rot_mat @ z_axis
        angle = math.acos(z_axis @ new_z_axis /
                          (np.linalg.norm(z_axis) * np.linalg.norm(new_z_axis)))
        # when z-axis rotation is 90deg (upright)
        return angle


class Arm:
    def __init__(self, kuka_id, ee_start_pos=(0.5, 0.3, 0.2), start_ori=(0, 0, 0, 1)):
        self.ee_start_pos = ee_start_pos
        self.start_ori = start_ori
        self.kukaId = kuka_id
        self.base_pos = np.array([0, 0, 0.1])
        self.min_dist = 0.3
        self.default_joint_pose = np.array(
            [0, math.pi / 2, math.pi / 2, 0, 0, 0, 0])

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
        self.rp = np.array(
            [math.pi / 4, (90 + 15) * math.pi / 180, 0, 0, 0, 0, 0])
        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.force = 500  # allow instantenous velocity = target velocity
        self.max_vel = 2 * math.pi / 16  # angular velocity
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

        self.joint_pose = None
        self.init_joints = self.get_target_joints(ee_start_pos, angle=0)
        self.reset(joint_pose=self.init_joints)  # sets joint_pose

    def update_joint_pose(self):
        joint_states = p.getJointStates(self.kukaId, range(self.num_joints))
        self.joint_pose = np.array([state[0] for state in joint_states])

    def get_link_positions(self, joint_pose=None):
        if joint_pose is not None:
            self.reset(joint_pose)
        link_states = p.getLinkStates(self.kukaId, range(self.num_joints))
        return [state[4] for state in link_states]

    def get_joint_link_positions(self, joint_pose=None, start_i=3, end_i=-1):
        # By default ignore the first 3 joints since will not be able to
        # interact with bottle
        joint_positions = self.get_link_positions(joint_pose)
        joint_positions = joint_positions[start_i:end_i]
        midpoints = []
        # only calculate midpoint btwn last static and 1st dynamic
        for i in range(len(joint_positions) - 1):
            midpoint = np.mean(np.array([
                joint_positions[i],
                joint_positions[i + 1]]), axis=0)
            midpoints.append(midpoint)
        # ignore first two links, which are static
        positions = midpoints + joint_positions
        return positions

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
        orn = p.getQuaternionFromEuler([-math.pi / 2, 0, angle - math.pi / 2])
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
