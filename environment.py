import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time

from sim_objects import Arm, Bottle


class ActionSpace():
    """Action space defined by incremental changes to individual joints. 
    These include positive and negative offsets as well as no change to any joint.
    """
    default_da_rad = 5.0 * math.pi / 180.0  # default 5 degrees offsets

    def __init__(self, num_DOF, da_rad=default_da_rad):
        self.num_DOF = num_DOF
        self.da_rad = da_rad

        pos_moves = np.eye(N=num_DOF) * da_rad
        neg_moves = np.eye(N=num_DOF) * -da_rad
        no_change = np.zeros((1, num_DOF))
        self.actions_mat = np.vstack([
            no_change, pos_moves, neg_moves
        ])
        self.num_actions = self.actions_mat.shape[0]
        self.action_ids = list(range(self.num_actions))

    def get_action(self, id):
        assert(isinstance(id, int))
        assert(0 <= id < self.num_actions)
        return self.actions_mat[id, :]


class Environment(object):
    # pybullet_data built-in models
    plane_urdf_filepath = "plane.urdf"
    arm_filepath = "kuka_iiwa/model.urdf"
    table_filepath = "table/table.urdf"
    gripper_path = "kuka_iiwa/kuka_with_gripper.sdf"
    INF = 1e10

    def __init__(self, arm, bottle, is_viz=True, N=500):
        # store arm and objects
        self.arm = arm
        self.bottle = bottle

        # simulation visualization params
        self.is_viz = is_viz
        self.trail_dur = 1  # length of visulizing arm trajectory
        self.SIM_VIZ_FREQ = 1/240.

        # simulation run params
        # if no object moves more than this thresh, terminate sim early
        self.no_movement_thresh = 0.001
        self.min_iters = 10  # enough iters to let action execute fully
        self.max_iters = 150  # max number of iters in case objects oscillating
        # number of random samples of internal params for stochastic simulation
        self.num_rand_samples = 10

        # cost parameters
        self.target_bottle_pos = np.zeros((3,))
        self.FALL_COST = Environment.INF
        self.MOVE_COST = 0  # 1
        self.dist_cost_scale = 100

        # Normal distribution of internal bottle params
        self.min_fric = 0.1
        self.max_fric = 0.2
        self.min_fill = self.bottle.min_fill
        self.max_fill = 1.0
        self.mean_friction = (self.min_fric + self.max_fric) / 2.
        # want min and max to be at 3 std deviations
        self.std_friction = (self.max_fric - self.mean_friction) / 3.
        self.mean_fillp = (self.min_fill + self.max_fill) / 2.
        self.std_fillp = (self.max_fill - self.mean_fillp) / 3.

    def eval_cost(self, is_fallen, bottle_pos):
        # dist = np.linalg.norm(self.target_bottle_pos[:2] - bottle_pos[:2])
        # return self.dist_cost_scale*dist + self.FALL_COST*is_fallen

        # any step incurs penalty of 1, but if falls, extra huge penalty
        return max(self.MOVE_COST, self.FALL_COST*is_fallen)

    def change_bottle_pos(self, new_pos):
        self.bottle.start_pos = new_pos

    def run_sim_stochastic(self, action, init_joints=None):
        """
        Randomly sample internal(unobservable to agent) bottle parameters.
        """
        expected_cost = 0
        expected_next_state = np.zeros((2,))
        for sim_iter in range(self.num_rand_samples):
            # randomly sample friction and fill-prop of bottle
            rand_fill = np.random.normal(
                loc=self.mean_fillp, scale=self.std_fillp)
            rand_fill = np.clip(rand_fill, self.min_fill, self.max_fill)
            rand_fric = np.random.normal(
                loc=self.mean_friction, scale=self.std_friction)
            rand_fric = np.clip(rand_fric, self.min_fric, self.max_fric)

            # set random parameters
            self.bottle.set_fill_proportion(rand_fill)
            self.bottle.lat_fric = rand_fric

            # run sim deterministically and average cost and new bottle pos
            cost, ns = self.run_sim(action, init_joints)
            expected_cost += cost
            expected_next_state += ns

        return (expected_cost / float(self.num_rand_samples),
                expected_next_state / float(self.num_rand_samples))

    def run_sim(self, action, init_joints=None, bottle_pos=None):
        """Deterministic simulation where all parameters are already set and 
        known.

        Arguments:
            action {np.ndarray} -- offset in joint space, generated in ActionSpace
        """
        if init_joints is None:  # use arm's current joint state
            init_joints = self.arm.joint_pose

        target_joint_pose = init_joints + action
        joint_traj = np.linspace(init_joints,
                                 target_joint_pose, num=self.min_iters)

        return self.simulate_plan(joint_traj=joint_traj, bottle_pos=bottle_pos)

    def simulate_plan(self, joint_traj, bottle_pos):
        """Run simulation with given joint-space trajectory.

        Arguments:
            joint_traj {[type]} -- N x num_DOF trajectory of joints

        Returns:
            [type] -- [description]
        """
        self.arm.reset(joint_traj[0, :])
        prev_arm_pos = p.getLinkState(self.arm.kukaId, self.arm.EE_idx)[4]

        # create new bottle object with parameters set beforehand
        if bottle_pos is not None:
            self.bottle.create_sim_bottle(bottle_pos)
            prev_bottle_pos = bottle_pos
        else:
            self.bottle.create_sim_bottle()
            prev_bottle_pos = self.bottle.start_pos
        bottle_vert_stopped = False
        bottle_horiz_stopped = False
        bottle_stopped = bottle_vert_stopped and bottle_horiz_stopped

        # iterate through action
        pos_change = 0  # init arbitrary
        thresh = 0.001
        EE_error = 0

        iter = 0
        while iter < self.min_iters or (iter < self.max_iters and not bottle_stopped):
            # set target joint pose
            if iter < self.min_iters:
                next_joint_pose = joint_traj[iter, :]
                for ji, jval in enumerate(next_joint_pose):
                    p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                            jointIndex=ji,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=jval,
                                            targetVelocity=0,
                                            force=self.arm.force,
                                            positionGain=self.arm.position_gain,
                                            velocityGain=self.arm.velocity_gain)
            # run one sim iter
            p.stepSimulation()
            self.arm.update_joint_pose()

            # get feedback and vizualize trajectories
            if self.is_viz and prev_arm_pos is not None:
                ls = p.getLinkState(self.arm.kukaId, self.arm.EE_idx)
                arm_pos = ls[4]
                # Uncomment below to visualize lines of target and actual trajectory
                # also slows down simulation, so only run if trying to visualize
                # p.addUserDebugLine(prev_target, next_target, [0, 0, 0.3], 1, 1)
                p.addUserDebugLine(arm_pos, prev_arm_pos, [1, 0, 0], 1,
                                   self.trail_dur)
                prev_arm_pos = arm_pos
                # time.sleep(self.SIM_VIZ_FREQ)

            # check status of other objects to possibly terminate sim early
            bottle_pos, bottle_ori = p.getBasePositionAndOrientation(
                self.bottle.bottle_id)
            bottle_vert_stopped = math.isclose(
                bottle_pos[2] - prev_bottle_pos[2],
                0.0, abs_tol=1e-05)
            bottle_horiz_stopped = math.isclose(
                np.linalg.norm(
                    np.array(bottle_pos)[:2] - np.array(prev_bottle_pos)[:2]),
                0.0, abs_tol=1e-05)
            bottle_stopped = bottle_vert_stopped and bottle_horiz_stopped
            prev_bottle_pos = bottle_pos

            iter += 1

        # generate cost and final position
        is_fallen = self.bottle.check_is_fallen()
        cost = self.eval_cost(is_fallen, bottle_pos)
        final_bottle_pos = np.array(bottle_pos[:2])

        # remove bottle object, can't just reset pos since need to change params each iter
        p.removeBody(self.bottle.bottle_id)

        return cost, final_bottle_pos
