import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from sim_objects import Arm, Bottle


class ActionSpace():
    """Action space defined by incremental changes to individual joints.
    These include positive and negative offsets and no-change if specified
    """
    default_da_rad = 5.0 * math.pi / 180.0  # default 5 degrees offsets

    def __init__(self, num_DOF, da_rad=default_da_rad, include_no_change=False,
                 ignore_last_joint=True):
        self.num_DOF = num_DOF
        self.da_rad = da_rad
        # self.traj_iter_set = [100, 150, 200]
        self.traj_iter_set = [200]

        pos_moves = np.eye(N=num_DOF) * da_rad
        neg_moves = np.eye(N=num_DOF) * -da_rad
        no_change = np.zeros((1, num_DOF))

        if ignore_last_joint:
            pos_moves = pos_moves[:-1, :]
            neg_moves = neg_moves[:-1, :]

        if include_no_change:
            self.actions_mat = np.vstack([
                no_change, pos_moves, neg_moves
            ])
        else:
            self.actions_mat = np.vstack([
                pos_moves, neg_moves
            ])
        self.num_actions = self.actions_mat.shape[0] * len(self.traj_iter_set)
        self.action_ids = list(range(self.num_actions))

    def get_action(self, id):
        assert(isinstance(id, int))
        assert (0 <= id < self.num_actions)
        dq_i = id % self.actions_mat.shape[0]
        num_iters = self.traj_iter_set[int(id / self.actions_mat.shape[0])]
        return (self.actions_mat[dq_i, :], num_iters)


class Environment(object):
    # pybullet_data built-in models
    plane_urdf_filepath = "plane.urdf"
    arm_filepath = "kuka_iiwa/model.urdf"
    table_filepath = "table/table.urdf"
    gripper_path = "kuka_iiwa/kuka_with_gripper.sdf"
    INF = 1e10
    SIM_AVG = 0
    SIM_MOST_COMMON = 1

    def __init__(self, arm, bottle, is_viz=True):
        # store arm and objects
        self.arm = arm
        self.bottle = bottle

        # simulation visualization params
        self.is_viz = is_viz
        self.trail_dur = 1  # length of visulizing arm trajectory
        self.SIM_VIZ_FREQ = 1 / 240.
        self.goal_line_id = None
        self.target_line_id = None

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

    def eval_cost(self, is_fallen, bottle_pos, ee_move_dist):
        # dist = np.linalg.norm(self.target_bottle_pos[:2] - bottle_pos[:2])
        # return self.dist_cost_scale*dist + self.FALL_COST*is_fallen

        # any step incurs penalty of 1, but if falls, extra huge penalty
        return max(ee_move_dist, self.FALL_COST * is_fallen)

    def change_bottle_pos(self, new_pos):
        self.bottle.start_pos = new_pos

    def run_sim_avg(self, action, init_joints=None, bottle_pos=None, bottle_ori=None):
        """
        Randomly sample internal(unobservable to agent) bottle parameters for each iteration, and return cost and next state averaged over all iterations. In cases where bottle falls, next state is not included in average next state.

        NOTE: this is still not stochastic because we overall are using some finalized cost and next state to represent successor of a (state, action) pair. In a true stochastic planner, successors are represented
        by a belief space, or distribution of possible next states, each with
        some probability.
        """
        avg_cost = 0
        avg_next_bpos = np.zeros(2)
        # next_bottle_oris = np.zeros(shape=(4, self.num_rand_samples))  # 4 x N
        avg_joint_pos = np.zeros(self.arm.num_DOF)
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
            cost, bpos, bori, arm_pos = self.run_sim(
                action, init_joints, bottle_pos, bottle_ori)
            avg_cost += cost
            avg_next_bpos += bpos
            # next_bottle_oris[:, sim_iter] = bori
            avg_joint_pos += arm_pos

            # avg_bori = self.avg_quaternion(next_bottle_oris)
        return (avg_cost / float(self.num_rand_samples),
                avg_next_bpos / float(self.num_rand_samples),
                bori,  # just return last bottle orientation for now
                avg_joint_pos / float(self.num_rand_samples))

    def run_sim_mode(self, action, cost_disc, state_disc, init_joints=None, bottle_pos=None, bottle_ori=None):
        """
        Similar to run_sim_avg except output cost and next state are chosen as the mode, or most common pair of outcomes. Outputs are discretized into bins.
        """
        # map discretized costs/states to their counts
        cost_bins = dict()
        max_cost_count, mode_cost = 0, None
        next_state_bins = dict()
        max_next_state_count, mode_next_state = 0, None
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
            cost, ns = self.run_sim(
                action, init_joints, bottle_pos, bottle_ori)

            # store results
            cost_i = cost / cost_disc
            ns_i = (ns / state_disc).astype(int)
            if cost_i in cost_bins:
                cost_bins[cost_i] += 1
            else:
                cost_bins[cost_i] = 1
            if cost_bins[cost_i] > max_cost_count:
                max_cost_count = cost_bins[cost_i]
                mode_cost = cost_i

            if ns_i in next_state_bins:
                next_state_bins[ns_i] += 1
            else:
                next_state_bins[ns_i] = 1
            if next_state_bins[ns_i] > max_next_state_count:
                max_next_state_count = next_state_bins[ns_i]
                mode_next_state = ns_i

        return mode_cost * cost_disc, ns_i * state_disc

    def run_sim(self, action, init_joints=None, bottle_pos=None, bottle_ori=None):
        """Deterministic simulation where all parameters are already set and
        known.

        Arguments:
            action {np.ndarray} -- offset in joint space, generated in ActionSpace
        """
        if init_joints is None:  # use arm's current joint state
            init_joints = self.arm.joint_pose

        dq, num_iters = action
        target_joint_pose = init_joints + dq
        joint_traj = np.linspace(init_joints,
                                 target_joint_pose, num=num_iters)

        return self.simulate_plan(joint_traj=joint_traj, bottle_pos=bottle_pos, bottle_ori=bottle_ori)

    def simulate_plan(self, joint_traj, bottle_pos, bottle_ori):
        """Run simulation with given joint-space trajectory. Does not reset arm
        joint angles after simulation is done, so that value can be guaranteed to be untouched.

        Arguments:
            joint_traj {[type]} -- N x num_DOF trajectory of joints

        Returns:
            [type] -- [description]
        """
        self.arm.reset(joint_traj[0, :])
        init_arm_pos = np.array(p.getLinkState(
            self.arm.kukaId, self.arm.EE_idx)[4])
        prev_arm_pos = np.copy(init_arm_pos)

        # create new bottle object with parameters set beforehand
        if bottle_pos is not None:
            self.bottle.create_sim_bottle(bottle_pos, ori=bottle_ori)
            prev_bottle_pos = bottle_pos
        else:
            self.bottle.create_sim_bottle(ori=bottle_ori)
            prev_bottle_pos = self.bottle.start_pos
        bottle_vert_stopped = False
        bottle_horiz_stopped = False
        bottle_stopped = bottle_vert_stopped and bottle_horiz_stopped
        is_collision = False

        iter = 0
        traj_len = joint_traj.shape[0]
        while iter < traj_len or (iter < self.max_iters and not bottle_stopped):
            # set target joint pose
            next_joint_pose = joint_traj[iter, :]
            for ji, jval in enumerate(next_joint_pose):
                p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                        jointIndex=ji,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jval,
                                        force=self.arm.force,
                                        positionGain=self.arm.position_gain)
            # run one sim iter
            p.stepSimulation()
            self.arm.update_joint_pose()

            contacts = p.getContactPoints(
                self.arm.kukaId, self.bottle.bottle_id)
            is_collision |= (len(contacts) > 0)

            # get feedback and vizualize trajectories
            if self.is_viz and prev_arm_pos is not None:
                ls = p.getLinkState(self.arm.kukaId, self.arm.EE_idx)
                arm_pos = ls[4]
                # Uncomment below to visualize lines of target and actual trajectory
                # also slows down simulation, so only run if trying to visualize
                # p.addUserDebugLine(prev_target, next_target, [0, 0, 0.3], 1, 1)
                # p.addUserDebugLine(arm_pos, prev_arm_pos, [1, 0, 0], 1,
                #                    self.trail_dur)
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
        bottle_pos, bottle_ori = p.getBasePositionAndOrientation(
            self.bottle.bottle_id)

        # remove bottle object, can't just reset pos since need to change params each iter
        p.removeBody(self.bottle.bottle_id)

        return is_fallen, is_collision, bottle_pos, bottle_ori, self.arm.joint_pose

    @staticmethod
    def draw_line(lineFrom, lineTo, lineColorRGB, lineWidth, lifeTime,
                  replaceItemUniqueId=None):
        if replaceItemUniqueId is not None:
            return p.addUserDebugLine(lineFrom, lineTo, lineColorRGB,
                                      lineWidth, lifeTime, replaceItemUniqueId=replaceItemUniqueId)
        else:
            return p.addUserDebugLine(lineFrom, lineTo, lineColorRGB,
                                      lineWidth, lifeTime)

    @staticmethod
    def avg_quaternion(quaternions):
        """Finds average of quaternions from this post. Doesn't seem to work
        too well though.
        https://www.mathworks.com/matlabcentral/fileexchange/40098-tolgabirdal-averaging_quaternions
        """
        A = np.zeros((4, 4))
        assert (quaternions.shape[0] == 4)  # 4 x N
        num_quats = quaternions.shape[1]
        for i in range(num_quats):
            q = quaternions[:, i].reshape((4, 1))
            A += (q @ q.T)
        A /= float(num_quats)

        # can't do eigenvalue decomposition since not square matrix
        U, s, VT = np.linalg.svd(A)
        # eigenvector corresponding to largest eigenvalue is avg quat
        # last eigenvector corresponds to largest eigenvalue
        avg_quat = VT[0, :]
        return avg_quat  # / np.linalg.norm(avg_quat)


def test_environment_avg_quat():
    r = R.from_euler('zyx', [
        [90, 0, 70],
        [45, 20, 0]], degrees=True)
    quaternions = r.as_quat().T
    avg_quat = Environment.avg_quaternion(quaternions)
    avg_angles = R.from_quat(avg_quat).as_euler('zyx', degrees=True)
    print(avg_angles)
