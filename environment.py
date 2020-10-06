import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import scipy.stats

from sim_objects import Arm, Bottle


class ActionSpace():
    """Action space defined by incremental changes to individual joints.
    These include positive and negative offsets and no-change if specified
    """
    default_da_rad = 5.0 * math.pi / 180.0  # default 5 degrees offsets

    def __init__(self, num_DOF, da_rad=default_da_rad, include_no_change=False,
                 ignore_last_joint=True):
        self.num_DOF = num_DOF
        # to ensure each action's new state is treated as new state due to discretization
        self.da_rad = da_rad * 1.2
        # self.traj_iter_set = [100, 150, 200]
        self.traj_iter_set = [200]

        pos_moves = np.eye(N=num_DOF) * self.da_rad
        neg_moves = np.eye(N=num_DOF) * -self.da_rad
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


class EnvParams(object):
    def __init__(self, bottle_fill, bottle_fric, bottle_fill_prob,
                 bottle_fric_prob):
        self.bottle_fill = bottle_fill
        self.bottle_fric = bottle_fric
        self.bottle_fill_prob = bottle_fill_prob
        self.bottle_fric_prob = bottle_fric_prob

    def __repr__(self):
        return "fill, fric, pfill, pfric: %.2f, %.2f, %.2f, %.2f" % (
            self.bottle_fill, self.bottle_fric, self.bottle_fill_prob, self.bottle_fric_prob)


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
        # normal distrib for bottle friction
        self.min_fric = 0.05
        self.max_fric = 0.2
        self.mean_friction = (self.min_fric + self.max_fric) / 2.
        # want min and max to be at 3 std deviations
        self.std_friction = (self.max_fric - self.mean_friction) / 3.
        # NOTE: DO NOT USE KWARGS for scipy norm, use ARGS
        # since scipy uses "loc" for mean and "scale" for stddev, avoid passing
        # in wrong kwargs and having them ignored
        self.fric_distrib = scipy.stats.norm(
            self.mean_friction, self.std_friction)

        # normal distrib for bottle fill proportion
        self.min_fill = self.bottle.min_fill
        self.max_fill = 1.0
        self.mean_fillp = (self.min_fill + self.max_fill) / 2.
        self.std_fillp = (self.max_fill - self.mean_fillp) / 3.
        self.fillp_distrib = scipy.stats.norm(
            self.mean_fillp, self.std_fillp)

    def eval_cost(self, is_fallen, bottle_pos, ee_move_dist):
        # dist = np.linalg.norm(self.target_bottle_pos[:2] - bottle_pos[:2])
        # return self.dist_cost_scale*dist + self.FALL_COST*is_fallen

        # any step incurs penalty of 1, but if falls, extra huge penalty
        return max(ee_move_dist, self.FALL_COST * is_fallen)

    def change_bottle_pos(self, new_pos):
        self.bottle.start_pos = new_pos

    def run_multiple_sims(self, action, sim_params_set, init_joints=None,
                          bottle_pos=None, bottle_ori=None):
        """
        Simply run multiple simulations with different environmental parameters.
        Return a list of all results, each entry as a tuple. Let the planner do
        post-processing of these results.
        """
        all_results = []
        for sim_params in sim_params_set:
            results = self.run_sim(action=action,
                                   sim_params=sim_params,
                                   init_joints=init_joints,
                                   bottle_pos=np.copy(bottle_pos),
                                   bottle_ori=bottle_ori)
            all_results.append(results)

            # extra optimization: if arm didn't touch bottle, no need for more
            # iterations
            (_, is_collision, new_bottle_pos, _, _) = results
            if not is_collision:
                break

        return all_results

    def run_sim(self, action, sim_params: EnvParams, init_joints=None, bottle_pos=None, bottle_ori=None):
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

        return self.simulate_plan(joint_traj=joint_traj, bottle_pos=bottle_pos, bottle_ori=bottle_ori, sim_params=sim_params)

    def command_new_pose(self, joint_pose):
        for ji, jval in enumerate(joint_pose):
            p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                    jointIndex=ji,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jval,
                                    force=self.arm.force,
                                    positionGain=self.arm.position_gain)

    def simulate_plan(self, joint_traj, bottle_pos, bottle_ori, sim_params: EnvParams):
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
        self.bottle.set_fill_proportion(sim_params.bottle_fill)
        self.bottle.lat_fric = sim_params.bottle_fric
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
            next_joint_pose = joint_traj[min(iter, traj_len - 1), :]
            self.command_new_pose(next_joint_pose)

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
            self.bottle.pos, self.bottle.ori = p.getBasePositionAndOrientation(
                self.bottle.bottle_id)
            bottle_vert_stopped = math.isclose(
                self.bottle.pos[2] - prev_bottle_pos[2],
                0.0, abs_tol=1e-05)
            bottle_horiz_stopped = math.isclose(
                np.linalg.norm(
                    np.array(self.bottle.pos)[:2] - np.array(prev_bottle_pos)[:2]),
                0.0, abs_tol=1e-05)
            bottle_stopped = bottle_vert_stopped and bottle_horiz_stopped
            prev_bottle_pos = self.bottle.pos

            iter += 1

        # generate cost and final position
        is_fallen = self.bottle.check_is_fallen()
        self.bottle.pos, self.bottle.ori = p.getBasePositionAndOrientation(
            self.bottle.bottle_id)

        # remove bottle object, can't just reset pos since need to change params each iter
        p.removeBody(self.bottle.bottle_id)

        return is_fallen, is_collision, self.bottle.pos, self.bottle.ori, self.arm.joint_pose

    def gen_random_env_param_set(self, num=1):
        rand_fills, rand_fill_probs = self.get_random_sample_prob(
            distrib=self.fillp_distrib, minv=self.min_fill, maxv=self.max_fill, num=num)
        rand_frics, rand_fric_probs = self.get_random_sample_prob(
            distrib=self.fric_distrib, minv=self.min_fric, maxv=self.max_fric, num=num)

        param_set = []
        for i in range(num):
            param = EnvParams(bottle_fill=rand_fills[i],
                              bottle_fric=rand_frics[i],
                              bottle_fill_prob=rand_fill_probs[i],
                              bottle_fric_prob=rand_fric_probs[i])
            param_set.append(param)
        return param_set

    @staticmethod
    def get_random_sample_prob(distrib, minv, maxv, num=1):
        """get N random samples and their "probability"
        Args:
            distrib (scipy.stats.distributions.rv_frozen object): [description]
            min ([type]): [description]
            max ([type]): [description]
            num (int, optional): [description]. Defaults to 1.
        Returns:
            [type]: [description]
        """
        rand_vars = distrib.rvs(size=num)
        rand_vars = np.clip(rand_vars, minv, maxv)
        probs = []
        for v in rand_vars:
            if v < distrib.mean():
                p = distrib.cdf(v)
            else:
                p = 1 - distrib.cdf(v)
            probs.append(p)
        return rand_vars, probs

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

    @ staticmethod
    def state_to_str(state):
        s = ", ".join(["%.3f" % val for val in state])
        return s


def test_environment_avg_quat():
    r = R.from_euler('zyx', [
        [90, 0, 70],
        [45, 20, 0]], degrees=True)
    quaternions = r.as_quat().T
    avg_quat = Environment.avg_quaternion(quaternions)
    avg_angles = R.from_quat(avg_quat).as_euler('zyx', degrees=True)
    print(avg_angles)
