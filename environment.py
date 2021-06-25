import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import scipy.stats
import typing as T
import collections

from sim_objects import Arm, Bottle

SimResults = collections.namedtuple(
    'SimResults', ['is_fallen', 'is_collision',
                   'bottle_pos', 'bottle_ori', 'joint_pose', 'z_rot_ang'])
StateTuple = collections.namedtuple('StateTuple', ['bottle_pos', 'bottle_ori', 'joints'])


class ActionSpace(object):
    """Action space defined by incremental changes to individual joints.
    These include positive and negative offsets and no-change if specified
    """
    default_da_rad = 5.0 * math.pi / 180.0  # default 5 degrees offsets

    def __init__(self, num_dof, da_rad=default_da_rad, include_no_change=False,
                 ignore_last_joint=True):
        self.num_dof = num_dof
        # to ensure each action's new state is treated as new state due to discretization
        self.da_rad = da_rad * 1.2
        # self.traj_iter_set = [200, 400]
        self.traj_iter_set = [200]
        self.max_iters = max(self.traj_iter_set)

        pos_moves = np.eye(N=num_dof) * self.da_rad
        neg_moves = np.eye(N=num_dof) * -self.da_rad
        no_change = np.zeros((1, num_dof))

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
        assert (isinstance(id, int))
        assert (0 <= id < self.num_actions)
        dq_i = id % self.actions_mat.shape[0]
        num_iters = self.traj_iter_set[int(id / self.actions_mat.shape[0])]
        return self.actions_mat[dq_i, :], num_iters

    def get_action_time_cost(self, action: T.Tuple):
        num_iters = action[1]
        return self.max_iters / float(num_iters)


class EnvParams(object):
    def __init__(self, bottle_fill, bottle_fric, bottle_fill_prob,
                 bottle_fric_prob):
        self.bottle_fill = bottle_fill
        self.bottle_fric = bottle_fric
        self.bottle_fill_prob = bottle_fill_prob
        self.bottle_fric_prob = bottle_fric_prob

    def __repr__(self):
        return "fill, fric, pfill, pfric: %.3f, %.3f, %.2f, %.2f" % (
            self.bottle_fill, self.bottle_fric, self.bottle_fill_prob, self.bottle_fric_prob)

    def __add__(self, other):
        return EnvParams(bottle_fill=self.bottle_fill + other.bottle_fill,
                         bottle_fric=self.bottle_fric + other.bottle_fric,
                         bottle_fill_prob=self.bottle_fill_prob + other.bottle_fill_prob,
                         bottle_fric_prob=self.bottle_fric_prob + other.bottle_fric_prob)

    def __radd__(self, other):
        if isinstance(other, int):
            return self
        return self + other

    def __truediv__(self, other):
        return EnvParams(bottle_fill=self.bottle_fill / other,
                         bottle_fric=self.bottle_fric / other,
                         bottle_fill_prob=self.bottle_fill_prob / other,
                         bottle_fric_prob=self.bottle_fric_prob / other)


class Environment(object):
    # pybullet_data built-in models
    plane_urdf_filepath = "plane.urdf"
    arm_filepath = "kuka_iiwa/model.urdf"
    table_filepath = "table/table.urdf"
    gripper_path = "kuka_iiwa/kuka_with_gripper.sdf"
    INF = 1e10
    SIM_AVG = 0
    SIM_MOST_COMMON = 1
    GRAVITY = -9.81

    def __init__(self, arm: Arm, bottle: Bottle, is_viz):
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
        self.max_iters = 600  # max number of iters in case objects oscillating

        # Default distributions for sampling bottle parameters
        self.fric_distrib = None
        self.fillp_distrib = None
        self.set_distribs()

        # tolerance for terminating simulation
        self.min_ang_rot = 0.8  # deg / SIM_VIZ_FREQ
        self.fall_ang_thresh = 30 * math.pi / 180.0
        self.no_movement_thresh = 3e-3

    def set_distribs(self, min_fric=None, max_fric=None, min_fill=None, max_fill=None):
        if min_fric is None:
            min_fric = self.bottle.min_fric
        if max_fric is None:
            max_fric = self.bottle.max_fric
        if min_fill is None:
            min_fill = self.bottle.min_fill
        if max_fill is None:
            max_fill = self.bottle.max_fill

        mean_fric = (min_fric + max_fric) / 2.
        std_fric = (max_fric - mean_fric) / 2.5  # want min and max to be at 2.5 std deviations
        # NOTE: DO NOT USE KWARGS for scipy norm, use ARGS
        # since scipy uses "loc" for mean and "scale" for stddev, avoid passing
        # in wrong kwargs and having them ignored
        self.fric_distrib = scipy.stats.norm(mean_fric, std_fric)

        mean_fillp = (min_fill + max_fill) / 2.
        # std_fillp = (max_fill - mean_fillp) / 3.
        self.fillp_distrib = scipy.stats.uniform(loc=min_fill, scale=max_fill - min_fill)

        print("Mean Fill: %.3f" % mean_fillp)
        print("Mean Fric: %.3f, Std: %.3f" % (mean_fric, std_fric))

    def change_bottle_pos(self, new_pos):
        self.bottle.start_pos = new_pos

    def run_multiple_sims(self, sim_params_set: T.List[EnvParams], action,
                          state: StateTuple):
        """
        Simply run multiple simulations with different bottle parameters.
        Return a list of all results, each entry as a tuple. Let the planner do
        post-processing of these results.
        """
        all_results = []
        for sim_params in sim_params_set:
            results = self.run_sim(action=action,
                                   sim_params=sim_params,
                                   state=state)
            all_results.append(results)

            # extra optimization: if arm didn't touch bottle, no need for more
            # iterations since different bottle friction/mass won't change outcome
            if not results.is_collision:
                break

        return all_results

    def run_sim(self, sim_params: EnvParams, action: T.Tuple,
                state: StateTuple) -> SimResults:
        """
        High-level interface with simulator: run simulation given some current state composed of
        bottle pose and arm joint poise. Specify some action to take. Generates a joint-space trajectory
        for lower-level simulation function to execute.
        """
        start_time = time.time()
        if state.joints is None:  # use arm's current joint state
            state.joints = self.arm.joint_pose

        dq, num_iters = action
        target_joint_pose = state.joints + dq
        joint_traj = np.linspace(state.joints, target_joint_pose, num=num_iters)

        bottle_pos = state.bottle_pos
        bottle_ori = state.bottle_ori

        results = self.simulate_plan(joint_traj=joint_traj,
                                     start_bottle_pos=bottle_pos, start_bottle_ori=bottle_ori,
                                     sim_params=sim_params)
        end_time = time.time()
        print("run_sim: %.4f" % (end_time - start_time))
        return results

    def reset(self):
        # pass
        p.resetSimulation()
        p.setGravity(0, 0, self.GRAVITY)
        p.loadURDF(self.plane_urdf_filepath, basePosition=[0, 0, 0])
        self.arm.kukaId = p.loadURDF(self.arm_filepath, basePosition=[0, 0, 0])

    def command_new_pose(self, joint_pose):
        for ji, jval in enumerate(joint_pose):
            p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                    jointIndex=ji,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jval,
                                    force=self.arm.force,
                                    positionGain=self.arm.position_gain)

    def simulate_plan(self, joint_traj, start_bottle_pos, start_bottle_ori, sim_params: EnvParams) -> SimResults:
        """Run simulation with given joint-space trajectory. Does not reset arm
        joint angles after simulation is done, so that value can be guaranteed to be untouched.

        Arguments:
            joint_traj {[type]} -- N x num_dof trajectory of joints

        Returns:
            [type] -- [description]
        """
        start = time.time()
        self.reset()
        end = time.time()
        sim_reset = end - start

        start = time.time()
        self.arm.reset(joint_traj[0, :])
        init_arm_pos = np.array(p.getLinkState(
            self.arm.kukaId, self.arm.EE_idx)[4])
        prev_arm_pos = np.copy(init_arm_pos)

        # create new bottle object with parameters set beforehand
        self.bottle.set_fill_proportion(sim_params.bottle_fill)
        self.bottle.set_fric(sim_params.bottle_fric)
        if start_bottle_pos is not None:
            self.bottle.create_sim_bottle(start_bottle_pos, ori=start_bottle_ori)
            prev_bottle_pos = start_bottle_pos
            prev_bottle_ori = start_bottle_ori
        else:
            self.bottle.create_sim_bottle(ori=start_bottle_ori)
            prev_bottle_pos = self.bottle.start_pos
            prev_bottle_ori = self.bottle.start_ori
        bottle_stopped = False
        is_collision = False
        end = time.time()
        setup_objects = end - start

        iter = 0
        traj_len = joint_traj.shape[0]

        command = 0
        step_sim = 0
        collision_time = 0
        check_bottle_state = 0
        while iter < traj_len:
            # set target joint pose
            start = time.time()
            next_joint_pose = joint_traj[min(iter, traj_len - 1), :]
            self.command_new_pose(next_joint_pose)
            end = time.time()
            command += end - start
            # print(np.concatenate([self.bottle.pos, self.arm.joint_pose]))

            # run one sim iter
            start = time.time()
            p.stepSimulation()
            end = time.time()
            step_sim += end - start

            start = time.time()
            contacts = p.getContactPoints(
                self.arm.kukaId, self.bottle.bottle_id)
            end = time.time()
            collision_time += end - start
            # if len(contacts) > 0 and not is_collision:
            #     print("COLLISION!!!")
            #     print(iter)
            is_collision |= (len(contacts) > 0)

            # get feedback and vizualize trajectories
            if self.is_viz and prev_arm_pos is not None:
                time.sleep(0.003)
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
            start = time.time()
            self.bottle.update_pose()
            bottle_vert_stopped = math.isclose(
                self.bottle.pos[2] - prev_bottle_pos[2],
                0.0, abs_tol=1e-05)
            bottle_horiz_stopped = math.isclose(
                np.linalg.norm(
                    np.array(self.bottle.pos)[:2] - np.array(prev_bottle_pos)[:2]),
                0.0, abs_tol=1e-05)
            # angle_diff = abs(self.bottle.calc_vert_angle(ori=prev_bottle_ori) -
            #                  self.bottle.calc_vert_angle()) * 180 / math.pi
            # bottle_angle_stopped = angle_diff <= self.min_ang_rot
            # bottle_stopped = bottle_vert_stopped and bottle_horiz_stopped  # and bottle_angle_stopped
            # prev_bottle_pos = self.bottle.pos
            # prev_bottle_ori = self.bottle.ori
            end = time.time()
            check_bottle_state += end - start

            iter += 1

        # generate cost and final position
        start = time.time()
        self.arm.update_joint_pose()
        self.bottle.update_pose()
        is_fallen, z_rot_ang = self.check_bottle_fallen(ori=self.bottle.ori)
        no_movement = np.linalg.norm(start_bottle_pos - self.bottle.pos) < self.no_movement_thresh
        end = time.time()
        check_bottle_fall = end - start

        # remove bottle object, can't just reset pos since need to change params each iter
        start = time.time()
        p.removeBody(self.bottle.bottle_id)
        end = time.time()
        remove_time = end - start

        # print("sim_reset: %.5f" % sim_reset)
        # print("command: %.5f" % command)
        # print("setup_objects: %.5f" % setup_objects)
        # print("step_sim: %.5f" % step_sim)
        # print("collision_time: %.5f" % collision_time)
        # print("check_bottle_state: %.5f" % (check_bottle_state + check_bottle_fall))
        # print("remove_time: %.5f" % remove_time)

        # , executed_traj
        return SimResults(is_fallen=is_fallen, is_collision=is_collision and not no_movement,
                          bottle_pos=self.bottle.pos, bottle_ori=self.bottle.ori,
                          joint_pose=self.arm.joint_pose, z_rot_ang=z_rot_ang)

    def check_bottle_fallen(self, ori):
        angle = self.bottle.calc_vert_angle(ori)
        return abs(angle) > self.fall_ang_thresh, angle

    def gen_random_env_param_set(self, num=1):
        rand_fills, rand_fill_probs = self.get_random_sample_prob(
            distrib=self.fillp_distrib, minv=self.bottle.min_fill, maxv=self.bottle.max_fill, num=num)
        rand_frics, rand_fric_probs = self.get_random_sample_prob(
            distrib=self.fric_distrib, minv=self.bottle.min_fric, maxv=self.bottle.max_fric, num=num)

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

    @staticmethod
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
