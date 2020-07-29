import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from sim_objects import Arm, Bottle
from trap_velocity_generator import gen_trapezoidal_velocity_profile, State


class ActionSpace():
    """Action space defined by incremental changes to individual joints.
    These include positive and negative offsets and no-change if specified
    """
    default_da_rad = 5.0 * math.pi / 180.0  # default 5 degrees offsets

    def __init__(self, num_DOF, da_rad=default_da_rad, include_no_change=False):
        self.num_DOF = num_DOF
        self.da_rad = da_rad

        pos_moves = np.eye(N=num_DOF) * da_rad
        neg_moves = np.eye(N=num_DOF) * -da_rad
        no_change = np.zeros((1, num_DOF))

        if include_no_change:
            self.actions_mat = np.vstack([
                no_change, pos_moves, neg_moves
            ])
        else:
            self.actions_mat = np.vstack([
                pos_moves, neg_moves
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
    SIM_AVG = 0
    SIM_MOST_COMMON = 1
    SIM_FREQ = 240.0  # Hz
    dt = 1 / SIM_FREQ

    def __init__(self, arm, bottle, state_disc, is_viz=True, use_3D=True, min_iters=10, max_iters=150):
        # store arm and objects
        self.arm = arm
        self.amax = 1
        self.bottle = bottle

        # state discretization for binning states
        self.state_disc = state_disc

        # simulation visualization params
        self.is_viz = is_viz
        self.trail_dur = 1  # length of visulizing arm trajectory

        # simulation run params
        # if no object moves more than this thresh, terminate sim early
        self.no_movement_thresh = 0.001
        self.min_iters = min_iters  # enough iters to let action execute fully
        self.max_iters = max_iters  # max number of iters in case objects oscillating
        # number of random samples of internal params for stochastic simulation
        self.num_rand_samples = 10

        # cost parameters
        self.target_bottle_pos = np.zeros((3,))
        self.FALL_COST = Environment.INF
        self.dist_cost_scale = 100
        self.use_3D = use_3D

        # Normal distribution of internal bottle params
        self.min_fric = 0.15
        self.max_fric = 0.45
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
        sum_cost = 0
        sum_next_bpos = np.zeros(3)
        next_bottle_ori_bins = []
        next_bottle_ori_counts = []
        sum_joint_pos = np.zeros(self.arm.num_DOF)
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
                action, init_joints, bottle_pos, bottle_ori,
                use_vel_control=False)
            sum_cost += cost
            sum_next_bpos += bpos

            # averaging angles  directly like this isn't safe, but assume that  arm will reach its target  orientation the same regardless of bottle parameters
            sum_joint_pos += arm_pos

            # bin bottle orientation
            ori_comparisons = [self.is_quat_equal(
                bori, q) for q in next_bottle_ori_bins]
            match = np.where(ori_comparisons)[0]
            if len(match) == 1:
                match_i = match[0]
                next_bottle_ori_counts[match_i] += 1

            else:
                next_bottle_ori_bins.append(bori)
                next_bottle_ori_counts.append(1)

            # extra optimization: if arm didn't touch bottle, no need for more iterations
            if np.allclose(bpos, bottle_pos, atol=1e-5):
                break

        most_common_ori = next_bottle_ori_bins[np.argmax(
            next_bottle_ori_counts)]

        return (sum_cost / float(self.num_rand_samples),
                sum_next_bpos / float(self.num_rand_samples),
                most_common_ori,
                sum_joint_pos / float(self.num_rand_samples))

    def run_sim_mode(self, action, init_joints=None, bottle_pos=None, bottle_ori=None):
        """
        Similar to run_sim_avg except output cost and next state are chosen as the mode, or most common pair of outcomes. Outputs are discretized into bins.
        """
        class CostCountTuple():
            def __init__(self, count, cost, bori):
                self.count = count
                self.cost = cost
                self.bori = bori

            def __repr__(self):
                return "cost(%.2f), count(%d), bori(%.2f,%.2f,%.2f,%.2f)" % (
                    self.cost, self.count, self.bori[0], self.bori[1],
                    self.bori[2], self.bori[3])

        # map discretized states to their counts and costs
        next_state_bins = dict()
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
                action, init_joints, bottle_pos, bottle_ori,
                use_vel_control=False)
            ns = np.concatenate([bpos, arm_pos])

            # store results
            ns_disc = tuple((ns / self.state_disc).astype(int))
            if ns_disc in next_state_bins:
                next_state_bins[ns_disc].count += 1
                next_state_bins[ns_disc].cost += cost
                # only store bottle orientation once... not great but averaging
                # doesn't work well

            else:
                next_state_bins[ns_disc] = CostCountTuple(
                    count=1, cost=cost, bori=bori)

            # extra optimization: if arm didn't touch bottle, no need for more iterations
            # not comparing z-value because bottle will drop a bit due to ground plane offset
            if np.allclose(bpos[:2], bottle_pos[:2], atol=1e-2):
                break

        # most common next state bin
        mode_ns_disc = max(next_state_bins.keys(),
                           key=lambda ns_disc: next_state_bins[ns_disc].count)
        mode_ns_cont = np.array(mode_ns_disc) * self.state_disc
        mode_bpos = mode_ns_cont[:3]
        mode_arm_pos = mode_ns_cont[3:]

        count = float(next_state_bins[mode_ns_disc].count)
        # don't divide by count since not average
        mode_bori = next_state_bins[mode_ns_disc].bori
        avg_cost = (next_state_bins[mode_ns_disc].cost / count)

        return avg_cost, mode_bpos, mode_bori, mode_arm_pos

    def run_sim(self, action, init_joints=None, bottle_pos=None, bottle_ori=None, use_vel_control=False):
        """Deterministic simulation where all parameters are already set and
        known.

        Arguments:
            action {np.ndarray} -- offset in joint space, generated in ActionSpace
        """
        if init_joints is None:  # use arm's current joint state
            init_joints = np.array(self.arm.joint_pose)
        else:
            init_joints = np.array(init_joints)
        target_joint_pose = init_joints + action

        if use_vel_control:
            joint_vel_traj = np.zeros((self.min_iters, self.arm.num_DOF))
            for qi, dq in enumerate(action):
                if not np.isclose(dq, 0):
                    start = State(x=init_joints[qi], v=0, t=0)
                    end = State(
                        x=target_joint_pose[qi], v=0,
                        t=self.min_iters)
                    # dqmax = self.arm.calc_max_joint_vel(
                    #     ji=qi, dt=self.dt, joint_pose=init_joints)
                    vel_profile = gen_trapezoidal_velocity_profile(
                        start=start, final=end, dt=1.0, duty_cycle=0.2) / self.dt
                    joint_vel_traj[:, qi] = vel_profile

            target_traj = joint_vel_traj

        # Position control
        else:
            # don't create linear interp, just let internal pybullet PID get to target
            target_traj = np.array([init_joints, target_joint_pose])

        return self.simulate_plan(init_pose=init_joints,
                                  traj=target_traj, bottle_pos=bottle_pos, bottle_ori=bottle_ori,
                                  use_vel_control=use_vel_control)

    def simulate_plan(self, init_pose, traj, bottle_pos, bottle_ori,
                      use_vel_control):
        """Run simulation with given joint-space trajectory. Does not reset arm
        joint angles after simulation is done, so that value can be guaranteed to be untouched.

        Arguments:
            joint_traj {[type]} -- N x num_DOF trajectory of joints

        Returns:
            [type] -- [description]
        """
        self.arm.reset(init_pose)
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

        iter = 0
        traj_len = traj.shape[0]
        while iter < self.min_iters or (iter < self.max_iters and not bottle_stopped):
            # set target joint pose
            next_target = traj[min(iter, traj_len - 1), :]
            if use_vel_control:
                for ji, joint_vel in enumerate(next_target):
                    p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                            jointIndex=ji,
                                            controlMode=p.VELOCITY_CONTROL,
                                            targetVelocity=joint_vel,
                                            force=self.arm.force,
                                            velocityGain=0.5)
            else:
                for ji, joint_pos in enumerate(next_target):
                    p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                            jointIndex=ji,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=joint_pos,
                                            force=self.arm.force,
                                            positionGain=self.arm.position_gain)
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
                # p.addUserDebugLine(arm_pos, prev_arm_pos, [1, 0, 0], 1,
                #                    self.trail_dur)
                prev_arm_pos = arm_pos
                # time.sleep(self.SIM_FREQ)

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
        final_arm_pos = np.array(p.getLinkState(
            self.arm.kukaId, self.arm.EE_idx)[4])

        # euclidean distance moved by arm is transition cost
        if self.use_3D:
            EE_move_dist = np.linalg.norm(final_arm_pos[:3] - init_arm_pos[:3])
        else:
            EE_move_dist = np.linalg.norm(final_arm_pos[:2] - init_arm_pos[:2])
        cost = self.eval_cost(is_fallen, bottle_pos, EE_move_dist)

        # remove bottle object, can't just reset pos since need to change params each iter
        p.removeBody(self.bottle.bottle_id)

        return cost, bottle_pos, bottle_ori, self.arm.joint_pose

    @ staticmethod
    def draw_line(lineFrom, lineTo, lineColorRGB, lineWidth, lifeTime):
        p.addUserDebugLine(lineFrom, lineTo, lineColorRGB, lineWidth, lifeTime)

    @staticmethod
    def is_quat_equal(q1, q2, eps=0.005):
        """Eps was determined empirically with several basic tests. Degree tolerance is ~10 degrees.

        Args:
            q1 (np.ndarray): [description]
            q2 (np.ndarray): [description]
            eps (float, optional): [description]. Defaults to 0.005.

        Returns:
            [type]: [description]
        """
        if not isinstance(q1, np.ndarray):
            q1 = np.array(q1)
            q2 = np.array(q2)
        return abs(q1 @ q2) > 1 - eps

    @ staticmethod
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
