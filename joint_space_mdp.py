import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace


class MDP():
    INF = Environment.INF

    def __init__(self, env, DEBUG=False):
        self.DEBUG = DEBUG
        # set environment and action space
        self.env = env
        self.A = ActionSpace(num_DOF=env.arm.num_joints)
        self.max_joint_indexes = (
            (self.env.arm.ul - self.env.arm.ll) / da).astype(int)

        # discount factor
        self.gamma = 0.8

        # state consists of [joints, ]
        # only consider one quadrant
        self.dx = self.dy = 0.1
        self.xmin, self.xmax = 0, 0.8
        self.ymin, self.ymax = 0, 0.8
        self.X = np.arange(
            start=self.xmin,
            stop=self.xmax+self.dx,
            step=self.dx)
        self.Y = np.arange(
            start=self.ymin,
            stop=self.ymax+self.dy,
            step=self.dy)
        self.H, self.W = len(self.Y), len(self.X)

        self.OUT_OF_BOUNDS_COST = MDP.INF
        self.V = dict()
        self.P = dict()

    def state_to_key(self, state):
        # (q1, q2, q3, q4, q5, q6, q7, x, y) = state
        da = self.A.da_rad
        joint_state = state[0:self.env.arm.num_joints]
        joint_indexes = np.rint((joint_state - self.env.arm.ll) / da)
        # make sure indexes lie within valid bounds
        joint_indexes = np.clip(joint_indexes, np.zeros_like(
            joint_indexes), self.max_joint_indexes)

        x, y = state[self.env.arm.num_joints:]
        xi = int(round((x - self.xmin)/self.dx))
        yi = int(round((y - self.ymin)/self.dy))
        # usually leave out-of-bounds as-is, but for special case of rounding
        # up to bound, just keep within index range
        if xi == self.W:
            xi -= 1
        if yi == self.H:
            yi -= 1

        key = tuple(np.append(joint_indexes, [xi, yi]))
        return key

    def solve_mdp(self):
        np.savez("value_policy_iter_0", V=self.V, P=self.P)
        for self.main_iter in range(self.max_iters):
            self.update_policy()
            self.evaluate_policy()
            np.savez("results/value_policy_iter_%d" %
                     (self.main_iter+1), V=self.V, P=self.P)
            print("Percent complete: %.3f" %
                  (self.main_iter / float(self.max_iters)))

    def evaluate_policy(self):
        """Use current policy to estimate new value function
        """
        print("Evaluting Policy to Update Value Function...")
        # new_V = np.ones_like(self.V) * self.OUT_OF_BOUNDS_COST
        num_states = float(len(self.valid_states))
        start = time.time()
        for (x, y) in self.valid_states:
            self.env.change_bottle_pos(
                new_pos=[x, y, 0.1],
                target_type=self.target_type)
            (xi, yi) = self.state_to_idx((x, y))
            best_actions = self.P[yi][xi]
            expected_value = 0
            # uniform distb for best actions
            prob = 1./float(len(best_actions))
            for ai in best_actions:
                action = self.A[ai]
                # if desired x,y,z is out of reach, skip this action
                # validity depends on contact height
                target_dist = np.linalg.norm(np.array([x, y]))
                if target_dist > self.env.arm.calc_max_horiz_dist(
                        action.height):
                    continue

                value, ns = self.env.run_sim_stochastic(action)
                try:
                    (nxi, nyi) = self.state_to_idx(ns)
                    expected_future_value = self.V[nyi, nxi]
                except IndexError:
                    if self.cost_based:
                        expected_future_value = self.OUT_OF_BOUNDS_COST
                    else:
                        expected_future_value = -1 * self.OUT_OF_BOUNDS_COST
                    # expected_future_cost = 0
                expected_value += prob * (
                    value + self.gamma*expected_future_value)

            # synchronous update
            self.V[yi, xi] = expected_value

        # self.V = new_V

        end = time.time()
        print("Total Runtime of Eval Policy: %.3f" % (end-start))

    def update_policy(self):
        """Use current value function to estimate new policy.
        """
        print("Updating Policy...")
        num_states = float(len(self.valid_states))
        total_time = 0

        # for each state, find best action(s) to take
        for (x, y) in self.valid_states:
            start = time.time()
            self.env.change_bottle_pos(
                new_pos=[x, y, 0.1],
                target_type=self.target_type)
            (xi, yi) = self.state_to_idx((x, y))
            best_value = 0
            best_actions = []
            sim_log = []
            for ai, action in enumerate(self.A):
                # if desired x,y,z is out of reach, skip this action
                # validity depends on contact height
                target_dist = np.linalg.norm(np.array([x, y]))
                if (target_dist > self.env.arm.calc_max_horiz_dist(
                        action.height)):
                    continue

                value, ns = self.env.run_sim_stochastic(action)

                try:
                    (nxi, nyi) = self.state_to_idx(ns)
                    expected_future_value = self.V[nyi, nxi]
                except IndexError:
                    # just use value at current state if next state is out of  bounds
                    # expected_future_cost = self.V[yi, xi]
                    if self.cost_based:
                        expected_future_value = self.OUT_OF_BOUNDS_COST
                    else:
                        expected_future_value = -1 * self.OUT_OF_BOUNDS_COST
                    # expected_future_cost = 0

                total_value = value + self.gamma*expected_future_value
                sim_log.append((ns, value, expected_future_value, total_value))

                if self.cost_based:
                    found_better = (total_value < best_value)
                else:
                    found_better = (total_value > best_value)
                if len(best_actions) == 0 or found_better:
                    best_value = total_value
                    best_actions = [ai]
                elif math.isclose(total_value, best_value, abs_tol=1e-6):
                    best_actions.append(ai)

            # self.plot_sim_results(sim_log)
            if self.DEBUG:
                filename = "logs/action_costs_of_%d_%d_iter_%d" % (
                    xi, yi, self.main_iter)
                np.save(filename, sim_log)

            self.P[yi][xi] = best_actions
            end = time.time()
            print("Time(s) for one state: %.3f" % (end - start))
            total_time += (end-start)
        print("Total Runtime of Update Policy: %.3f" % total_time)

    def init_action_space(self, run_full_mdp):
        A = []  # need to maintain order
        self.da = math.pi/80
        if run_full_mdp:
            self.dh = 5
            self.velocities = np.arange(start=0.1, stop=0.31, step=0.1)
            self.angle_offsets = np.arange(
                start=-2*self.da, stop=3*self.da, step=self.da)
        else:
            self.dh = 3
            self.velocities = np.arange(start=0.1, stop=0.21, step=0.05)
            self.angle_offsets = np.arange(
                start=-2*self.da, stop=3*self.da, step=self.da)

        self.contact_heights = np.arange(
            start=self.env.bottle.height/self.dh,
            stop=self.env.bottle.height + self.env.bottle.height/self.dh,
            step=self.env.bottle.height/self.dh)

        self.dr = 0.25  # proportion of max reach
        self.reach_ranges = np.arange(
            start=self.dr,
            stop=1.0+self.dr,
            step=self.dr)

        for h in self.contact_heights:
            for v in self.velocities:
                for a in self.angle_offsets:
                    for r in self.reach_ranges:
                        action = Action(
                            angle_offset=a, velocity=v, height=h, reach_p=r)
                        A.append(action)

        return A

    def plot_sim_results(self, sim_log):
        # n = len(sim_log)
        # color_vals = np.random.randint(0, 0xFFFFFF, size=n)  # +1 for target
        # colors = [('#%06X' % v) for v in color_vals]
        for (_, start, end) in sim_log:
            print(start, end)
            plt.plot([start[0], end[0]], [start[1], end[1]])

        plt.legend([str(action) for action in self.A], loc='upper left')
        plt.show()

    def test_reach_p(self):
        action = self.A[0]
        action.velocity = 0.2
        action.height = self.contact_heights[-1]
        for x in self.X:
            for y in self.Y:
                # if desired x,y,z is out of reach, skip this action
                # validity depends on contact height
                dist_from_base = np.linalg.norm(np.array([x, y]))
                max_reach = self.env.arm.calc_max_horiz_dist(action.height)
                too_close = (self.env.init_reach_p * dist_from_base <
                             self.env.arm.min_dist)
                too_far = dist_from_base > self.env.arm.MAX_REACH
                if (too_close or too_far):
                    print("%.2f,%.2f,%.2f out of reach, skipping.." %
                          (x, y, action.height))
                    continue

                self.env.change_bottle_pos([x, y, 0.1],
                                           target_type=self.target_type)
                for rp in self.reach_ranges:
                    print("%.2f, %.2f, reach: %.1f" % (x, y, rp))
                    action.reach_p = rp
                    cost, ns = self.env.run_sim(action)

    def test_action_space(self):
        x = 0.80
        y = 0.10
        # best action was 20
        xi, yi = self.state_to_idx((x, y))
        self.env.change_bottle_pos([x, y, 0.1], target_type=self.target_type)
        for ai, action in enumerate(self.A):
            print(ai, action)
            cost, ns = self.env.run_sim(action)
            print(cost)

    def view_state_space(self):
        # visualize
        action = self.A[0]
        action.velocity = 0.1
        action.reach_p = 0.25
        x = self.X[5]
        y = self.Y[7]
        for x in self.X:
            for y in self.Y:
                dist_from_base = np.linalg.norm(
                    np.array([x, y]) - self.env.arm.base_pos[:2])
                if (dist_from_base < self.env.arm.min_dist or
                        dist_from_base > self.env.arm.MAX_REACH):
                    continue
                self.env.change_bottle_pos(
                    [x, y, 0.1], target_type=self.target_type)
                cost, ns = self.env.run_sim(action)
                dist = np.linalg.norm(ns - self.env.target_bottle_pos[:2])
                print(cost, dist, self.env.target_thresh,
                      ns, self.env.target_bottle_pos[:2])


def main():
    # initialize simulator environment
    VISUALIZE = True
    LOGGING = False
    GRAVITY = -9.81
    RUN_FULL_MDP = False
    if VISUALIZE:
        p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, GRAVITY)
    planeId = p.loadURDF(Environment.plane_urdf_filepath)
    kukaId = p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0])
    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, "fully_functional.mp4")

    # starting end-effector pos, not base pos
    EE_start_pos = np.array([0.2, 0.2, 0.3]).astype(float)
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    arm = Arm(kuka_id=kukaId, ee_start_pos=EE_start_pos, start_ori=base_start_ori)

    # bottle
    bottle_start_pos = np.array([0.7, 0.6, 0.1]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)

    N = 700
    cost_based = False
    env = Environment(arm, bottle, is_viz=VISUALIZE, N=N,
                      run_full_mdp=RUN_FULL_MDP, cost_based=cost_based)

    solver = MDP(env, RUN_FULL_MDP, DEBUG=True, target_type="const",
                 cost_based=cost_based)
    # solver.test_action_space()
    # solver.solve_mdp()
    # solver.test_reach_p()
    examine_results(mdp=solver)
    # solver.view_state_space()
    # check_results(env, mdp=solver)
    # test_bottle_dynamics(env)
    # A = init_action_space(env.bottle, RUN_FULL_MDP)
    # test(env, A[0])
    # test_state_to_idx(mdp=solver)

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

    # to deal with the issue of bottle being pushed out of range of arm and reaching a state with some unknown value since we can't even reach, just leave V(out of bounds states) = 0, but set every state's reward to be euclidean distance from some goal


def check_results(env: Environment, mdp: MDP):
    path = "results/value_policy_iter_%d.npz"
    print("Value Functions:")
    policies = []
    for iter in range(1, mdp.max_iters+1):
        data = np.load(path % iter, allow_pickle=True)
        V, P = data["V"], data["P"]
        policies.append(P)
        print(V)

    print("Policies:")
    for P in policies:
        print(P)


def test_state_to_idx(mdp: MDP):
    # self.dx = self.dy = 0.1
    # self.xlim, self.ylim = 1.5, 1.5
    # self.X = np.arange(start=-self.xlim, stop=self.xlim, step=self.dx)
    # self.Y = np.arange(start=-self.ylim, stop=self.ylim, step=self.dy)
    x, y = mdp.xmin, mdp.ymin
    print(mdp.state_to_idx((x, y)))
    x, y = mdp.xmax, mdp.ymax
    print(mdp.state_to_idx((x, y)))
    x, y = 0, 0
    print(mdp.state_to_idx((x, y)))
    x, y = mdp.xmax + mdp.dx*0.9, mdp.ymax + mdp.dy*0.9
    print(mdp.state_to_idx((x, y)))


def plot_heat_map(mat, horiz_ticks, vert_ticks, title, xlabel, ylabel):
    df_cm = pd.DataFrame(mat, horiz_ticks, vert_ticks)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    cmap = sn.cm.rocket_r
    sn.heatmap(df_cm, annot=True, annot_kws={
               "size": 16}, cmap=cmap)  # font size
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def test_bottle_dynamics(env):
    # [-0.6999999999999993, -0.29999999999999893, 0.1]
    fill_prop, lat_fric = 1.0, 0.1
    env.change_bottle_pos(new_pos=[-0.7, -0.3, 0.1])
    env.bottle.set_fill_proportion(fill_prop)
    env.bottle.lat_fric = lat_fric
    action = Action(0, 0.01, env.bottle.height / 2)
    env.run_sim(action)


def examine_results(mdp: MDP):
    horiz_ticks = ["%.2f" % x for x in mdp.X]
    vert_ticks = ["%.2f" % y for y in mdp.Y]
    # for iter in range(1,10):
    #     data = np.load("results/value_policy_iter_%d.npz" % iter, allow_pickle=True)
    #     if iter > 1:
    #         prev_data = np.load("results/value_policy_iter_%d.npz" % (iter-1), allow_pickle=True)
    #     else: prev_data = mdp.V
    #     V, P = data["V"], data["P"]

    #     plot_heat_map(V, horiz_ticks=horiz_ticks, vert_ticks=vert_ticks,
    #         title="Iter %d V-table" % iter, xlabel="X", ylabel="Y")
    #     print(np.array_str(V, suppress_small=True, precision=2))
    #     print(np.array_str(P, suppress_small=True, precision=2))
    # target = np.array([0.65, 0.55, 0.1])
    # txi, tyi = mdp.state_to_idx((target[1], target[0]))
    # print(txi, tyi)
    # print(np.array_str(V, suppress_small=True, precision=2))
    # print(np.array_str(P, suppress_small=True, precision=2))

    iter = 9
    data = np.load("good_results_arm_needs_tuning/value_policy_iter_%d.npz" %
                   iter, allow_pickle=True)
    V, P = data["V"], data["P"]
    for (x, y) in mdp.valid_states:
        print("x:%.2f, y:%.2f:" % (x, y))
        (xi, yi) = mdp.state_to_idx((x, y))
        best_actions = P[yi, xi]
        print(best_actions)
        if isinstance(best_actions, list):
            for ai in best_actions:
                action = mdp.A[ai]
                print(action)
                mdp.env.change_bottle_pos(
                    [x, y, 0.1], target_type=mdp.target_type)
                cost, ns = mdp.env.run_sim_stochastic(action)
        else:
            ai = best_actions
            action = mdp.A[ai]
            print(action)
            mdp.env.change_bottle_pos([x, y, 0.1], target_type=mdp.target_type)
            cost, ns = mdp.env.run_sim_stochastic(action)

    #         try:
    #             (nxi, nyi) = mdp.state_to_idx(ns)
    #             expected_future_cost = V[nyi, nxi]
    #         except IndexError:
    #             # just use value at current state if next state is out of  bounds
    #             expected_future_cost = V[yi, xi]
    #             # expected_future_cost = 0

    #         total_cost = cost + mdp.gamma*expected_future_cost
    #         print("nyi, nxi: %d, %d, Y: %d, X: %d, cost:%.2f, future:%.2f"
    #             % (nyi, nxi, mdp.H, mdp.W, cost, expected_future_cost))


def test(env, action):
    X = [0.5, 1.5]
    Y = np.arange(start=-0.5, stop=1.5, step=0.1)
    velocities = np.arange(start=0.1, stop=0.31, step=0.1)
    for x in X:
        for y in Y:
            for v in velocities:
                action.velocity = v
                print(v)
                action.reach_p = 1
                dist_from_base = np.linalg.norm(
                    np.array([x, y]) - env.arm.base_pos[:2])
                if (dist_from_base < env.arm.min_dist or
                        dist_from_base > env.arm.MAX_REACH):
                    continue
                env.change_bottle_pos([x, y, 0.1])
                expected_cost = env.run_sim(action)


if __name__ == '__main__':
    main()


# Two Ideas to fix:
# make  goal actually lie within reach of arm, but what about cost at each state?
# fix velocity by trying set max number of iters and just increasing by velocity and cap max reach
