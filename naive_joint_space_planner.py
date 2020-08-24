import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sn
import heapq
from pyquaternion import Quaternion

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace


"""
Given: some initial joint space configuration as well as start and end positions (x,y) of bottle

State space: [bx,by,q1,q2,....q7] where (bx,by) is bottle position
and q1-q7 are the 7 joint angles of arm
Notice state space composed only of steady-state, and bottle orientation
is stored as metadata in state->state transitions and not in the state

Action space: change in one joint angle at a time

Algorithm:
Simplify for now by having arm start in contact with bottle, specifically behind bottle.
start at normal start, then continuously iterate for all actions, find successors, pick next action based on heuristic: distance of bottle from goal


Pseudocode:
thresh = 1e-1
A = [set of all incremental joint angle changes]
transitions = dict()
start = [x,y,q1,q2,....q6]
goal = [x,y,q1,q2,....q6]
open_set = [(cost, state),...] as minHeap
goal_expanded = False  #  only true when goal bottle (x,y) expanded
while !goal_expanded
    (min_cost, state) = heap_pop(open_set)
    goal_expanded |= (euc_dist(d[0]-goal[0], state[1]-goal[1]) < thresh)
    if goal_expanded:
        goal = state  # change so goal joints aren't undefined
    for action in A:
        next_state, cost = env.sim(state, action)
        transitions[next_state] = (state, action)
        heappush((cost, next_state))

# reconstruct path:
policy = []
state = goal
while state != start:
    prev, action = transitions[state]
    policy.append(action)
    state = prev
"""


SINGLE = 0
AVG = 1
MODE = 2


class Node(object):
    def __init__(self, cost, state, bottle_ori=np.array([0, 0, 0, 1])):
        self.cost = cost
        self.state = state
        self.bottle_ori = bottle_ori

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        s = "C(%.2f): " % self.cost
        s += ",".join(["%.2f" % v for v in self.state])
        return s


class NaivePlanner():
    def __init__(self, env: Environment, xbounds, ybounds, dist_thresh=1e-1, eps=1, dx=0.1, dy=0.1, dz=0.1, da_rad=15 * math.pi / 180.0, use_3D=True, sim_mode=SINGLE, num_rand_samples=1):
        """[summary]

        Args:
            start ([type]): [description]
            goal ([type]): [description]
            env ([type]): [description]
            xbounds ([type]): [description]
            ybounds ([type]): [description]
            dist_thresh ([type], optional): [description]. Defaults to 1e-1.
            eps (int, optional): [description]. Defaults to 1.
            dx (float, optional): [description]. Defaults to 0.1.
            dy (float, optional): [description]. Defaults to 0.1.
            dz (float, optional): [description]. Defaults to 0.1.
            da_rad ([type], optional): [description]. Defaults to 15*math.pi/180.0.
            use_3D (bool, optional): whether to use 3D or 2D euclidean distance. Defaults to True.
        """
        # state = [x,y,z,q1,q2...,q7]
        self.env = env
        self.xbounds = np.array(xbounds)  # [minx, maxx]
        self.ybounds = np.array(ybounds)  # [miny, maxy]
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dpos = np.array([dx, dy, dz])

        # define action space
        self.da = da_rad
        self.num_joints = env.arm.num_joints
        self.A = ActionSpace(num_DOF=self.num_joints, da_rad=self.da)

        # store state g-values and all states(joint space) that are seen
        self.G = dict()

        # random samples of environmental parameters
        self.num_rand_samples = num_rand_samples
        self.sim_params_set = []  # populated below
        self.gen_new_env_params()

        # method of simulating an action
        self.sim_mode = sim_mode
        if sim_mode == SINGLE:
            self.sim_func = self.env.run_sim
            assert(self.num_rand_samples == 1)
        elif sim_mode == AVG:
            self.sim_func = self.env.run_sim_avg
            # 1 is ok, can change, but to double-check that we're not using default value
            assert(self.num_rand_samples > 1)
        elif sim_mode == MODE:
            self.sim_func = self.env.run_sim_mode
            assert(self.num_rand_samples > 1)
        else:
            print("Invalid sim mode specified: {}, defaulting to SINGLE".format(sim_mode))
            self.sim_func = self.env.run_sim
            assert(self.num_rand_samples == 1)

        # search parameters
        self.dist_thresh = dist_thresh
        self.eps = eps
        self.use_3D = use_3D
        self.NORMALIZER = 4 * da_rad * 180 / math.pi

        # discretize continuous state/action space
        self.xi_bounds = (
            (self.xbounds - self.xbounds[0]) / self.dx).astype(int)
        self.yi_bounds = (
            (self.ybounds - self.ybounds[0]) / self.dy).astype(int)
        self.TWO_PI_i = 2 * math.pi / self.da
        # let Arm() handle the actual manipulator limits, here just treat [0, 2pi] as bounds for discretization
        # self.joint_bounds = (self.env.arm.ul / self.da).astype(int)
        # self.max_joint_indexes = (
        #     (self.env.arm.ul - self.env.arm.ll) / self.A.da_rad).astype(int)

    def gen_new_env_params(self):
        self.sim_params_set = [self.env.gen_random_env_param()
                               for i in range(self.num_rand_samples)]

    def debug_view_state(self, state):
        joint_pose = self.joint_pose_from_state(state)
        bx, by, _ = self.bottle_pos_from_state(state)
        self.env.arm.reset(joint_pose)
        link_states = p.getLinkStates(self.env.arm.kukaId, range(7))
        eex, eey = link_states[-1][4][:2]
        dist = ((bx - eex) ** 2 + (by - eey) ** 2) ** 0.5
        # print("dist, bx, by: %.2f, (%.2f, %.2f)" % (dist, bx, by))

    def plan(self, start, goal):
        """Naive A* planner that replans from scratch

        Returns:
            [type]: [description]
        """
        # state = [x,y,z,q1,q2...,q7]
        self.start = np.array(start)
        self.goal = np.array(goal)
        # initialize open set with start and G values
        open_set = [Node(0, self.start)]
        closed_set = set()
        self.G = dict()
        self.G[self.state_to_key(self.start)] = 0
        transitions = dict()

        # metrics on performance of planner
        num_expansions = 0

        # find solution
        goal_expanded = False
        while not goal_expanded and len(open_set) > 0:
            num_expansions += 1

            # get next state to expand
            n = heapq.heappop(open_set)
            state = n.state
            state_key = self.state_to_key(state)
            bottle_ori = n.bottle_ori
            cur_joints = self.joint_pose_from_state(state)
            bottle_pos = self.bottle_pos_from_state(state)
            # print(n)
            # print("Heuristic: %.2f" % self.heuristic(n.state))
            # self.debug_view_state(state)

            # duplicates are possible since heapq doesn't handle same state but diff costs
            # print(state_key)
            # print(state[:3])
            if state_key in closed_set:
                # happens when duplicate states are entered with different f-vals are added to open-set
                # print("avoid re-expanding closed state: %s" % n)
                continue
            closed_set.add(state_key)

            print("Expanded: (%s) (%s)" %
                  (self.state_to_str(state[:3]), self.state_to_str(state[3:] * 180 / math.pi)))

            # check if found goal, if so loop will terminate in next iteration
            if self.reached_goal(state):
                goal_expanded = True
                self.goal = state

            # extra current total move-cost of current state
            assert(state_key in self.G)
            cur_cost = self.G[state_key]
            # dup_eps = self.calc_soft_eps(state)

            # explore all actions from this state
            for ai in self.A.action_ids:
                # action defined as an offset of joint angles of arm
                dq = self.A.get_action(ai)

                # (state, action) -> (cost, next_state)
                if self.sim_mode == SINGLE:
                    # only use one simulation parameter set
                    (trans_cost, next_bottle_pos,
                     next_bottle_ori, next_joint_pose) = self.sim_func(
                        action=dq, init_joints=cur_joints,
                        bottle_pos=bottle_pos, bottle_ori=bottle_ori,
                        sim_params=self.sim_params_set[0])
                else:
                    (trans_cost, next_bottle_pos,
                     next_bottle_ori, next_joint_pose) = self.sim_func(
                        action=dq, init_joints=cur_joints,
                        bottle_pos=bottle_pos, bottle_ori=bottle_ori,
                        sim_params_set=self.sim_params_set)

                # completely ignore actions that knock over bottle
                if self.is_invalid_transition(trans_cost):
                    continue

                # build next state and check if already expanded
                next_state = np.concatenate([next_bottle_pos, next_joint_pose])
                next_state_key = self.state_to_key(next_state)
                if next_state_key in closed_set:  # if already expanded, skip
                    continue

                f = self.heuristic(next_state)
                new_G = cur_cost + trans_cost
                # if state not expanded or found better path to next_state
                if next_state_key not in self.G or (
                        self.G[next_state_key] > new_G):
                    self.G[next_state_key] = new_G

                    overall_cost = new_G + self.eps * f
                    # print("Soft eps: %.2f, g: %.2f, f: %.2f, eps*f: %.2f, cost: %.2f, action: %s, next: %s" %
                    #       (dup_eps, new_G, f, self.eps * dup_eps * f, overall_cost,
                    #        self.state_to_str(dq * 180 / math.pi), self.state_to_str(next_state[3:] * 180 / math.pi)))
                    # print("Trans, heuristic change: %.3f, %.3f" % (
                    #     trans_cost, self.eps * (self.heuristic(state) - self.heuristic(next_state))))
                    # print("Overall new cost: %.2f" % overall_cost)
                    # print(next_state_key)

                    heapq.heappush(open_set, Node(
                        cost=overall_cost, state=next_state, bottle_ori=next_bottle_ori))

                    # build directed graph
                    transitions[next_state_key] = (state_key, ai)

        print("States Expanded: %d" % num_expansions)
        if not goal_expanded:
            return [], []
        # reconstruct path
        policy = []
        planned_path = []
        state_key = self.state_to_key(self.goal)
        start_key = self.state_to_key(self.start)
        # NOTE: planned_path does not include initial starting pose!
        while state_key != start_key:
            planned_path.append(self.key_to_state(state_key))
            prev, ai = transitions[state_key]
            policy.append(self.A.get_action(ai))
            state_key = prev

        # need to reverse since backwards ordering
        planned_path.reverse()
        policy.reverse()
        return planned_path, policy

    def dist_bottle_to_goal(self, state):
        bottle_pos = np.array(self.bottle_pos_from_state(state))
        goal_bottle_pos = np.array(self.bottle_pos_from_state(self.goal))
        return np.linalg.norm(bottle_pos[:2] - goal_bottle_pos[:2])

    def dist_arm_to_bottle(self, state, use_EE=False):
        """Calculates distance from bottle to arm in two forms:
        1. distance from end-effector(EE) to bottle
        2. shortest distance from any non-static joint or middle of link to bottle

        Distance can be either 2D or 3D.

        Args:
            bottle_pos (np.ndarray): 3 x 1 vector of [x,y,z]
            joint_pose (np.ndarray): (self.num_joints x 1) vec of joint angles
            use_EE (bool, optional): whether to use EE or shortest joint to bottle distance. Defaults to False.

        Returns:
            float: distance from arm to bottle
        """
        bottle_pos = self.bottle_pos_from_state(state)
        joint_pose = self.joint_pose_from_state(state)
        bottle_pos = bottle_pos + self.env.bottle.center_of_mass
        bx, by, bz = bottle_pos

        # set arm to specified joint pose to calculate joint distances
        self.env.arm.reset(joint_pose)
        joint_positions = self.env.arm.get_joint_positions()

        if use_EE:
            EE_pos = np.array(joint_positions[-1])
            if self.use_3D:
                return np.linalg.norm(bottle_pos[:3] - EE_pos[:3])
            else:
                return np.linalg.norm(bottle_pos[:2] - EE_pos[:2])
        else:
            midpoints = []
            # only calculate midpoint btwn last static and 1st dynamic
            for i in range(2, len(joint_positions) - 1):
                midpoint = np.mean(np.array([
                    joint_positions[i],
                    joint_positions[i + 1]]), axis=0)
                midpoints.append(midpoint)
            # ignore first two links, which are static
            positions = joint_positions[2:] + midpoints
            min_sq_dist = None
            min_i = 0
            for i, pos in enumerate(positions):
                if self.use_3D:
                    sq_dist = np.linalg.norm(np.array(pos) - bottle_pos[:3])
                else:
                    sq_dist = np.linalg.norm(
                        np.array(pos[:2]) - bottle_pos[:2])
                if min_sq_dist is None or sq_dist < min_sq_dist:
                    min_sq_dist = sq_dist
                    min_i = i
            return math.sqrt(min_sq_dist)

    def heuristic(self, state):
        dist_to_goal = self.dist_bottle_to_goal(state)

        dist_arm_to_bottle = self.dist_arm_to_bottle(
            state, use_EE=False)
        # print("BG: %.2f, EB: %.2f" % (dist_to_goal, dist_arm_to_bottle))
        return dist_to_goal + dist_arm_to_bottle

    def reached_goal(self, state):
        dist_to_goal = self.dist_bottle_to_goal(state)
        print("Pos: %.2f,%.2f" % tuple(self.bottle_pos_from_state(state)[:2]))
        print("%.2f ?< %.2f" % (dist_to_goal, self.dist_thresh))
        return dist_to_goal < self.dist_thresh

    def state_to_key(self, state):
        pos = np.array(self.bottle_pos_from_state(state))
        pos_i = np.rint(pos / self.dpos)
        joints = self.joint_pose_from_state(state)
        joints_i = np.rint(joints / self.da) % self.TWO_PI_i
        # tuple of ints as unique id
        return (tuple(pos_i), tuple(joints_i))

    def key_to_state(self, key):
        (pos_i, joints_i) = key
        joints_i = np.array(joints_i)
        pos = np.array(pos_i) * self.dpos
        joints = (joints_i * self.da) + self.env.arm.ul
        return np.concatenate([pos, joints])
        # out_of_bounds = not (self.xi_bounds[0] <= xi <= self.xi_bounds[1] and
        #                      self.yi_bounds[0] <= yi <= self.yi_bounds[1])
        # if out_of_bounds:
        #     return None

    @ staticmethod
    def state_to_str(state):
        s = ", ".join(["%.2f" % val for val in state])
        return s

    @ staticmethod
    def is_invalid_transition(trans_cost):
        return trans_cost == Environment.INF

    @ staticmethod
    def bottle_pos_from_state(state):
        return state[:3]

    @ staticmethod
    def joint_pose_from_state(state):
        return state[3:]


def test_state_indexing():
    state = list(range(10))
    assert(NaivePlanner.bottle_pos_from_state(state) == [0, 1, 2])
    assert(NaivePlanner.joint_pose_from_state(
        state) == [3, 4, 5, 6, 7, 8, 9])
