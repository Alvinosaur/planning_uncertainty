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
    def __init__(self, start, goal, env, xbounds, ybounds, dist_thresh=1e-1, eps=1, dx=0.1, dy=0.1, dz=0.1, da_rad=15 * math.pi / 180.0, use_3D=True):
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
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.env = env
        self.xbounds = np.array(xbounds)  # [minx, maxx]
        self.ybounds = np.array(ybounds)  # [miny, maxy]
        self.sq_dist_thresh = dist_thresh ** 2
        self.eps = eps
        self.dx, self.dy, self.dz = dx, dy, dz
        # discretize continuous state space
        self.dpos = np.array([dx, dy, dz])
        self.use_3D = use_3D

        # define action space
        self.da = da_rad
        self.num_joints = env.arm.num_joints
        self.A = ActionSpace(num_DOF=self.num_joints, da_rad=self.da)
        self.G = dict()

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

    def debug_view_state(self, state):
        joint_pose = self.joint_pose_from_state(state)
        bx, by, _ = self.bottle_pos_from_state(state)
        self.env.arm.reset(joint_pose)
        link_states = p.getLinkStates(self.env.arm.kukaId, range(7))
        eex, eey = link_states[-1][4][:2]
        dist = ((bx - eex) ** 2 + (by - eey) ** 2) ** 0.5
        print("dist, bx, by: %.2f, (%.2f, %.2f)" % (dist, bx, by))

    def plan(self):
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
            print("Expanded:")
            print(n)
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

            # check if found goal, if so loop will terminate in next iteration
            if self.reached_goal(state):
                goal_expanded = True
                self.goal = state

            # extra current total move-cost of current state
            assert(state_key in self.G)
            cur_cost = self.G[state_key]

            # explore all actions from this state
            for ai in self.A.action_ids:
                # action defined as an offset of joint angles of arm
                dq = self.A.get_action(ai)

                # (state, action) -> (cost, next_state)
                (trans_cost, next_bottle_pos,
                 next_bottle_ori, next_joint_pose) = self.env.run_sim(
                    action=dq, init_joints=cur_joints,
                    bottle_pos=bottle_pos, bottle_ori=bottle_ori)

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
                    # print("Trans, heuristic change: %.3f, %.3f" % (
                    #     trans_cost, self.eps * (self.heuristic(state) - self.heuristic(next_state))))
                    # print("Overall new cost: %.2f" % overall_cost)
                    # print(next_state_key)

                    # add to open set
                    heapq.heappush(open_set, Node(
                        cost=overall_cost, state=next_state, bottle_ori=next_bottle_ori))
                    # print(overall_cost, f)

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

    def dist_arm_to_bottle(self, bottle_pos, joint_pose,
                           use_EE=False):
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
                # color = [1, 0, 0] if i < len(joint_positions[2:]) else [0, 1, 0]
                # debug_pos = np.array([lx, ly, lz])
                # Environment.draw_line(
                #     lineFrom=debug_pos,
                #     lineTo=debug_pos + np.array([0, 0, 1]),
                #     lineColorRGB=color,
                #     lineWidth=10, lifeTime=self.env.max_iters/float(self.env.SIM_VIZ_FREQ))
                if self.use_3D:
                    sq_dist = np.linalg.norm(np.array(pos) - bottle_pos[:3])
                else:
                    sq_dist = np.linalg.norm(
                        np.array(pos[:2]) - bottle_pos[:2])
                if min_sq_dist is None or sq_dist < min_sq_dist:
                    min_sq_dist = sq_dist
                    min_i = i

            # debug
            # if min_i >= len(joint_positions[:2]):
            #     print("closest is midpoint %d, %.2f" %
            #           (min_i - len(joint_positions[:2]), min_sq_dist))
            # else:
            #     print("closest is link %d, %.2f" % (min_i, min_sq_dist))
            # debug_pos = np.array(positions[min_i])
            # Environment.draw_line(
            #     lineFrom=debug_pos,
            #     lineTo=debug_pos + np.array([0, 0, 1]),
            #     lineColorRGB=[1, 0, 0],
            #     lineWidth=10, lifeTime=self.env.max_iters/float(self.env.SIM_VIZ_FREQ))
            return math.sqrt(min_sq_dist)

        # print(math.sqrt(min_sq_dist))
        # print("(%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f)" % (
        #     joint_positions[0][0], joint_positions[0][1], joint_positions[0][2], joint_positions[1][0], joint_positions[1][1], joint_positions[1][2]))

    def heuristic(self, state):
        bottle_pos = np.array(self.bottle_pos_from_state(state))
        goal_bottle_pos = np.array(self.bottle_pos_from_state(self.goal))
        # exception is to not use 3D for bottle position since only on flat environment
        dist_to_goal = np.linalg.norm(bottle_pos[: 2] - goal_bottle_pos[: 2])
        joints = self.joint_pose_from_state(state)
        dist_arm_to_bottle = self.dist_arm_to_bottle(
            bottle_pos, joints, use_EE=False)
        # print("BG: %.2f, EB: %.2f" % (dist_to_goal, dist_arm_to_bottle))
        return dist_to_goal + dist_arm_to_bottle

    def reached_goal(self, state):
        x, y, _ = self.bottle_pos_from_state(state)
        gx, gy, _ = self.bottle_pos_from_state(self.goal)
        dist_to_goal = (x - gx) ** 2 + (y - gy) ** 2
        return dist_to_goal < self.sq_dist_thresh

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
