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

State space: joint angles and bottle position (x,y)
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
    def __init__(self, cost, state):
        self.cost = cost
        self.state = state

    def __lt__(self, other):
        return self.cost < other.cost


class NaivePlanner():
    # discretize continuous state space
    dx = dy = dz = 0.1
    dpos = np.array([dx, dy, dz])
    qbins = 180.0 / 5.0  # same discretization as 5-degree increments

    def __init__(self, start, goal, env, xbounds, ybounds, dist_thresh=1e-1, eps=1):
        # state = [x,y,z,qx,qy,qz,qw,q1,q2...,q7]
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.env = env
        self.xbounds = np.array(xbounds)  # [minx, maxx]
        self.ybounds = np.array(ybounds)  # [miny, maxy]
        self.sq_dist_thresh = dist_thresh ** 2
        self.eps = eps

        # define action space
        self.num_joints = env.arm.num_joints
        self.A = ActionSpace(num_DOF=self.num_joints)
        self.G = dict()

        # discretize continuous state/action space
        self.da = self.A.da_rad
        self.xi_bounds = (
            (self.xbounds - self.xbounds[0]) / self.dx).astype(int)
        self.yi_bounds = (
            (self.ybounds - self.ybounds[0]) / self.dy).astype(int)
        # self.max_joint_indexes = (
        #     (self.env.arm.ul - self.env.arm.ll) / self.A.da_rad).astype(int)

    def debug_view_state(self, state):
        joint_pose = self.joint_pose_from_state(state)
        bx, by, _ = self.bottle_pos_from_state(state)
        self.env.arm.reset(joint_pose)
        link_states = p.getLinkStates(self.env.arm.kukaId, range(7))
        eex, eey = link_states[-1][4][:2]
        dist = ((bx - eex) ** 2 + (by - eey) ** 2) ** 0.5
        print(dist, bx, by)

    def plan(self):
        open_set = [Node(0, self.start)]
        closed_set = set()
        self.G = dict()
        self.G[self.state_to_key(self.start)] = 0
        transitions = dict()
        num_expansions = 0

        goal_expanded = False
        while not goal_expanded and len(open_set) > 0:
            num_expansions += 1
            n = heapq.heappop(open_set)
            state = n.state
            cur_joints = self.joint_pose_from_state(state)
            bottle_pos = self.bottle_pos_from_state(state)
            bottle_ori = self.bottle_ori_from_state(state)
            # print("Expanded: %.2f" % n.cost, self.heuristic(state))
            # self.debug_view_state(state)
            # print(self.heuristic(state))
            state_key = self.state_to_key(state)
            # duplicates are possible since heapq doesn't handle same state but diff costs
            if state_key in closed_set:
                continue
            assert(state_key in self.G)
            cur_cost = self.G[state_key]
            closed_set.add(state_key)

            if self.reached_goal(state):
                goal_expanded = True
                self.goal = state

            for ai in self.A.action_ids:
                dq = self.A.get_action(ai)
                trans_cost, next_bottle_pos, next_bottle_ori = self.env.run_sim(
                    action=dq, init_joints=cur_joints,
                    bottle_pos=bottle_pos, bottle_ori=bottle_ori)
                next_joint_pose = self.env.arm.joint_pose
                next_state = np.concatenate(
                    [next_bottle_pos, next_bottle_ori, next_joint_pose])
                next_state_key = self.state_to_key(next_state)

                if next_state_key in closed_set:
                    continue

                f = self.heuristic(next_state)
                new_G = cur_cost + trans_cost
                if next_state_key not in self.G or (
                        self.G[next_state_key] > new_G):
                    self.G[next_state_key] = new_G
                    overall_cost = new_G + self.eps * f
                    heapq.heappush(open_set, Node(overall_cost, next_state))
                    # print(overall_cost, f)
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

    def dist_arm_to_bottle(self, bottle_pos, joint_pose):
        bottle_pos = bottle_pos + self.env.bottle.center_of_mass
        self.env.arm.reset(joint_pose)
        link_positions = self.env.arm.get_link_positions()
        # min_sq_dist = None
        # for (lx, ly, lz) in link_positions:
        #     sq_dist = (bx - lx) ** 2 + (by - ly) ** 2
        #     if min_sq_dist is None or sq_dist < min_sq_dist:
        #         min_sq_dist = sq_dist
        # return math.sqrt(min_sq_dist)
        EE_pos = np.array(link_positions[-1])
        ee_link_dist = np.linalg.norm(bottle_pos[:2] - EE_pos[:2])
        return ee_link_dist

    def heuristic(self, state):
        bottle_pos = np.array(self.bottle_pos_from_state(state))
        goal_bottle_pos = np.array(self.bottle_pos_from_state(self.goal))
        dist_to_goal = np.linalg.norm(bottle_pos[:2] - goal_bottle_pos[:2])
        joints = self.joint_pose_from_state(state)
        dist_arm_to_bottle = self.dist_arm_to_bottle(bottle_pos, joints)
        return 5 * dist_to_goal + dist_arm_to_bottle

    def reached_goal(self, state):
        x, y, _ = self.bottle_pos_from_state(state)
        gx, gy, _ = self.bottle_pos_from_state(self.goal)
        dist_to_goal = (x - gx) ** 2 + (y - gy) ** 2
        return dist_to_goal < self.sq_dist_thresh

    def state_to_key(self, state):
        pos = np.array(self.bottle_pos_from_state(state))
        pos_i = (pos / self.dpos).astype(int)
        ori_i = self.quat_to_key(
            quat=self.bottle_ori_from_state(state),
            qbins=self.qbins)
        joints = self.joint_pose_from_state(state)
        joints_i = ((joints - self.env.arm.ul) / self.da).astype(int)
        # tuple of ints as unique id
        return (tuple(pos_i), tuple(ori_i), tuple(joints_i))

    def key_to_state(self, key):
        (pos_i, ori_i, joints_i) = key
        joints_i = np.array(joints_i)
        pos = np.array(pos_i) * self.dpos
        ori = self.key_to_quat(key=ori_i, qbins=self.qbins)
        joints = (joints_i * self.da) + self.env.arm.ul
        return np.concatenate([pos, ori, joints])
        # out_of_bounds = not (self.xi_bounds[0] <= xi <= self.xi_bounds[1] and
        #                      self.yi_bounds[0] <= yi <= self.yi_bounds[1])
        # if out_of_bounds:
        #     return None

    @staticmethod
    def quat_to_key(quat, qbins):
        return (np.array(quat) * qbins).astype(int)

    @staticmethod
    def key_to_quat(key, qbins):
        return np.array(key) / qbins

    @staticmethod
    def bottle_pos_from_state(state):
        return state[:3]

    @staticmethod
    def bottle_ori_from_state(state):
        return state[3:7]

    @staticmethod
    def joint_pose_from_state(state):
        return state[7:]


def test_quaternion_discretization():
    # generate random quaterions: http://planning.cs.uiuc.edu/node198.html
    avg_error = 0
    iters = 1000
    twopi = 2 * math.pi
    qbins = NaivePlanner.qbins
    for i in range(iters):
        # [0, 1) is essentially same as [0, 1] since p(x=1) = 0 for continuous
        # probability density function
        quat = np.array(list(Quaternion.random()))
        quat[3] = abs(quat[3])
        key = NaivePlanner.quat_to_key(list(quat), qbins)
        quat1 = np.array(
            list(Quaternion(NaivePlanner.key_to_quat(key, qbins))))
        # error = quat * quat1.inverse
        # print(quat, quat1)
        error = np.linalg.norm(quat - quat1)
        avg_error += error

    avg_error = avg_error / float(iters)
    print("Average error: %.3f" % avg_error)


def test_state_indexing():
    state = list(range(14))
    assert(NaivePlanner.bottle_pos_from_state(state) == [0, 1, 2])
    assert(NaivePlanner.bottle_ori_from_state(state) == [3, 4, 5, 6])
    assert(NaivePlanner.joint_pose_from_state(
        state) == [7, 8, 9, 10, 11, 12, 13])