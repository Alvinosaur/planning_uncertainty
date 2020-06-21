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
import heapq

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


class NaivePlanner():
    def __init__(self, start, goal, env, xbounds, ybounds, dist_thresh=1e-1, eps=2):
        # state = [x,y,q1,q2...,q7]
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

        # discretize continuous state space
        self.dx = self.dy = 0.1
        self.da = self.A.da_rad
        self.xi_bounds = (
            (self.xbounds - self.xbounds[0]) / self.dx).astype(int)
        self.yi_bounds = (
            (self.ybounds - self.ybounds[0]) / self.dy).astype(int)
        # self.max_joint_indexes = (
        #     (self.env.arm.ul - self.env.arm.ll) / self.A.da_rad).astype(int)

    def plan(self):
        open_set = [(0, self.start)]
        closed_set = set()
        self.G = dict()
        self.G[self.state_to_key(self.start)] = 0
        transitions = dict()

        goal_expanded = False
        while not goal_expanded and len(open_set) > 0:
            _, state = heapq.heappop(open_set)
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
                cur_joints = state[2:]

                # run deterministic simulation for now
                trans_cost, next_state = self.env.run_sim(
                    action=dq, init_joints=cur_joints)
                next_state_key = self.state_to_key(next_state)

                if next_state_key in closed_set:
                    continue

                f = self.heuristic(next_state)
                new_G = cur_cost + trans_cost
                if next_state_key not in self.G or (
                        self.G[next_state_key] > new_G):
                    self.G[next_state_key] = new_G
                    overall_cost = new_G + self.eps * f
                    heapq.heappush((overall_cost, next_state), open_set)
                    transitions[next_state_key] = (state_key, ai)

        if not goal_expanded:
            return [], []
        # reconstruct path
        policy = []
        planned_path = []
        state_key = self.state_to_key(self.goal)
        start_key = self.state_to_key(self.start)
        while state_key != start_key:
            planned_path.append(self.key_to_state(state_key))
            prev, ai = transitions[state_key]
            policy.append(self.A.get_action(ai))
            state_key = prev

        # need to reverse since backwards ordering
        planned_path.reverse()
        policy.reverese()
        return planned_path, policy

    def heuristic(self, state):
        x, y = state[0:2]
        gx, gy = self.goal[0:2]
        dist_to_goal = (x - gx) ** 2 + (y - gy) ** 2
        return dist_to_goal

    def reached_goal(self, state):
        x, y = state[0:2]
        gx, gy = self.goal[0:2]
        dist_to_goal = (x - gx) ** 2 + (y - gy) ** 2
        return dist_to_goal < self.sq_dist_thresh

    def state_to_key(self, state):
        x, y = state[0:2]
        joints = state[2:]
        joints_i = ((joints - self.env.arm.ul) / self.da).astype(int)
        xi = int((x - self.xbounds[0]) / self.dx)
        yi = int((y - self.ybounds[0]) / self.dy)
        return (xi, yi, joints_i)  # tuple of ints as unique id

    def key_to_state(self, key):
        (xi, yi, joints_i) = key
        x = (xi * self.dx) + self.xbounds[0]
        y = (yi * self.dy) + self.ybounds[0]
        joints = (joints_i * self.da) + self.env.arm.ul
        return np.array([x, y] + list(joints))
        # out_of_bounds = not (self.xi_bounds[0] <= xi <= self.xi_bounds[1] and
        #                      self.yi_bounds[0] <= yi <= self.yi_bounds[1])
        # if out_of_bounds:
        #     return None


def main():
    VISUALIZE = False
    LOGGING = False
    GRAVITY = -9.81
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

    # bottle
    bottle_start_pos = np.array([0.5, 0.4, 0.1]).astype(float)
    bottle_goal_pos = np.array([0.7, 0.3, 0.1]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)

    # starting end-effector pos, not base pos
    # NOTE: just temporarily setting arm to starting bottle position with some offset
    offset = -np.array([0.05, 0, 0])
    EE_start_pos = bottle_start_pos + offset
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    arm = Arm(EE_start_pos=EE_start_pos,
              start_ori=base_start_ori,
              kukaId=kukaId)
    start_joints = arm.get_target_joints()

    N = 500
    env = Environment(arm, bottle, is_viz=VISUALIZE, N=N)
    start = list(bottle_start_pos[0:2]) + list(start_joints)
    # goal joints are arbitrary and populated later in planner
    goal = list(bottle_goal_pos[0:2]) + [0]*arm.num_joints
    xbounds = [0.4, 0.9]
    ybounds = [0.1, 0.9]
    dist_thresh = 1e-1
    eps = 2
    planner = NaivePlanner(start, goal, env, xbounds,
                           ybounds, dist_thresh, eps)
    state_path, policy = planner.plan()


if __name__ == "__main__":
    main()
