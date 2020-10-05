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
    def __init__(self, cost, state, nearest_arm_pos_i, nearest_arm_pos,
                 bottle_ori=np.array([0, 0, 0, 1]), g=0, h=0):
        self.cost = cost
        self.g = g
        self.h = h
        self.state = state
        self.bottle_ori = bottle_ori
        self.nearest_arm_pos_i = nearest_arm_pos_i
        self.nearest_arm_pos = nearest_arm_pos

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        s = "C(%.2f): " % self.cost
        s += ",".join(["%.2f" % v for v in self.state])
        return s


class NaivePlanner():

    def __init__(self, start, goal, env, xbounds, ybounds, dist_thresh=1e-1, eps=1, dx=0.1, dy=0.1, dz=0.1, da_rad=15 * math.pi / 180.0, visualize=False):
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

        # define action space
        self.da = da_rad
        self.num_joints = env.arm.num_joints
        self.A = ActionSpace(num_DOF=self.num_joints, da_rad=self.da)
        self.G = dict()
        self.use_EE = False
        self.guided_direction = True

        # discretize continuous state/action space
        self.xi_bounds = (
            (self.xbounds - self.xbounds[0]) / self.dx).astype(int)
        self.yi_bounds = (
            (self.ybounds - self.ybounds[0]) / self.dy).astype(int)

        self.visualize = visualize

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
        arm_positions = self.env.arm.get_joint_link_positions(
            self.joint_pose_from_state(self.start))
        _, nn_joint_i, nn_joint_pos = (
            self.dist_arm_to_bottle(self.start, arm_positions))
        open_set = [
            Node(0, self.start,
                 nearest_arm_pos_i=nn_joint_i,
                 nearest_arm_pos=nn_joint_pos)]
        closed_set = set()
        self.G = dict()
        self.G[self.state_to_key(self.start)] = 0
        transitions = dict()

        if self.visualize:
            # visualize a vertical blue line representing goal pos of bottle
            # just to make line vertical
            vertical_offset = np.array([0, 0, 0.5])
            goal_bpos = self.bottle_pos_from_state(self.goal)
            Environment.draw_line(lineFrom=goal_bpos,
                                  lineTo=goal_bpos + vertical_offset,
                                  lineColorRGB=[0, 0, 1], lineWidth=1,
                                  lifeTime=0)

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
            guided_bottle_pos = self.get_guided_bottle_pos(bottle_pos)
            Environment.draw_line(lineFrom=guided_bottle_pos,
                                  lineTo=guided_bottle_pos +
                                  np.array([0, 0, 1]),
                                  lineColorRGB=[1, 0, 0], lineWidth=1,
                                  lifeTime=5)
            Environment.draw_line(lineFrom=bottle_pos,
                                  lineTo=bottle_pos + np.array([0.5, 0, 0]),
                                  lineColorRGB=[0, 1, 0], lineWidth=1,
                                  lifeTime=0)
            print(n)
            if state_key in closed_set:
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
                action = self.A.get_action(ai)

                # (state, action) -> (cost, next_state)
                (is_fallen, is_collision, next_bottle_pos,
                 next_bottle_ori, next_joint_pose) = self.env.run_sim(
                    action=action, init_joints=cur_joints,
                    bottle_pos=bottle_pos, bottle_ori=bottle_ori)

                new_arm_positions = self.env.arm.get_joint_link_positions(
                    next_joint_pose)
                trans_cost = self.calc_trans_cost(n, new_arm_positions)

                # completely ignore actions that knock over bottle
                if is_fallen:
                    continue

                # build next state and check if already expanded
                next_state = np.concatenate([next_bottle_pos, next_joint_pose])
                next_state_key = self.state_to_key(next_state)
                if next_state_key in closed_set:  # if already expanded, skip
                    continue

                arm_bottle_dist, nn_joint_i, nn_joint_pos = (
                    self.dist_arm_to_bottle(next_state, new_arm_positions))
                h = self.heuristic(next_state, arm_bottle_dist)
                new_G = cur_cost + trans_cost
                # if state not expanded or found better path to next_state
                if next_state_key not in self.G or (
                        self.G[next_state_key] > new_G):
                    self.G[next_state_key] = new_G
                    overall_cost = new_G + self.eps * h

                    # del_h = self.heuristic(next_state, arm_bottle_dist) - n.h
                    # print("del_g, del_h, eps*del_h: %.3f, %.3f, %.3f" % (
                    #     trans_cost,
                    #     del_h,
                    #     self.eps * del_h))

                    # add to open set
                    heapq.heappush(open_set, Node(
                        cost=overall_cost,
                        g=new_G, h=h,
                        state=next_state,
                        nearest_arm_pos_i=nn_joint_i,
                        nearest_arm_pos=nn_joint_pos,
                        bottle_ori=next_bottle_ori))

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

    def get_guided_bottle_pos(self, bpos, dist_offset=0.1):
        new_bpos = np.copy(bpos)
        goal_pos = self.bottle_pos_from_state(self.goal)
        vec_cur_to_goal = (goal_pos[:2] - new_bpos[:2])
        vec_cur_to_goal /= np.linalg.norm(vec_cur_to_goal)
        # [dx, dy] offset opposite of direction from bottle to goal
        new_bpos[:2] -= dist_offset * vec_cur_to_goal
        return new_bpos

    def dist_arm_to_bottle(self, state, positions):
        bottle_pos = self.bottle_pos_from_state(state)
        bottle_pos = bottle_pos + self.env.bottle.center_of_mass

        # fake bottle position to be behind bottle in the direction towards the goal
        if self.guided_direction:
            bottle_pos = self.get_guided_bottle_pos(bottle_pos)

        min_dist = None
        min_i = 0
        min_pos = None
        for i, pos in enumerate(positions):
            dist = np.linalg.norm(np.array(pos) - bottle_pos)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_i = i
                min_pos = pos
        return min_dist, min_i, min_pos

    def calc_trans_cost(self, n: Node, new_arm_positions):
        prev_nearest_ji = n.nearest_arm_pos_i
        prev_nearest_j = np.array(n.nearest_arm_pos)
        new_nearest_j = np.array(new_arm_positions[prev_nearest_ji])
        dist_traveled = np.linalg.norm(prev_nearest_j - new_nearest_j)
        return dist_traveled

    def heuristic(self, state, dist_arm_to_bottle):
        bottle_pos = np.array(self.bottle_pos_from_state(state))
        goal_bottle_pos = np.array(self.bottle_pos_from_state(self.goal))
        dist_to_goal = np.linalg.norm(bottle_pos[:2] - goal_bottle_pos[:2])
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
        joints_i = np.rint((joints - self.env.arm.ul) / self.da)
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

    @ staticmethod
    def state_to_str(state):
        s = ", ".join(["%.2f" % val for val in state])
        return s


def test_state_indexing():
    state = list(range(10))
    assert(NaivePlanner.bottle_pos_from_state(state) == [0, 1, 2])
    assert(NaivePlanner.joint_pose_from_state(
        state) == [3, 4, 5, 6, 7, 8, 9])
