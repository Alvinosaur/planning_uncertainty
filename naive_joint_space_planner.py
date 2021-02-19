import pybullet as p
import time
import math
import numpy as np

import heapq
from pyquaternion import Quaternion

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace, StateTuple


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

DEG2RAD = math.pi / 180.0
RAD2DEG = 1 / DEG2RAD
SINGLE = "single"
AVG = "avg"
ALWAYS_N = 0
ALWAYS_1 = 1
FAR_N = 2
CLOSE_N = 3
SIM_TYPE_TO_ID = {
    "always_N": ALWAYS_N,
    "always_1": ALWAYS_1,
    "far_N": FAR_N,
    "close_N": CLOSE_N
}


class Node(object):
    def __init__(self, cost, state, nearest_arm_pos_i,
                 nearest_arm_pos, ee_pos, fall_history=[], mode_sim_param=None, fall_prob=-1,
                 bottle_ori=np.array([0, 0, 0, 1]), g=0, h=0, bottle_goal_dist=np.inf, arm_bottle_dist=np.inf):
        self.cost = cost
        self.fall_prob = fall_prob
        self.fall_history = fall_history
        self.mode_sim_param = mode_sim_param
        self.g = g
        self.h = h
        self.bottle_goal_dist = bottle_goal_dist
        self.arm_bottle_dist = arm_bottle_dist
        self.state = state
        self.bottle_ori = bottle_ori
        self.nearest_arm_pos_i = nearest_arm_pos_i
        self.nearest_arm_pos = nearest_arm_pos
        self.ee_pos = ee_pos

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        s = "C(%.2f) | h(%.3f) | d(%.3f) | " % (self.cost, self.h, self.bottle_goal_dist)
        s += "Bpos(" + ",".join(["%.2f" % v for v in self.state[:3]]) + "), "
        s += "Bang(" + "%.1f" % (Bottle.calc_vert_angle(self.bottle_ori) * RAD2DEG) + "), "
        s += "Joints(" + ",".join(["%.3f" % v for v in self.state[3:]]) + "), "
        # s += "), fall history:"
        # s += ",".join(["%d" % v for v in self.fall_history])
        s += " | %s" % self.mode_sim_param
        return s


class NaivePlanner(object):
    def __init__(self, env, sim_mode, sim_params_set, dist_thresh=1e-1, eps=1, dx=0.1, dy=0.1, dz=0.1,
                 da_rad=15 * DEG2RAD, visualize=False, start=None, goal=None,
                 fall_thresh=0.2, use_ee_trans_cost=True, simulate_prev_trans=False, sim_type="always_N",
                 sim_dist_thresh=0.18, single_param=None):
        # state = [x,y,z,q1,q2...,q7]
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.env = env
        self.dist_thresh = dist_thresh
        self.sq_dist_thresh = dist_thresh ** 2
        self.eps = eps
        self.dx, self.dy, self.dz = dx, dy, dz
        # discretize continuous state space
        self.dpos = np.array([dx, dy, dz])

        # Costs and weights
        self.dist_cost_weights = np.array([3, 3, 2])  # x, y, z
        self.use_ee_trans_cost = use_ee_trans_cost
        self.pos_var_w = 30
        self.ang_var_w = 30
        self.move_cost_w = 0.01
        # self.time_cost_weight = 0.025  # chosen since avg EE dist moved cost is 0.05
        # and time_cost_weight at most will be scaled by 2

        # define action space
        self.da = da_rad
        self.num_joints = env.arm.num_joints
        self.A = ActionSpace(num_dof=self.num_joints, da_rad=self.da)
        self.G = dict()
        self.use_EE = False
        self.guided_direction = True

        # choose to simulate not only (s_t,a_t) but also (s_t-1, a_t-1)
        # only useful for AVG planner which uses mode next state as the successor
        # and this mode may not accurately reflect next state of some simulations
        if simulate_prev_trans and sim_mode == SINGLE:
            print("Trying to use simulate_prev_trans=True with SINGLE planner is useless, ignoring...")
        self.simulate_prev_trans = simulate_prev_trans and (sim_mode != SINGLE)

        self.visualize = visualize

        # random samples of environmental parameters
        self.sim_params_set = sim_params_set
        self.single_param = single_param  # when SINGLE mode or when running 1 sim in AVG mode

        # overall probability of a given simulation is product of probabilities
        # of random env params
        self.param_probs = [(p.bottle_fill_prob * p.bottle_fric_prob)
                            for p in self.sim_params_set]

        # safety threshold for prob(bottle fall) to deem invalid transitions
        self.fall_thresh = fall_thresh

        # Optimizations for deciding when to run N vs 1 simulation
        assert sim_type in SIM_TYPE_TO_ID, "{} not a valid sim_type!".format(sim_type)
        self.sim_type = SIM_TYPE_TO_ID[sim_type]
        self.sim_dist_thresh = sim_dist_thresh

        # method of simulating an action
        self.sim_mode = sim_mode
        if sim_mode == SINGLE:
            self.sim_func = self.env.run_sim
        elif sim_mode == AVG:
            self.sim_func = self.env.run_multiple_sims
            self.process_multiple_sim_results = self.avg_results
        else:
            print("Invalid sim mode specified: {}, defaulting to SINGLE".format(sim_mode))
            self.sim_func = self.env.run_sim

    def debug_view_state(self, state):
        joint_pose = self.joint_pose_from_state(state)
        bx, by, _ = self.bottle_pos_from_state(state)
        self.env.arm.reset(joint_pose)
        link_states = p.getLinkStates(self.env.arm.kukaId, range(7))
        eex, eey = link_states[-1][4][:2]
        dist = ((bx - eex) ** 2 + (by - eey) ** 2) ** 0.5
        print("dist, bx, by: %.2f, (%.2f, %.2f)" % (dist, bx, by))

    def avg_results(self, results, sim_params_set):
        """
        Randomly sample internal(unobservable to agent) bottle parameters for
        each iteration to calculate expected fall rate.
        Only the first sim param (which should be same as the single planner)
         will be used to determine cost and next state.
        NOTE: this is still not stochastic because we overall are using some finalized cost and next state to represent successor of a (state, action) pair. In a true stochastic planner, successors are represented
        by a belief space, or distribution of possible next states, each with
        some probability.
        """

        # calculate proportion and see if exceeds probability threshold
        avg_fall_prob = 0
        fall_prob_norm = 0

        # calculate variance in next state
        all_bpos = []
        all_bang = []

        # draw random param from set to determine next state and reward
        bpos_bins = dict()  # state_key -> [state, bori, count]
        sim_params_bins = dict()

        # NOTE: simulation automatically terminates if the arm doesn't touch
        # bottle since no need to simulate different bottle parameters
        # so num_iters <= self.num_rand_samples
        num_iters = len(results)
        fall_history = []
        for i in range(num_iters):
            is_fallen, is_collision, bpos, bori, next_joint_pos, z_rot_ang = results[i]
            sim_param = sim_params_set[i]

            bpos_disc = np.rint(bpos / self.dpos)
            rot_angle_disc = round(z_rot_ang / self.da)
            key = (tuple(bpos_disc), rot_angle_disc)

            if key in bpos_bins:
                bpos_bins[key][-1] += 1
                sim_params_bins[key].append(sim_param)
            else:
                bpos_bins[key] = [bpos, bori, next_joint_pos, 1]
                sim_params_bins[key] = [sim_param]

            # weighted sum of fall counts: sum(pi*xi) / sum(pi)
            fall_history.append(is_fallen)
            avg_fall_prob += is_fallen  # self.param_probs[i]
            fall_prob_norm += 1
            # fall_prob_norm += self.param_probs[i]

            all_bpos.append(bpos)
            all_bang.append(z_rot_ang)

        # normalize by sum of weights (not all but only up to num_iters)
        fall_proportion = avg_fall_prob / fall_prob_norm

        # find mode of next state
        max_key = max(bpos_bins, key=lambda k: bpos_bins[k][-1])
        [bpos, bori, next_joint_pos, count] = bpos_bins[max_key]
        # mode_sim_param = sum(sim_params_bins[max_key]) / len(sim_params_bins[max_key])
        mode_sim_param = ['%.3f' % param.bottle_fric for param in sim_params_bins[max_key]]
        # print(bpos_bins[max_key][-1])
        # print(self.state_to_str(bpos), self.state_to_str(bori))

        # calculate variance
        pos_variance = np.sum(np.var(all_bpos, axis=0))  # sum of variance of each dimension (x, y, z)
        z_ang_variance = np.var(all_bang)
        z_ang_mean = np.mean(all_bang)

        return fall_proportion, pos_variance, z_ang_variance, z_ang_mean, fall_history, bpos, bori, next_joint_pos, mode_sim_param

    def plan(self, bottle_ori=np.array([0, 0, 0, 1])):
        # initialize open set with start and G values
        try:
            arm_positions = self.env.arm.get_joint_link_positions(
            self.joint_pose_from_state(self.start))
        except:
            self.env.reset()
            arm_positions = self.env.arm.get_joint_link_positions(
            self.joint_pose_from_state(self.start))
        _, nn_joint_i, nn_joint_pos = (
            self.dist_arm_to_bottle(self.start, arm_positions))
        open_set = [
            Node(0, self.start,
                 nearest_arm_pos_i=nn_joint_i,
                 nearest_arm_pos=nn_joint_pos,
                 bottle_ori=bottle_ori,
                 ee_pos=arm_positions[-1])]
        closed_set = set()
        self.G = dict()
        self.G[self.state_to_key(self.start)] = 0
        transitions = dict()

        if self.visualize:
            # visualize a vertical blue line representing goal pos of bottle
            # just to make line vertical
            vertical_offset = np.array([0, 0, 0.5])
            goal_bpos = self.bottle_pos_from_state(self.goal)
            self.env.goal_line_id = self.env.draw_line(
                lineFrom=goal_bpos,
                lineTo=goal_bpos + vertical_offset,
                lineColorRGB=[0, 0, 1], lineWidth=1,
                replaceItemUniqueId=self.env.goal_line_id,
                lifeTime=0)

        # metrics on performance of planner
        num_expansions = 0

        # find solution
        goal_expanded = False

        while not goal_expanded and len(open_set) > 0:
            # start_time = time.time()
            num_expansions += 1

            # get next state to expand
            n = heapq.heappop(open_set)
            state = n.state
            state_key = self.state_to_key(state)

            if self.simulate_prev_trans and num_expansions > 1:
                prev_ai, prev_node = transitions[state_key]
                prev_action = self.A.get_action(prev_ai)
                prev_joints = self.joint_pose_from_state(prev_node.state)
                prev_bottle_pos = self.bottle_pos_from_state(prev_node.state)
                prev_bottle_ori = prev_node.bottle_ori
                prev_state_tuple = StateTuple(bottle_pos=prev_bottle_pos,
                                              bottle_ori=prev_bottle_ori,
                                              joints=prev_joints)
            else:
                prev_state_tuple = None
                prev_action = None

            bottle_ori = n.bottle_ori
            cur_joints = self.joint_pose_from_state(state)
            bottle_pos = self.bottle_pos_from_state(state)
            cur_state_tuple = StateTuple(bottle_pos=bottle_pos, bottle_ori=bottle_ori, joints=cur_joints)
            guided_bottle_pos = self.get_guided_bottle_pos(bottle_pos)

            print(n, flush=True)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            # check if found goal, if so loop will terminate in next iteration
            if self.reached_goal_node(n):
                goal_expanded = True
                self.goal = state

            # extra current total move-cost of current state
            assert(state_key in self.G)
            cur_cost = self.G[state_key]

            # explore all actions from this state
            # for ai in self.A.action_ids:
            for ai in self.A.action_ids:
                if self.visualize:
                    vertical_offset = np.array([0, 0, 0.5])
                    goal_bpos = self.bottle_pos_from_state(self.goal)
                    self.env.goal_line_id = self.env.draw_line(
                        lineFrom=goal_bpos,
                        lineTo=goal_bpos + vertical_offset,
                        lineColorRGB=[0, 0, 1], lineWidth=1,
                        # replaceItemUniqueId=self.env.goal_line_id,
                        lifeTime=0)

                # action defined as an offset of joint angles of arm
                action = self.A.get_action(ai)
                # print(np.array2string(action[0], precision=2))

                # (state, action) -> (cost, next_state)
                if self.sim_mode == SINGLE:
                    # only use one simulation parameter set
                    (is_fallen, _, next_bottle_pos,
                     next_bottle_ori, next_joint_pose) = self.sim_func(
                        action=action, state=cur_state_tuple,
                        prev_state=prev_state_tuple, prev_action=prev_action,
                        sim_params=self.single_param)

                    mode_sim_param = self.single_param
                    invalid = is_fallen
                    fall_history = [is_fallen]
                    fall_prob = 1 if is_fallen else 0
                    pos_variance = 0
                    z_ang_variance = 0

                else:
                    if (self.sim_type == ALWAYS_N or
                            (self.sim_type == FAR_N and n.bottle_goal_dist > self.sim_dist_thresh) or
                            (self.sim_type == CLOSE_N and n.bottle_goal_dist <= self.sim_dist_thresh)):
                        sim_params_set = self.sim_params_set
                    else:
                        sim_params_set = self.single_param

                    results = self.sim_func(
                        action=action, state=cur_state_tuple,
                        prev_state=prev_state_tuple, prev_action=prev_action,
                        sim_params_set=sim_params_set)

                    (fall_prob, pos_variance, z_ang_variance, z_ang_mean, fall_history, next_bottle_pos,
                     next_bottle_ori, next_joint_pose, mode_sim_param) = (
                        self.process_multiple_sim_results(results, sim_params_set))
                    invalid = fall_prob > self.fall_thresh
                    # print(invalid, fall_prob)

                # print("Is fallen: %d, action: %s" % (invalid, action))
                next_state = np.concatenate([next_bottle_pos, next_joint_pose])
                next_state_key = self.state_to_key(next_state)

                # completely ignore actions that knock over bottle with high
                # probability
                if invalid:
                    continue

                new_arm_positions = self.env.arm.get_joint_link_positions(
                    next_joint_pose)

                move_cost = self.calc_move_cost(n, new_arm_positions)
                trans_cost = self.calc_trans_cost(move_cost=move_cost,
                                                  fall_prob=fall_prob,
                                                  pos_variance=pos_variance,
                                                  z_ang_variance=z_ang_variance)
                trans_cost += z_ang_mean
                # self.time_cost_weight * self.A.get_action_time_cost(action))

                # build next state and check if already expanded
                next_state = np.concatenate([next_bottle_pos, next_joint_pose])
                next_state_key = self.state_to_key(next_state)
                if next_state_key in closed_set:  # if already expanded, skip
                    continue

                arm_bottle_dist, nn_joint_i, nn_joint_pos = (
                    self.dist_arm_to_bottle(next_state, new_arm_positions))

                # Quick FIX: use EE for transition costs so set nn_joint to EE
                # still true b/c midpoints + joints, so last item is still last joint (EE)
                bottle_goal_dist, arm_goal_dist = self.dist_to_goal(next_state, nn_joint_pos)
                print("     ai[%d] bottle-goal, arm-bottle: %.3f, %.3f" % (
                    ai,
                    bottle_goal_dist - n.bottle_goal_dist,
                    arm_bottle_dist - n.arm_bottle_dist))
                h = self.heuristic(bottle_goal_dist, arm_bottle_dist, arm_goal_dist)
                print("     ai[%d] delta h: %.3f, costs: %.3f, %.3f, %.3f, %.3f" % (
                    ai, self.eps * (h - n.h), self.move_cost_w * move_cost,
                    self.pos_var_w * pos_variance, self.ang_var_w * z_ang_variance,
                    z_ang_mean))
                new_G = cur_cost + trans_cost

                # if state not expanded or found better path to next_state
                if next_state_key not in self.G or (
                        self.G[next_state_key] > new_G):
                    self.G[next_state_key] = new_G
                    overall_cost = new_G + self.eps * h

                    # add to open set
                    new_node = Node(
                        cost=overall_cost,
                        fall_prob=fall_prob,
                        fall_history=fall_history,
                        mode_sim_param=mode_sim_param,
                        g=new_G, h=h, bottle_goal_dist=bottle_goal_dist,
                        arm_bottle_dist=arm_bottle_dist,
                        state=next_state,
                        nearest_arm_pos_i=nn_joint_i,
                        nearest_arm_pos=nn_joint_pos,
                        bottle_ori=next_bottle_ori,
                        ee_pos=new_arm_positions[-1])
                    print("     ai[%d] -> %s" % (ai, new_node))
                    print("     ", np.array2string(np.array(next_bottle_ori), precision=2),
                          "%.2f" % Bottle.calc_vert_angle(next_bottle_ori))
                    heapq.heappush(open_set, new_node)

                    # build directed graph
                    transitions[next_state_key] = (ai, n)
                    # print("%s -> %s" % (self.state_to_str(state), self.state_to_str(next_state)))

            # print("time: %.3f" % (time.time() - start_time))

        print("States Expanded: %d, found goal: %d" %
              (num_expansions, goal_expanded))
        if not goal_expanded:
            return [], []
        # reconstruct path
        policy = []
        planned_path = []
        node_path = []
        state = self.goal
        state_key = self.state_to_key(state)
        start_key = self.state_to_key(self.start)
        # NOTE: planned_path does not include initial starting pose!
        while state_key != start_key:
            # state[i]  ex: goal
            planned_path.append(state)
            # state[i-1] + action[i-1] -> node[i] containing state[i]
            ai, n = transitions[state_key]
            policy.append(self.A.get_action(ai))
            state_key = self.state_to_key(n.state)
            node_path.append(n)

        # need to reverse since backwards ordering
        # (node[i], policy[i]) -> state[i]
        # or in other words, ith policy led to ith state
        # so after taking action i, we should reach state i
        planned_path.reverse()
        policy.reverse()
        node_path.reverse()
        return planned_path, policy, node_path

    def get_guided_bottle_pos(self, bpos, dist_offset=0.1):
        new_bpos = np.copy(bpos)
        goal_pos = self.bottle_pos_from_state(self.goal)
        vec_cur_to_goal = (goal_pos[:2] - new_bpos[:2])
        vec_cur_to_goal /= np.linalg.norm(vec_cur_to_goal)
        # [dx, dy] offset opposite of direction from bottle to goal
        new_bpos[:2] -= dist_offset * vec_cur_to_goal
        return new_bpos

    def dist_to_goal(self, state, nn_joint_pos):
        bottle_pos = np.array(self.bottle_pos_from_state(state))
        goal_bottle_pos = np.array(self.bottle_pos_from_state(self.goal))
        dist_bottle_to_goal = np.linalg.norm(bottle_pos[:2] - goal_bottle_pos[:2])
        dist_arm_to_goal = np.linalg.norm(nn_joint_pos[:2] - goal_bottle_pos[:2])
        return dist_bottle_to_goal, dist_arm_to_goal

    def dist_arm_to_bottle(self, state, positions):
        bottle_pos = self.bottle_pos_from_state(state)

        # fake bottle position to be behind bottle in the direction towards the goal
        if self.guided_direction:
            bottle_pos = self.get_guided_bottle_pos(
                bottle_pos, dist_offset=0.1)
        if self.visualize:
            vertical_offset = np.array([0, 0, 0.5])
            self.env.target_line_id = self.env.draw_line(
                lineFrom=bottle_pos,
                lineTo=bottle_pos + vertical_offset,
                lineColorRGB=[1, 0, 0], lineWidth=1,
                replaceItemUniqueId=self.env.target_line_id,
                lifeTime=0)

        min_dist = None
        min_i = 0
        min_pos = None
        for i, pos in enumerate(positions):
            diff = self.dist_cost_weights * (np.array(pos) - bottle_pos)
            dist = np.linalg.norm(diff)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_i = i
                min_pos = pos

        return min_dist, min_i, min_pos

    def calc_trans_cost(self, move_cost, fall_prob, pos_variance, z_ang_variance):
        # z_ang_variance = 70 * (z_ang_variance / self.max_ang_var)
        # pos_variance = 20 * (pos_variance / self.max_pos_var)
        # fall_prob = 10 * fall_prob
        # move_cost = 0.5 * (move_cost / self.max_move_dist)
        return self.move_cost_w * move_cost + fall_prob + self.pos_var_w * pos_variance + self.ang_var_w * z_ang_variance

    def calc_move_cost(self, n: Node, new_arm_positions):
        if self.use_ee_trans_cost:
            dist_traveled = np.linalg.norm(
                np.array(n.ee_pos) - np.array(new_arm_positions[-1]))

        else:
            prev_nearest_ji = n.nearest_arm_pos_i
            prev_nearest_j = np.array(n.nearest_arm_pos)
            new_nearest_j = np.array(new_arm_positions[prev_nearest_ji])
            dist_traveled = np.linalg.norm(prev_nearest_j - new_nearest_j)
        return dist_traveled

    def heuristic(self, dist_to_goal, dist_arm_to_bottle, arm_goal_dist):
        # print(dist_to_goal, 0.6 * dist_arm_to_bottle, 0.5 * arm_goal_dist)
        return dist_to_goal + 0.25 * dist_arm_to_bottle  # + 0.8 * arm_goal_dist

    def reached_goal_node(self, node: Node):
        return node.bottle_goal_dist < self.dist_thresh

    def reached_goal(self, state):
        x, y, _ = self.bottle_pos_from_state(state)
        gx, gy, _ = self.bottle_pos_from_state(self.goal)
        dist_to_goal = (x - gx) ** 2 + (y - gy) ** 2
        return dist_to_goal < self.sq_dist_thresh

    def state_to_key(self, state):
        pos = np.array(self.bottle_pos_from_state(state))
        pos_i = np.rint(pos / self.dpos)
        joints = self.joint_pose_from_state(state)
        joints_i = np.rint(joints / self.da)
        # tuple of ints as unique id
        return tuple(pos_i), tuple(joints_i)

    def key_to_state(self, key):
        (pos_i, joints_i) = key
        joints_i = np.array(joints_i)
        pos = np.array(pos_i) * self.dpos
        joints = (joints_i * self.da) + self.env.arm.ul
        return np.concatenate([pos, joints])

    @ staticmethod
    def is_invalid_transition(trans_cost):
        return trans_cost == Environment.INF

    @ staticmethod
    def bottle_pos_from_state(state):
        return state[:3].copy()

    @ staticmethod
    def joint_pose_from_state(state):
        return state[3:].copy()

    @ staticmethod
    def format_state(bottle_pos, joints):
        return np.concatenate([bottle_pos, joints])

    @ staticmethod
    def state_to_str(state):
        s = ", ".join(["%.2f" % val for val in state])
        return s


def test_state_indexing():
    state = list(range(10))
    assert(NaivePlanner.bottle_pos_from_state(state) == [0, 1, 2])
    assert(NaivePlanner.joint_pose_from_state(
        state) == [3, 4, 5, 6, 7, 8, 9])
