import pybullet as p
import time
import math
import numpy as np
from scipy.stats import beta as scipy_beta

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
SINGLE = 0
FULL = 1
LAZY = 2
BETA = 3
TYPE_TO_NAME = {
    SINGLE: "single",
    FULL: "full",
    LAZY: "lazy",
    BETA: "beta",
}


class Node(object):
    def __init__(self, cost, state, state_key, ee_pos, prev_n, prev_ai, fall_history=[], mode_sim_param=None,
                 fall_prob=-1.0,
                 bottle_ori=np.array([0, 0, 0, 1]), g=0, h=0,
                 extra_info=None,
                 is_fully_evaluated=False):
        self.cost = cost
        self.fall_prob = fall_prob
        self.fall_history = fall_history
        self.mode_sim_param = mode_sim_param
        self.g = g
        self.h = h
        nearest_arm_pos_i, nearest_arm_pos, bottle_goal_dist, arm_bottle_dist = extra_info
        self.bottle_goal_dist = bottle_goal_dist
        self.arm_bottle_dist = arm_bottle_dist
        self.state = state
        self.state_key = state_key
        self.bottle_ori = bottle_ori
        self.nearest_arm_pos_i = nearest_arm_pos_i
        self.nearest_arm_pos = nearest_arm_pos
        self.ee_pos = ee_pos
        self.is_fully_evaluated = is_fully_evaluated  # only used for lazy approach
        self.prev_n = prev_n
        self.prev_ai = prev_ai

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        s = "(" + ",".join("%d" % v for v in self.state_key[0]) + "|" + ",".join(
            "%d" % v for v in self.state_key[1]) + "), "
        s += "Bpos(" + ",".join(["%.2f" % v for v in self.state[:3]]) + "), "
        s += "C(%.2f) | h(%.2f) | d_bg(%.2f) | d_ab(%.2f)" % (
            self.cost, self.h, self.bottle_goal_dist, self.arm_bottle_dist)
        return s


class NaivePlanner(object):
    def __init__(self, env, plan_type, sim_params_set, dist_thresh=1e-1, eps=1, dx=0.1, dy=0.1, dz=0.1,
                 da_rad=15 * DEG2RAD, visualize=False, start=None, goal=None,
                 fall_thresh=0.2, use_ee_trans_cost=True, save_edge_betas=False,
                 sim_dist_thresh=0.18, single_param=None, var_thresh=0.02):
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

        # beta distribution parameters
        # TODO: find a better initialization for these parameters, specific to each edge
        self.alpha_prior = 1
        self.beta_prior = 1
        self.var_thresh = var_thresh  # (alpha=6, beta=1), (alpha=7, beta=2) passes
        self.move_cost_w = 0.01
        self.sample_thresh = len(sim_params_set)
        self.save_edge_betas = save_edge_betas  # optionally save all (state,action,alpha,beta)

        # define action space
        self.da = da_rad
        self.num_joints = env.arm.num_joints
        self.A = ActionSpace(num_dof=self.num_joints, da_rad=self.da)
        self.G = dict()
        self.use_EE = False
        self.guided_direction = True

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

        self.sim_dist_thresh = sim_dist_thresh
        self.move_cost_w = 0.01

        # method of simulating an action
        self.plan_type = plan_type

    def eval_edge_beta(self, state_tuple, action):
        # TODO: pick better initialization
        alpha = self.alpha_prior
        beta = self.beta_prior

        # shuffle the sim params
        sim_params = np.random.permutation(self.sim_params_set)
        all_results = []

        count = 0  # represents the same idea as alpha + beta
        while scipy_beta.var(a=alpha, b=beta) > self.var_thresh and count < self.sample_thresh:
            # sample from parameter distribution
            sampled_sim_param = sim_params[count]

            results = self.env.run_sim(
                action=action, state=state_tuple,
                sim_params=sampled_sim_param)

            all_results.append(results)

            is_fallen = results[0]
            if is_fallen:
                beta += 1
            else:
                alpha += 1

            count += 1

            if not results.is_collision:
                break

        if results.is_collision:
            print("Alpha: %d, beta: %d" % (alpha, beta))
            self.edge_beta_states.append(
                np.concatenate([state_tuple.bottle_pos, state_tuple.bottle_ori, state_tuple.joints]))
            self.edge_beta_actions.append(np.concatenate([action[0], [action[1]]]))
            self.edge_beta_values.append(np.array([alpha, beta]))

        return all_results

    def expand_state(self, n, state, state_key):
        self.closed_set.add(state_key)
        bottle_ori = n.bottle_ori
        cur_joints = self.joint_pose_from_state(state)
        bottle_pos = self.bottle_pos_from_state(state)
        cur_state_tuple = StateTuple(bottle_pos=bottle_pos, bottle_ori=bottle_ori, joints=cur_joints)
        assert (state_key in self.G)

        # check if found goal, if so loop will terminate in next iteration
        if self.reached_goal_node(n):
            self.goal_expanded = True
            self.new_goal = state

        # explore all actions from this state
        for ai in self.A.action_ids:
            self.check_visualize(is_guide=False)
            # action defined as an offset of joint angles of arm
            action = self.A.get_action(ai)
            # print("   action: " + ",".join(["%.2f" % v for v in action[0]]) + "), ")

            # N = 1 only use one simulation parameter set
            if self.plan_type[LAZY] or self.plan_type[SINGLE]:
                self.num_lazy_evals += 1
                results = [self.env.run_sim(
                    action=action, state=cur_state_tuple,
                    sim_params=self.single_param), ]

            # N > 1
            else:
                if self.plan_type[FULL]:
                    self.num_full_evals += 1
                    results = self.env.run_multiple_sims(
                        action=action, state=cur_state_tuple,
                        sim_params_set=self.sim_params_set)
                elif self.plan_type[BETA]:
                    self.num_full_evals += 1
                    results = self.eval_edge_beta(state_tuple=cur_state_tuple, action=action)
                else:
                    assert False, "Unknown plan_type: %s!" % self.plan_type
            self.check_visualize(is_guide=False)
            self.process_new_successor(prev_n=n, prev_ai=ai, results=results)

    def reevaluate_edge(self, n):
        prev_ai, prev_n = n.prev_ai, n.prev_n
        prev_action = self.A.get_action(prev_ai)
        prev_bottle = self.bottle_pos_from_state(prev_n.state)
        prev_joints = self.joint_pose_from_state(prev_n.state)
        prev_state_tuple = StateTuple(bottle_pos=prev_bottle, bottle_ori=prev_n.bottle_ori, joints=prev_joints)
        if self.plan_type[BETA]:
            results = self.eval_edge_beta(state_tuple=prev_state_tuple, action=prev_action)

        else:
            results = self.env.run_multiple_sims(
                action=prev_action, state=prev_state_tuple,
                sim_params_set=self.sim_params_set)
        self.process_new_successor(prev_n=prev_n, prev_ai=prev_ai, results=results, is_fully_evaluated=True)

    def process_new_successor(self, prev_n, prev_ai, results, is_fully_evaluated=None):
        if len(results) == 1:
            (is_fallen, is_collided, next_bottle_pos,
             next_bottle_ori, next_joint_pose, z_ang_mean) = results[0]
            mode_sim_param = self.single_param
            invalid = is_fallen
            fall_history = [is_fallen]
            fall_prob = 1 if is_fallen else 0
        else:
            (fall_prob, pos_variance, z_ang_variance, z_ang_mean, fall_history, next_bottle_pos,
             next_bottle_ori, next_joint_pose, mode_sim_param) = (
                self.avg_results(results, self.sim_params_set))
            is_collided = results[0][1]
            invalid = fall_prob > self.fall_thresh

        next_state = np.concatenate([next_bottle_pos, next_joint_pose])
        next_state_key = self.state_to_key(next_state)

        if is_collided:
            pass
            # print("   collision", flush=True)

        # completely ignore actions that knock over bottle with high
        # probability
        if invalid:
            # print("   invalid: p=%.2f" % fall_prob, flush=True)
            self.invalid_count += 1
            return

        if is_fully_evaluated is None:
            is_fully_evaluated = not is_collided

        new_arm_positions = self.env.arm.get_joint_link_positions(
            next_joint_pose)
        next_state = np.concatenate([next_bottle_pos, next_joint_pose])
        h, trans_cost, extra_info = self.calc_all_costs(n=prev_n, new_arm_positions=new_arm_positions,
                                                        next_state=next_state)
        new_G = prev_n.g + trans_cost

        if next_state_key not in self.G or (
                self.G[next_state_key] > new_G):
            self.G[next_state_key] = new_G
            overall_cost = new_G + self.eps * h
            new_node = Node(
                cost=overall_cost,
                fall_prob=fall_prob,
                fall_history=fall_history,
                mode_sim_param=mode_sim_param,
                g=new_G, h=h, extra_info=extra_info,
                state=next_state,
                state_key=next_state_key,
                bottle_ori=next_bottle_ori,
                ee_pos=new_arm_positions[-1],
                is_fully_evaluated=is_fully_evaluated,
                prev_n=prev_n,
                prev_ai=prev_ai)
            # print("   successor: %s" % new_node, flush=True)

            heapq.heappush(self.open_set, new_node)
            self.transitions[next_state_key] = (prev_ai, prev_n)
        else:
            pass
            # print("   successor: skipping", flush=True)

    def plan(self, bottle_ori=np.array([0, 0, 0, 1])):
        total_start_time = time.time()

        # initialize open set with start and G values
        self.setup(bottle_ori=bottle_ori)

        while not self.goal_expanded and len(self.open_set) > 0:
            # get next state to expand
            n = heapq.heappop(self.open_set)
            state = n.state
            state_key = self.state_to_key(state)
            if state_key in self.closed_set:
                continue

            # re-evaluate previous transition fully
            if self.plan_type[LAZY] and not n.is_fully_evaluated:
                self.num_full_evals += 1
                print("reevaluating edge: %s" % n, flush=True)
                self.reevaluate_edge(n)

            # expand this state's successors
            else:
                print("expanding edge: %s" % n, flush=True)
                start_time = time.time()
                self.num_expansions += 1
                self.expand_state(n, state, state_key)
                self.avg_expand_time += time.time() - start_time

        if not self.goal_expanded:
            return ([], [], []), (
                self.num_expansions, self.goal_expanded, self.num_full_evals, self.num_lazy_evals, self.invalid_count,
                self.avg_expand_time / self.num_expansions)

        # reconstruct path
        return self.reconstruct_path(), (
            self.num_expansions, self.goal_expanded, self.num_full_evals, self.num_lazy_evals, self.invalid_count,
            self.avg_expand_time / self.num_expansions)

    def reconstruct_path(self):
        policy = []
        planned_path = []
        node_path = []
        state = self.new_goal
        state_key = self.state_to_key(state)
        start_key = self.state_to_key(self.start)
        # NOTE: planned_path does not include initial starting pose!
        while state_key != start_key:
            # state[i]  ex: goal
            planned_path.append(state)
            # state[i-1] + action[i-1] -> node[i] containing state[i]
            ai, n = self.transitions[state_key]
            policy.append(self.A.get_action(ai))
            state_key = self.state_to_key(n.state)
            state = n.state
            node_path.append(n)

        # need to reverse since backwards ordering
        # (node[i], policy[i]) -> state[i]
        # or in other words, ith policy led to ith state
        # so after taking action i, we should reach state i
        planned_path.reverse()
        policy.reverse()
        node_path.reverse()

        return planned_path, policy, node_path

    def setup(self, bottle_ori):
        try:
            arm_positions = self.env.arm.get_joint_link_positions(
                self.joint_pose_from_state(self.start))
        except:
            self.env.reset()
            arm_positions = self.env.arm.get_joint_link_positions(
                self.joint_pose_from_state(self.start))
        arm_bottle_dist, nn_joint_i, nn_joint_pos = (
            self.dist_arm_to_bottle(self.start, arm_positions))
        bottle_goal_dist, _ = self.dist_to_goal(self.start, nn_joint_pos)
        extra_info = (nn_joint_i, nn_joint_pos, bottle_goal_dist, arm_bottle_dist)
        self.open_set = [
            Node(0, state=self.start, state_key=self.state_to_key(self.start),
                 extra_info=extra_info,
                 bottle_ori=bottle_ori,
                 ee_pos=arm_positions[-1],
                 is_fully_evaluated=True,
                 prev_n=None,
                 prev_ai=None)]
        self.closed_set = set()
        self.G = dict()
        self.G[self.state_to_key(self.start)] = 0
        self.transitions = dict()

        # metrics on performance of planner
        self.num_expansions = 0
        self.num_full_evals = 0
        self.num_lazy_evals = 0
        self.invalid_count = 0
        self.avg_expand_time = 0

        # find solution
        self.goal_expanded = False
        self.total_pre_expansion_time = 0
        self.total_calc_cost_time = 0
        self.total_sim_time = 0
        self.total_process_sim_time = 0
        self.start_total_time = time.time()

        # store data
        self.edge_beta_states = []
        self.edge_beta_actions = []
        self.edge_beta_values = []

    def debug_view_state(self, state):
        joint_pose = self.joint_pose_from_state(state)
        bx, by, _ = self.bottle_pos_from_state(state)
        self.env.arm.reset(joint_pose)
        link_states = p.getLinkStates(self.env.arm.kukaId, range(7))
        eex, eey = link_states[-1][4][:2]
        dist = ((bx - eex) ** 2 + (by - eey) ** 2) ** 0.5
        print("dist, bx, by: %.2f, (%.2f, %.2f)" % (dist, bx, by))

    def check_visualize(self, is_guide, goal_bpos=None):
        # return
        if not self.visualize:
            return

        vertical_offset = np.array([0, 0, 0.5])
        if is_guide:
            assert goal_bpos is not None
            self.env.target_line_id = self.env.draw_line(
                lineFrom=goal_bpos,
                lineTo=goal_bpos + vertical_offset,
                lineColorRGB=[0, 0, 1], lineWidth=1,
                replaceItemUniqueId=self.env.target_line_id,
                lifeTime=0)

        else:
            goal_bpos = self.bottle_pos_from_state(self.goal)
            self.env.goal_line_id = self.env.draw_line(
                lineFrom=goal_bpos,
                lineTo=goal_bpos + vertical_offset,
                lineColorRGB=[0, 0, 1], lineWidth=1,
                replaceItemUniqueId=self.env.goal_line_id,
                lifeTime=0)

    def calc_all_costs(self, n: Node, new_arm_positions, next_state):
        arm_bottle_dist, nn_joint_i, nn_joint_pos = (
            self.dist_arm_to_bottle(next_state, new_arm_positions))
        move_cost = self.calc_move_cost(n, new_arm_positions)
        trans_cost = self.calc_trans_cost(move_cost=move_cost)

        # Quick FIX: use EE for transition costs so set nn_joint to EE
        # still true b/c midpoints + joints, so last item is still last joint (EE)
        bottle_goal_dist, arm_goal_dist = self.dist_to_goal(next_state, nn_joint_pos)
        h = self.heuristic(bottle_goal_dist, arm_bottle_dist)

        return h, trans_cost, (nn_joint_i, nn_joint_pos, bottle_goal_dist, arm_bottle_dist)

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

        # calculate variance
        pos_variance = np.sum(np.var(all_bpos, axis=0))  # sum of variance of each dimension (x, y, z)
        z_ang_variance = np.var(all_bang)
        z_ang_mean = np.mean(all_bang)

        return fall_proportion, pos_variance, z_ang_variance, z_ang_mean, fall_history, bpos, bori, next_joint_pos, mode_sim_param

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
        self.check_visualize(is_guide=True, goal_bpos=bottle_pos)

        dist = np.linalg.norm(self.dist_cost_weights * (np.array(positions[-1]) - bottle_pos))
        return dist, -1, positions[-1]

    def calc_trans_cost(self, move_cost):
        return self.move_cost_w * move_cost

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

    def heuristic(self, dist_to_goal, dist_arm_to_bottle):
        return dist_to_goal + dist_arm_to_bottle

    def reached_goal_node(self, node: Node):
        return node.bottle_goal_dist < self.dist_thresh

    def reached_goal(self, state):
        x, y, _ = self.bottle_pos_from_state(state)
        gx, gy, _ = self.bottle_pos_from_state(self.goal)
        dist_to_goal = (x - gx) ** 2 + (y - gy) ** 2
        return dist_to_goal < self.sq_dist_thresh

    def state_to_key(self, state):
        pos = np.array(self.bottle_pos_from_state(state))
        pos_i = np.rint(pos / self.dpos).astype(np.int)
        joints = self.joint_pose_from_state(state)
        joints_i = np.rint(joints / self.da).astype(np.int)
        # tuple of ints as unique id
        return tuple(pos_i), tuple(joints_i)

    def key_to_state(self, key):
        (pos_i, joints_i) = key
        joints_i = np.array(joints_i)
        pos = np.array(pos_i) * self.dpos
        joints = (joints_i * self.da) + self.env.arm.ul
        return np.concatenate([pos, joints])

    @staticmethod
    def is_invalid_transition(trans_cost):
        return trans_cost == Environment.INF

    @staticmethod
    def bottle_pos_from_state(state):
        return state[:3].copy()

    @staticmethod
    def joint_pose_from_state(state):
        return state[3:].copy()

    @staticmethod
    def format_state(bottle_pos, joints):
        return np.concatenate([bottle_pos, joints])

    @staticmethod
    def state_to_str(state):
        s = ", ".join(["%.2f" % val for val in state])
        return s


def test_state_indexing():
    state = list(range(10))
    assert (NaivePlanner.bottle_pos_from_state(state) == [0, 1, 2])
    assert (NaivePlanner.joint_pose_from_state(
        state) == [3, 4, 5, 6, 7, 8, 9])
