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


class Node(object):
    def __init__(self, cost, state, nearest_arm_pos_i,
                 nearest_arm_pos, ee_pos, fall_history=[], mode_sim_param=None, fall_prob=-1.0,
                 bottle_ori=np.array([0, 0, 0, 1]), g=0, h=0, bottle_goal_dist=np.inf, arm_bottle_dist=np.inf,
                 is_fully_evaluated=False, prev_n=None, prev_ai=None):
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
        self.is_fully_evaluated = is_fully_evaluated  # only used for lazy approach
        self.prev_n = prev_n
        self.prev_ai = prev_ai

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
    def __init__(self, env, sim_params_set, dist_thresh=1e-1, eps=1, dx=0.1, dy=0.1, dz=0.1,
                 da_rad=15 * DEG2RAD, visualize=False, start=None, goal=None,
                 fall_thresh=0.2, use_ee_trans_cost=True,
                 single_param=None, var_thresh=0.02):
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

        # method of simulating an action
        self.sim_func = self.env.run_sim
        self.full_sim_func = self.env.run_multiple_sims
        self.process_multiple_sim_results = self.avg_results

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

        return fall_proportion, fall_history, bpos, bori, next_joint_pos, mode_sim_param

    def plan(self, bottle_ori=np.array([0, 0, 0, 1])):
        # initialize open set with start and G values
        start = time.time()
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
                 ee_pos=arm_positions[-1],
                 is_fully_evaluated=True)]
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
        num_full_evals = 0
        num_lazy_evals = 0
        invalid_count = 0

        # find solution
        goal_expanded = False
        end = time.time()
        setup_time = end - start
        total_pre_expansion_time = 0
        total_calc_cost_time = 0
        total_sim_time = 0
        total_process_sim_time = 0
        start_total_time = time.time()
        while not goal_expanded and len(open_set) > 0:
            sim_time = 0
            process_sim_time = 0
            insert_open_time = 0
            calc_cost_time = 0

            start = time.time()

            # get next state to expand
            n = heapq.heappop(open_set)
            state = n.state
            state_key = self.state_to_key(state)

            if state_key in closed_set:
                continue

            num_expansions += 1

            bottle_ori = n.bottle_ori
            cur_joints = self.joint_pose_from_state(state)
            bottle_pos = self.bottle_pos_from_state(state)
            cur_state_tuple = StateTuple(bottle_pos=bottle_pos, bottle_ori=bottle_ori, joints=cur_joints)

            if n.is_fully_evaluated:
                closed_set.add(state_key)

                # check if found goal, if so loop will terminate in next iteration
                if self.reached_goal_node(n):
                    goal_expanded = True
                    self.new_goal = state

                # extra current total move-cost of current state
                assert (state_key in self.G)
                cur_cost = self.G[state_key]
                end = time.time()
                pre_expansion = end - start

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

                    # N = 1 only use one simulation parameter set
                    start = time.time()
                    (is_fallen, is_collided, next_bottle_pos,
                     next_bottle_ori, next_joint_pose, _) = self.sim_func(
                        action=action, state=cur_state_tuple,
                        sim_params=self.single_param)
                    sim_time += time.time() - start

                    start = time.time()
                    mode_sim_param = self.single_param
                    invalid = is_fallen
                    fall_history = [is_fallen]
                    fall_prob = 1 if is_fallen else 0
                    num_lazy_evals += 1

                    # completely ignore actions that knock over bottle with high
                    # probability
                    if invalid:
                        invalid_count += 1
                        continue

                    process_sim_time += time.time() - start

                    start = time.time()
                    new_arm_positions = self.env.arm.get_joint_link_positions(
                        next_joint_pose)

                    move_cost = self.calc_move_cost(n, new_arm_positions)
                    trans_cost = self.calc_trans_cost(move_cost=move_cost)

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
                    h = self.heuristic(bottle_goal_dist, arm_bottle_dist, arm_goal_dist)
                    new_G = cur_cost + trans_cost

                    end = time.time()
                    calc_cost_time += end - start

                    # if state not expanded or found better path to next_state
                    start = time.time()
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
                            ee_pos=new_arm_positions[-1],
                            prev_n=n,
                            prev_ai=ai,
                            is_fully_evaluated=not is_collided)  # if no collision, no need to simulate more
                        # print("     ai[%d] -> %s" % (ai, new_node))
                        # print("     ", np.array2string(np.array(next_bottle_ori), precision=2),
                        #       "%.2f" % Bottle.calc_vert_angle(next_bottle_ori))
                        heapq.heappush(open_set, new_node)
                        transitions[next_state_key] = (ai, n)
                        # print("%s -> %s" % (self.state_to_str(state), self.state_to_str(next_state)))

                    end = time.time()
                    insert_open_time += end - start

            # re-evaluate using N simulations
            else:
                num_full_evals += 1
                prev_ai, prev_n = n.prev_ai, n.prev_n
                prev_action = self.A.get_action(prev_ai)
                prev_bottle = self.bottle_pos_from_state(prev_n.state)
                prev_joints = self.joint_pose_from_state(prev_n.state)
                prev_state_tuple = StateTuple(bottle_pos=prev_bottle, bottle_ori=prev_n.bottle_ori, joints=prev_joints)

                # TODO: pick better initialization
                alpha = self.alpha_prior
                beta = self.beta_prior

                # shuffle the sim params
                sim_params = np.random.permutation(self.sim_params_set)
                all_results = []

                end = time.time()
                pre_expansion = end - start

                count = 0  # represents the same idea as alpha + beta
                while scipy_beta.var(a=alpha, b=beta) > self.var_thresh and count < self.sample_thresh:
                    # sample from parameter distribution
                    sampled_sim_param = sim_params[count]

                    start = time.time()
                    results = self.sim_func(
                        action=prev_action, state=prev_state_tuple,
                        sim_params=sampled_sim_param)

                    sim_time += time.time() - start

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

                start = time.time()
                (fall_prob, fall_history, next_bottle_pos, next_bottle_ori,
                 next_joint_pose, mode_sim_param) = self.process_multiple_sim_results(all_results,
                                                                                      sim_params_set=sim_params[:count])
                process_sim_time += time.time() - start

                invalid = fall_prob > self.fall_thresh

                next_state = np.concatenate([next_bottle_pos, next_joint_pose])
                next_state_key = self.state_to_key(next_state)

                # completely ignore actions that knock over bottle with high
                # probability
                if invalid:
                    invalid_count += 1
                    continue

                if next_state_key in closed_set:
                    continue

                start = time.time()
                new_arm_positions = self.env.arm.get_joint_link_positions(
                    next_joint_pose)

                move_cost = self.calc_move_cost(n, new_arm_positions)
                trans_cost = self.calc_trans_cost(move_cost=move_cost)

                # build next state and check if already expanded
                next_state = np.concatenate([next_bottle_pos, next_joint_pose])
                next_state_key = self.state_to_key(next_state)

                arm_bottle_dist, nn_joint_i, nn_joint_pos = (
                    self.dist_arm_to_bottle(next_state, new_arm_positions))

                # Quick FIX: use EE for transition costs so set nn_joint to EE
                # still true b/c midpoints + joints, so last item is still last joint (EE)
                bottle_goal_dist, arm_goal_dist = self.dist_to_goal(next_state, nn_joint_pos)
                h = self.heuristic(bottle_goal_dist, arm_bottle_dist, arm_goal_dist)
                cur_cost = self.G[self.state_to_key(prev_n.state)]
                new_G = cur_cost + trans_cost

                end = time.time()
                calc_cost_time += end - start

                start = time.time()
                if next_state_key not in self.G or (
                        self.G[next_state_key] > new_G):
                    self.G[next_state_key] = new_G
                    overall_cost = new_G + self.eps * h
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
                        ee_pos=new_arm_positions[-1],
                        is_fully_evaluated=True)

                    heapq.heappush(open_set, new_node)

                    # build directed graph
                    transitions[next_state_key] = (prev_ai, prev_n)

                    end = time.time()
                    insert_open_time += end - start

            print(n, flush=True)

            total_pre_expansion_time += pre_expansion
            total_calc_cost_time += calc_cost_time
            total_sim_time += sim_time
            total_process_sim_time += process_sim_time
            print("expansion_sim_time: %.2f" % sim_time)

        end_total_time = time.time()
        print("Total time: %.4f" % (end_total_time - start_total_time))
        print("Setup time: %.2f" % setup_time)
        print("pre_expansion time: %.2f" % total_pre_expansion_time)
        print("calc cost time: %.2f" % total_calc_cost_time)
        print("sim time: %.2f" % total_sim_time)
        print("process sim time: %.2f" % total_process_sim_time)

        print("States Expanded: %d, found goal: %d" %
              (num_expansions, goal_expanded))
        print("Num full evaluations: %d" % num_full_evals)
        print("Num lazy evaluations: %d" % num_lazy_evals)

        print("invalid count: %d")
        print("Total evaluations: %d" % (num_lazy_evals + num_full_evals))
        print("invalid rate: %.3f" % (invalid_count / (num_lazy_evals + num_full_evals)))
        if not goal_expanded:
            print("path reverse time: %.2f" % 0.0)
            return [], [], []

        start = time.time()
        # reconstruct path
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
        end = time.time()
        path_reverse_time = end - start
        print("path reverse time: %.2f" % path_reverse_time)

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

        dist = np.linalg.norm(self.dist_cost_weights * (np.array(positions[-1]) - bottle_pos))
        return dist, -1, positions[-1]

        # min_dist = None
        # min_i = 0
        # min_pos = None
        # for i, pos in enumerate(positions):
        #     diff = self.dist_cost_weights * (np.array(pos) - bottle_pos)
        #     dist = np.linalg.norm(diff)
        #     if min_dist is None or dist < min_dist:
        #         min_dist = dist
        #         min_i = i
        #         min_pos = pos
        #
        # return min_dist, min_i, min_pos

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

    def heuristic(self, dist_to_goal, dist_arm_to_bottle, arm_goal_dist):
        # print(dist_to_goal, 0.6 * dist_arm_to_bottle, 0.5 * arm_goal_dist)
        return dist_to_goal + dist_arm_to_bottle  # + 0.8 * arm_goal_dist

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
