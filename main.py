import pybullet as p
import pybullet_data
import math
import numpy as np
import pickle
import time
import os
import sys
import json
import datetime
import typing as t
import re

from param import parse_arguments
from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace, EnvParams, StateTuple
from naive_joint_space_planner import NaivePlanner, SINGLE, CUSTOM, LAZY, FULL
import helpers

# Constants and Enums
DEG2RAD = math.pi / 180.0
RAD2DEG = 1 / DEG2RAD
GRAVITY = -9.81
BIMODAL = "bimodal"
HIGH_FRIC = "high_fric"
DEFAULT = "default"

"""
Useful Commands:
Run planning on single planner using pre-generated simulation parameters:
$ python main.py --single --single_low_fric --load_params --num_plan_sims 20 --redirect_stdout

Avg planning:
$ python main.py --avg --high_fric --num_plan_sims 10 --redirect_stdout --fall_thresh 0.1

Execute the plan generated above:
$ python main.py --single --single_low_fric --load_params --num_plan_sims 20 \
--redirect_stdout --replay_results --replay_dir <path/to/directory/with/results>

Parse output of execution and planning steps above to calculate some statistics:
$ python parse_results.py --results_dir --redirect_stdout <path/to/directory/with/results>
"""


def run_policy(planner: NaivePlanner, env: Environment, policy,
               exec_params: EnvParams, cur_bottle_ori=np.array([0, 0, 0, 1.]),
               break_on_fail=True, visualize=False):
    cur_bottle_pos = planner.bottle_pos_from_state(planner.start)
    cur_joints = planner.joint_pose_from_state(planner.start)
    env.arm.reset(cur_joints)

    bottle_goal = planner.bottle_pos_from_state(planner.goal)
    is_fallen = False
    executed_traj = []

    print(exec_params)
    for step in range(len(policy)):
        if visualize:
            env.goal_line_id = env.draw_line(
                lineFrom=bottle_goal,
                lineTo=bottle_goal + np.array([0, 0, 1]),
                lineColorRGB=[0, 0, 1], lineWidth=1,
                replaceItemUniqueId=None,
                lifeTime=0)

        s = "[%d]: " % step
        s += "Bpos(" + ",".join(["%.2f" % v for v in cur_bottle_pos]) + "), "
        s += "Bang(" + "%.1f" % (Bottle.calc_vert_angle(cur_bottle_ori) * RAD2DEG) + "), "
        s += "Joints(" + ",".join(["%.3f" % v for v in cur_joints]) + "), "
        print(s)

        print("Action: %s" % np.array2string(policy[step][0], precision=3))

        dist_bottle_to_goal = np.linalg.norm(cur_bottle_pos[:2] - bottle_goal[:2])
        print("Bottle dist %.3f ? %.3f (thresh)" % (dist_bottle_to_goal, planner.sim_dist_thresh))

        state_tuple = StateTuple(bottle_pos=cur_bottle_pos, bottle_ori=cur_bottle_ori,
                                 joints=cur_joints)
        step_is_fallen, is_collision, cur_bottle_pos, cur_bottle_ori, cur_joints, _ = (
            env.run_sim(state=state_tuple, action=policy[step], sim_params=exec_params)
        )
        if visualize:
            env.goal_line_id = env.draw_line(
                lineFrom=bottle_goal,
                lineTo=bottle_goal + np.array([0, 0, 1]),
                lineColorRGB=[0, 0, 1], lineWidth=1,
                replaceItemUniqueId=None,
                lifeTime=0)
        is_fallen |= step_is_fallen
        executed_traj.append(
            planner.format_state(bottle_pos=cur_bottle_pos, joints=cur_joints))

        if is_fallen and break_on_fail:
            break

    is_success = planner.reached_goal(cur_bottle_pos) and not is_fallen
    return is_fallen, is_success, executed_traj, cur_bottle_ori


def piecewise_execution(planner: NaivePlanner, env: Environment,
                        exec_params_set: t.List[EnvParams],
                        replay_results,
                        res_fname,
                        visualize,
                        max_time_s,
                        cur_bottle_ori=np.array([0, 0, 0, 1])):
    if not replay_results:
        start_time = time.time()
        try:
            with helpers.time_limit(max_time_s):
                state_path, policy, node_path = planner.plan(bottle_ori=cur_bottle_ori)

            time_taken = time.time() - start_time
            print("time taken: %.3f" % time_taken, flush=True)
            print("Saving plan to %s" % res_fname, flush=True)
            np.savez("%s" % res_fname,
                     state_path=state_path, policy=policy, node_path=node_path)

        except helpers.TimeoutException:
            print("time taken: NA", flush=True)

    else:
        print("Replaying results!")
        try:
            results = np.load("%s.npz" % res_fname, allow_pickle=True)
        except Exception as e:
            print("Results: %s not found! due to %s" % (res_fname, e))
            return

        policy = results["policy"]
        state_path = results["state_path"]
        node_path = results["node_path"]
        if len(policy) == 0:
            print("Empty Path! Skipping....", flush=True)
            return

        fall_count = 0
        success_count = 0
        for exec_i, exec_params in enumerate(exec_params_set):
            print("New Test with params: %s" % exec_params, flush=True)
            is_fallen, is_success, _, _ = run_policy(planner, env, policy, exec_params,
                                                     break_on_fail=True, visualize=visualize)
            env.reset()
            print("Exec #%d: fell: %d, success: %d" %
                  (exec_i, is_fallen, is_success), flush=True)
            print()
            success_count += is_success
            fall_count += is_fallen

        print("Fall Rate: %.2f, success rate: %.2f" % (
            fall_count / float(len(exec_params_set)),
            success_count / float(len(exec_params_set))
        ), flush=True)


def piecewise_execution_replan_helper(planner: NaivePlanner, env: Environment,
                                      exec_params: EnvParams, res_fname: str, max_time_s: int,
                                      cur_bottle_ori):
    is_success = False
    is_fallen = False
    plan_count = 0
    final_state_path = []
    final_policy = []
    final_node_path = []
    executed_traj = None

    try:
        start_time = time.time()
        with helpers.time_limit(max_time_s):
            while not is_success and not is_fallen:
                # plan from current state
                plan_count += 1
                if executed_traj is not None:
                    planner.start = executed_traj[-1]
                state_path, policy, node_path = planner.plan(bottle_ori=cur_bottle_ori)
                final_state_path += state_path
                final_policy += policy
                final_node_path += node_path

                is_fallen, is_success, executed_traj, cur_bottle_ori = (
                    run_policy(planner, env, policy, exec_params,
                               cur_bottle_ori=cur_bottle_ori,
                               break_on_fail=True, visualize=False)
                )
                if is_success:
                    status_msg = "success"
                elif is_fallen:
                    status_msg = "failure, knocked over"
                else:
                    status_msg = "failure, replanning..."

                print(f"Execution of Plan: {status_msg}")
                time_taken = time.time() - start_time
                print("time taken: %.3f" % time_taken, flush=True)
                print("Saving plan to %s" % res_fname, flush=True)
                np.savez("%s" % res_fname,
                         state_path=final_state_path, policy=final_policy, node_path=final_node_path)

    except helpers.TimeoutException:
        print("time taken: NA", flush=True)

    return is_success, is_fallen, plan_count


def piecewise_execution_replan(planner: NaivePlanner, env: Environment,
                               exec_params_set,
                               res_fname,
                               max_time_s,
                               cur_bottle_ori=np.array([0, 0, 0, 1])):
    orig_start = planner.start
    orig_goal = planner.goal

    for exec_i, exec_params in enumerate(exec_params_set):
        planner.start = orig_start
        planner.goal = orig_goal
        piecewise_execution_replan_helper(planner, env,
                                          exec_params=exec_params,
                                          res_fname=res_fname + "_exec_%d" % exec_i,
                                          max_time_s=max_time_s,
                                          cur_bottle_ori=cur_bottle_ori)


class Main(object):
    def __init__(self, args):
        self.planner_folder = None
        self.setup_dirs(args)

        self.env = None
        self.setup_sim(args)

        self.single_param = None
        self.plan_params_set = None
        self.exec_params_set = None
        self.setup_sim_params(args)

        self.planner = None
        self.targets = None
        self.setup_planner(args)

    def setup_dirs(self, args):
        date_time_str = '@'.join(str(datetime.datetime.now()).split(' '))
        argparse_dict = vars(args)
        arg_str = json.dumps(argparse_dict, indent=4)
        print(arg_str)

        if args.single:
            self.plan_type = SINGLE
            sub_dir = date_time_str
        elif args.full:
            self.plan_type = FULL
            sub_dir = "fall_%.2f_%s" % (args.fall_thresh, date_time_str)
        elif args.lazy:
            self.plan_type = LAZY
            sub_dir = "fall_%.2f_%s" % (args.fall_thresh, date_time_str)
        elif args.custom:
            self.plan_type = CUSTOM
            sub_dir = "fall_%.2f_beta_var_%.2f_%s" % (args.fall_thresh, args.beta_var_thresh, date_time_str)
        else:
            raise Exception("Need to specify planner type (single, full, lazy, custom)")
        main_dir = self.plan_type

        if args.replay_results:
            assert args.replay_dir
            self.planner_folder = args.replay_dir
        else:
            self.planner_folder = os.path.join(main_dir, sub_dir)
            if not os.path.exists(self.planner_folder):
                os.makedirs(self.planner_folder)

        if args.redirect_stdout:
            if args.replay_results:
                sys.stdout = open(os.path.join(self.planner_folder, "exec_output.txt"), "w")
            else:
                sys.stdout = open(os.path.join(self.planner_folder, "plan_output.txt"), "w")
        print("python " + " ".join(sys.argv), flush=True)  # print specified arguments

    def setup_sim(self, args):
        if args.visualize:
            p.connect(p.GUI)  # or p.DIRECT for nongraphical version
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=145, cameraPitch=-10,
                cameraTargetPosition=[0, 0, 0])
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        p.loadURDF(Environment.plane_urdf_filepath, basePosition=[0, 0, 0])

        # if  the below isn't true, you're expecting bottle to fall in exactly
        # the same state bin as the goal
        assert (args.goal_thresh >= args.dx)

        bottle = Bottle()
        arm = Arm(kuka_id=p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0]))
        self.env = Environment(arm, bottle, is_viz=args.visualize)

    def setup_sim_params(self, args):
        # Load execution params (unused if args.replay_results False)
        global_params_path = "sim_params_set.obj"
        if args.params_path != "":
            params_path = args.params_path
        else:
            params_path = global_params_path

        with open(params_path, "rb") as f:
            default_params = pickle.load(f)

        # generate new planning env parameters, not exec
        if args.load_params:
            self.plan_params_set = default_params["plan_params_set"]
            self.exec_params_set = default_params["exec_params_set"]
            self.single_param = default_params["single_param"]
            if args.num_plan_sims != len(self.plan_params_set):
                raise Exception(
                    f"num_plan_sims {args.num_plan_sims} != len(plan_params_set) {len(self.plan_params_set)}")

        else:
            self.env.set_distribs(min_fric=self.env.bottle.low_fric_min, max_fric=self.env.bottle.low_fric_max)
            self.exec_params_set = self.env.gen_random_env_param_set(num=(args.num_exec_sims // 3))
            self.plan_params_set = self.env.gen_random_env_param_set(num=(args.num_plan_sims // 3))

            self.env.set_distribs(min_fric=self.env.bottle.low_fric_max, max_fric=self.env.bottle.high_fric_min)
            self.exec_params_set += self.env.gen_random_env_param_set(
                num=(args.num_exec_sims // 3 + args.num_exec_sims % 3))
            self.plan_params_set += self.env.gen_random_env_param_set(
                num=(args.num_plan_sims // 3 + args.num_plan_sims % 3))

            self.env.set_distribs(min_fric=self.env.bottle.high_fric_min, max_fric=self.env.bottle.high_fric_max)
            self.exec_params_set += self.env.gen_random_env_param_set(num=(args.num_exec_sims // 3))
            self.plan_params_set += self.env.gen_random_env_param_set(num=(args.num_plan_sims // 3))

            # Test if single planner using low and high friction produce different performance
            if args.single_low_fric:
                print("Manually forcing single planner to use LOW friction")
                self.env.set_distribs(min_fric=self.env.bottle.low_fric_min, max_fric=self.env.bottle.low_fric_max)
                self.single_param = self.env.gen_random_env_param_set(num=1)[0]

            elif args.single_high_fric:
                print("Manually forcing single planner to use HIGH friction")
                self.env.set_distribs(min_fric=self.env.bottle.high_fric_min, max_fric=self.env.bottle.high_fric_max)
                self.single_param = self.env.gen_random_env_param_set(num=1)[0]

            elif args.single_med_fric:
                print("Manually forcing single planner to use MEDIUM friction")
                self.env.set_distribs(min_fric=self.env.bottle.low_fric_max, max_fric=self.env.bottle.high_fric_min)
                self.single_param = self.env.gen_random_env_param_set(num=1)[0]

            else:
                self.single_param = None

        # save all parameters used
        if not args.replay_results:
            print("single_param:")
            print(self.single_param)

            print("plan_params_set:")
            for param in self.plan_params_set:
                print(param)

            with open("%s/sim_params_set.obj" % self.planner_folder, "wb") as f:
                exec_plan_params = dict(exec_params_set=self.exec_params_set,
                                        plan_params_set=self.plan_params_set,
                                        single_param=self.single_param)
                pickle.dump(exec_plan_params, f)
            with open(os.path.join(self.planner_folder, "args.json"), "w") as outfile:
                json.dump(self.planner_folder, outfile, indent=4)

            if args.save_global_params:
                with open(global_params_path, "wb") as f:
                    exec_plan_params = dict(exec_params_set=self.exec_params_set,
                                            plan_params_set=self.plan_params_set,
                                            single_param=self.single_param)
                    pickle.dump(exec_plan_params, f)
        else:
            print("exec_params_set:")
            for param in self.exec_params_set:
                print(param)

    def setup_planner(self, args):
        with open("test_start_goals.obj", "rb") as f:
            self.start_goals = pickle.load(f)
        # (startb, goalb, start_joints) = start_goals[3]
        # # startb += np.array([0.2, 0.1, 0])
        # goalb += np.array([-0., -0, 0])  # .43, .06
        # start_goals[3] = (startb, goalb, start_joints)

        # with open("filtered_start_goals.obj", "wb") as f:
        #     pickle.dump(start_goals, f)

        # Create planner
        da_rad = args.dtheta * math.pi / 180.
        self.planner = NaivePlanner(env=self.env, plan_type=self.plan_type, sim_params_set=self.plan_params_set,
                                    dist_thresh=args.goal_thresh, eps=args.eps, da_rad=da_rad,
                                    dx=args.dx, dy=args.dy, dz=args.dz, visualize=args.visualize,
                                    fall_thresh=args.fall_thresh, use_ee_trans_cost=args.use_ee_trans_cost,
                                    single_param=self.single_param,
                                    lazy=args.lazy)

        # Possibly specify a specific start-goal pair to run
        if args.start_goal != -1:
            self.targets = [args.start_goal, ]
        elif args.start_goal_range is not None:
            try:
                left, right = re.findall("(\d+)-(\d+)", args.start_goal_range)[0]
                self.targets = list(range(int(left), int(right) + 1))
            except:
                print("Failed to parse start_goal_range string: %s, requires format (\d+)-(\d+)" %
                      args.start_goal_range)
                exit(-1)
        else:
            self.targets = list(range(0, 11))

    def run_experiments(self):
        for start_goal_idx in self.targets:
            print("Start goal idx: %d" % start_goal_idx, flush=True)

            if args.solved_path_index is not None:
                assert args.replay_dir is not None
                res_fname = "%s/results_%d" % (args.replay_dir, start_goal_idx)
                results = np.load("%s.npz" % res_fname, allow_pickle=True)
                node_path = results["node_path"]
                (_, goalb, _) = self.start_goals[start_goal_idx]
                start = node_path[args.solved_path_index]
                startb = self.planner.bottle_pos_from_state(start.state)
                start_joints = self.planner.joint_pose_from_state(start.state)
                start_bottle_ori = start.bottle_ori

            else:
                res_fname = "%s/results_%d" % (self.planner_folder, start_goal_idx)
                (startb, goalb, start_joints) = self.start_goals[start_goal_idx]
                start_bottle_ori = np.array([0, 0, 0, 1])

            start_state = helpers.bottle_EE_to_state(
                bpos=startb, arm=self.env.arm, joints=start_joints)
            goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=self.env.arm)
            self.planner.start = start_state
            self.planner.goal = goal_state

            if args.use_replan and not args.replay_results:
                piecewise_execution_replan_helper(self.planner, self.env,
                                                  exec_params=self.exec_params_set[0],
                                                  res_fname=res_fname,
                                                  max_time_s=args.max_time,
                                                  cur_bottle_ori=start_bottle_ori)
            else:
                start = time.time()
                piecewise_execution(self.planner, self.env,
                                    exec_params_set=self.exec_params_set,
                                    replay_results=args.replay_results,
                                    res_fname=res_fname,
                                    visualize=args.visualize,
                                    max_time_s=args.max_time,
                                    cur_bottle_ori=start_bottle_ori)
                end = time.time()
                print("total planning time: %.2f" % (end - start))


if __name__ == "__main__":
    args = parse_arguments()
    main = Main(args)
    main.run_experiments()
