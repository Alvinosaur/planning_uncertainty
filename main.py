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

from param import parse_arguments
from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace, EnvParams, StateTuple
from naive_joint_space_planner import NaivePlanner, SINGLE, AVG
import helpers

# Constants and Enums
GRAVITY = -9.81
BIMODAL = "bimodal"
HIGH_FRIC = "high_fric"
DEFAULT = "default"

"""
Useful Commands:
Run planning on single planner using pre-generated simulation parameters:
$ python main.py --single --single_low_fric --load_params --n_sims 20 --redirect_stdout

Avg planning:
$ python main.py --avg --high_fric --n_sims 10 --redirect_stdout --fall_thresh 0.1

Execute the plan generated above:
$ python main.py --single --single_low_fric --load_params --n_sims 20 \
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
        state_tuple = StateTuple(bottle_pos=cur_bottle_pos, bottle_ori=cur_bottle_ori,
                                 joints=cur_joints)
        step_is_fallen, is_collision, cur_bottle_pos, cur_bottle_ori, cur_joints = (
            env.run_sim(state=state_tuple, action=policy[step], sim_params=exec_params,
                        prev_state=None, prev_action=None)
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

    is_success = planner.reached_goal(cur_bottle_pos)
    return is_fallen, is_success, executed_traj, cur_bottle_ori


def piecewise_execution(planner: NaivePlanner, env: Environment,
                        exec_params_set: t.List[EnvParams],
                        replay_results,
                        res_fname,
                        visualize,
                        max_time_s):
    if not replay_results:
        start_time = time.time()
        try:
            with helpers.time_limit(max_time_s):
                state_path, policy, node_path = planner.plan()

            time_taken = time.time() - start_time
            print("time taken: %.3f" % time_taken, flush=True)
            print("Saving plan to %s" % res_fname, flush=True)
            np.savez("%s" % res_fname,
                     state_path=state_path, policy=policy, node_path=node_path)

        except helpers.TimeoutException:
            print("time taken: NA", flush=True)

    else:
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
                                      exec_params: EnvParams, res_fname: str, max_time_s: int):
    is_success = False
    is_fallen = False
    plan_count = 0
    final_state_path = []
    final_policy = []
    final_node_path = []
    cur_bottle_ori = [0, 0, 0, 1]
    executed_traj = None

    try:
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
                print(f"Execution of Plan: {'success' if is_success else 'fail'}")
        if is_success:
            print("Saving plan to %s" % res_fname, flush=True)
            np.savez("%s" % res_fname,
                     state_path=final_state_path, policy=final_policy, node_path=final_node_path)

    except helpers.TimeoutException:
        print("time taken: NA", flush=True)

    return is_success, is_fallen, plan_count


def piecewise_execution_replan(planner: NaivePlanner, env: Environment,
                               exec_params_set,
                               res_fname,
                               max_time_s):
    orig_start = planner.start
    orig_goal = planner.goal

    for exec_i, exec_params in enumerate(exec_params_set):
        planner.start = orig_start
        planner.goal = orig_goal
        piecewise_execution_replan_helper(planner, env,
                                          exec_params=exec_params,
                                          res_fname=res_fname + "_exec_%d" % exec_i,
                                          max_time_s=max_time_s)


def main():
    date_time_str = '@'.join(str(datetime.datetime.now()).split(' '))
    args = parse_arguments()
    argparse_dict = vars(args)
    arg_str = json.dumps(argparse_dict, indent=4)
    print(arg_str)

    # Parsing planner type
    if args.bimodal:
        sample_strat = BIMODAL
    elif args.high_fric:
        sample_strat = HIGH_FRIC
    else:
        sample_strat = DEFAULT

    if args.single:
        plan_type = SINGLE
        sub_dir = "/%s" % date_time_str
        max_time_s = args.max_time

    elif args.avg:
        plan_type = AVG
        fall_thresh = args.fall_thresh
        sub_dir = "/%s_fall_thresh_%.2f" % (date_time_str, fall_thresh)
        # Avg Planner allotted N x max time of single planner since at worse
        # case needs to simulate each action N times
        max_time_s = args.n_sims * args.max_time
    else:
        raise Exception("Need to specify planner type with --single or --avg")

    if args.replay_results:
        planner_folder = args.replay_dir
    else:
        planner_folder = f"{plan_type}_{sample_strat}" + sub_dir
        if not os.path.exists(planner_folder):
            os.makedirs(planner_folder)

    # General
    if args.redirect_stdout:
        if args.replay_results:
            sys.stdout = open("%s/exec_output.txt" % planner_folder, "w")
        else:
            sys.stdout = open("%s/plan_output.txt" % planner_folder, "w")
    print("python " + " ".join(sys.argv), flush=True)  # print specified arguments

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

    # Create Arm, Bottle and set up Environment
    bottle = Bottle()
    arm = Arm(kuka_id=p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0]))
    env = Environment(arm, bottle, is_viz=args.visualize)

    # if  the below isn't true, you're expecting bottle to fall in exactly
    # the same state bin as the goal
    assert (args.goal_thresh >= args.dx)

    # Load start-goal pairs to solve
    with open("filtered_start_goals.obj", "rb") as f:
        start_goals = pickle.load(f)

    # Load execution params (unused if args.replay_results False)
    params_path = "sim_params_set.obj"
    if args.params_path != "":
        params_path = args.params_path
    with open(params_path, "rb") as f:
        default_params = pickle.load(f)
        exec_params_set = default_params["exec_params_set"]

    # generate new planning env parameters, not exec
    if args.load_params:
        plan_params_sets = default_params["plan_params_sets"]
        if args.n_sims != len(plan_params_sets):
            raise Exception(
                f"n_sims {args.n_sims} != len(plan_params_set) {len(plan_params_sets)}")
    else:
        print(f"Sampling {sample_strat} distribution")
        # unimodal high friction
        if sample_strat == HIGH_FRIC:
            env.set_distribs(min_fric=0.15, max_fric=0.2)
            plan_params_sets = env.gen_random_env_param_set(num=args.n_sims)

        # bimodal low and high friction with mode being high friction
        elif sample_strat == BIMODAL:
            # Sample 3/4 from high friction distribution since mode simulation
            # results needs to be high friction to be more conservative and avoid
            # avoid knocking bottle over
            env.set_distribs(min_fric=0.15, max_fric=0.2)
            plan_params_sets = env.gen_random_env_param_set(
                num=int(args.n_sims * 3 / 4.))

            # Sample 1/4 from low friction distribution
            env.set_distribs(min_fric=0.05, max_fric=0.1)
            plan_params_sets += env.gen_random_env_param_set(
                num=int(args.n_sims * 1 / 4.))

        else:
            # unimodal at medium friction
            env.set_distribs(min_fric=0.05, max_fric=0.2)
            plan_params_sets = env.gen_random_env_param_set(num=args.n_sims)
    # Test if single planner using low and high friction produce different performance
    if plan_type == SINGLE:
        if args.single_low_fric:
            print("Manually forcing single planner to use LOW friction")
            plan_params_sets[0] = EnvParams(0.70, 0.08, 0.33, 0.11)

        elif args.single_high_fric:
            print("Manually forcing single planner to use HIGH friction")
            plan_params_sets[0] = EnvParams(0.70, 0.175, 0.33, 0.11)

    # save all parameters used
    if not args.replay_results:
        with open("%s/sim_params_set.obj" % planner_folder, "wb") as f:
            exec_plan_params = dict(exec_params_set=exec_params_set,
                                    plan_params_sets=plan_params_sets)
            pickle.dump(exec_plan_params, f)
        with open(os.path.join(planner_folder, "args.json"), "w") as outfile:
            json.dump(argparse_dict, outfile, indent=4)

    # Create planner
    da_rad = args.dtheta * math.pi / 180.
    planner = NaivePlanner(env=env, sim_mode=plan_type, sim_params_set=plan_params_sets,
                           dist_thresh=args.goal_thresh, eps=args.eps, da_rad=da_rad,
                           dx=args.dx, dy=args.dy, dz=args.dz, visualize=args.visualize,
                           fall_thresh=args.fall_thresh, use_ee_trans_cost=args.use_ee_trans_cost)

    # Possibly specify a specific start-goal pair to run
    if args.start_goal != -1:
        targets = [args.start_goal, ]
    else:
        targets = list(range(0, 12))

    # Possibly specify a specific exec_param to run
    if args.exec_param != -1:
        exec_params_set = [exec_params_set[args.exec_param], ]

    for start_goal_idx in targets:
        print("Start goal idx: %d" % start_goal_idx, flush=True)
        res_fname = "%s/results_%d" % (planner_folder, start_goal_idx)
        (startb, goalb, start_joints) = start_goals[start_goal_idx]
        start_state = helpers.bottle_EE_to_state(
            bpos=startb, arm=arm, joints=start_joints)
        goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=arm)
        planner.start = start_state
        planner.goal = goal_state

        if args.use_replan:
            if args.replan_exec_low_fric:
                exec_params = EnvParams(0.70, 0.08, 0.33, 0.11)
            elif args.replan_exec_high_fric:
                exec_params = EnvParams(0.70, 0.16, 0.33, 0.11)
            else:
                exec_params = EnvParams(0.70, 0.13, 0.33, 0.11)

            piecewise_execution_replan_helper(planner, env,
                                              exec_params=exec_params,
                                              res_fname=res_fname,
                                              max_time_s=max_time_s)
        else:
            piecewise_execution(planner, env,
                                exec_params_set=exec_params_set,
                                replay_results=args.replay_results,
                                res_fname=res_fname,
                                visualize=args.visualize,
                                max_time_s=max_time_s)


if __name__ == "__main__":
    main()
