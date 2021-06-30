import pybullet as p
import pybullet_data
import math
import numpy as np
import pickle
import os

import helpers
from param import parse_arguments
from main import Main, piecewise_execution
from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace, EnvParams, StateTuple

DEG_TO_RAD = math.pi / 180.0


def generate_random_start_goals(main: Main, args, num_pairs=10):
    main.env.arm.set_general_max_reach(
        all_contact_heights=np.linspace(start=0.2 * main.env.bottle.height, stop=main.env.bottle.height, num=3))
    max_horiz_dist = 1.0 * main.env.arm.MAX_HORIZ_REACH
    min_horiz_dist = 0.2 * main.env.arm.MAX_HORIZ_REACH
    empty_action = (np.zeros(main.env.arm.num_joints), 50)

    min_start_goal_dist = 0.5

    # arbitrary
    single_param = main.env.gen_random_env_param_set(num=1)[0]

    start_goal_pairs = []
    count = 0
    while count < num_pairs:
        goal_bottle_angle = np.random.uniform(
            low=0, high=90)
        start_bottle_angle = np.random.uniform(
            low=0, high=90)
        arm_EE_psi = np.random.uniform(
            low=-5, high=90)

        # needs to behind bottle generally facing towards goal due to simple
        # heuristic
        max_offset = 30
        arm_EE_theta = np.random.uniform(
            low=start_bottle_angle - max_offset,
            high=start_bottle_angle + max_offset)
        rand_joints = np.random.uniform(
            low=main.env.arm.ll, high=main.env.arm.ul) % (2 * math.pi)
        rand_joints[0] = (arm_EE_theta * DEG_TO_RAD) % (2 * math.pi)
        rand_joints[1] = (arm_EE_psi * DEG_TO_RAD) % (2 * math.pi)

        goal_dist = np.random.uniform(
            low=0.3, high=1.0) * max_horiz_dist
        start_dist = np.random.uniform(
            low=0.3, high=1.0) * max_horiz_dist

        start_x = start_dist * math.cos(start_bottle_angle * DEG_TO_RAD)
        start_y = start_dist * math.sin(start_bottle_angle * DEG_TO_RAD)
        goal_x = goal_dist * math.cos(goal_bottle_angle * DEG_TO_RAD)
        goal_y = goal_dist * math.sin(goal_bottle_angle * DEG_TO_RAD)

        dist = ((start_x - goal_x) ** 2 + (start_y - goal_y) ** 2) ** 0.5
        if dist < min_start_goal_dist:
            continue

        # make sure initial arm configuration is stable
        startb = [start_x, start_y] + [main.env.bottle.PLANE_OFFSET]
        goalb = [goal_x, goal_y] + [0]
        state_tuple = StateTuple(bottle_pos=startb, bottle_ori=[0, 0, 0, 1],
                                 joints=rand_joints)
        res = main.env.run_sim(action=empty_action, state=state_tuple, sim_params=single_param)
        start_state = helpers.bottle_EE_to_state(
            bpos=startb, arm=main.env.arm, joints=res.joint_pose)
        goal_state = helpers.bottle_EE_to_state(
            bpos=goalb, arm=main.env.arm, joints=res.joint_pose)
        main.planner.start = start_state
        main.planner.goal = goal_state

        # make sure configuration is solvable
        success = piecewise_execution(main.planner, main.env,
                                      exec_params_set=main.exec_params_set,
                                      replay_results=args.replay_results,
                                      directory=main.planner_folder,
                                      start_goal_idx=count,
                                      visualize=args.visualize,
                                      max_time_s=args.max_time)
        if success:
            count += 1
            start_goal_pairs.append((startb, goalb, res.joint_pose))

    return start_goal_pairs


if __name__ == "__main__":
    # args = parse_arguments()
    # main = Main(args)
    #
    # start_goals = generate_random_start_goals(main, args, num_pairs=10)

    # aggregate all saved start goals
    start_goals = []
    data_root = "/home/alvin/research/planning_uncertainty/single"
    for dir in os.listdir(data_root):
        for f in os.listdir(os.path.join(data_root, dir)):
            if f[-3:] == "npz":
                results = np.load(os.path.join(data_root, dir, f), allow_pickle=True)
                try:
                    start = results["node_path"][0].state
                    goal = results["state_path"][-1]
                    start_goals.append((start[:3], goal[:3], start[3:]))
                except Exception as e:
                    print(e)

    print("Saved %d pairs" % len(start_goals))
    with open(os.path.join("/home/alvin/research/planning_uncertainty/start_goals.obj"), "wb") as outfile:
        pickle.dump(start_goals, outfile)
