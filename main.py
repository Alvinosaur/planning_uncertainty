import pybullet as p
import pybullet_data
import math
import numpy as np
import pickle
import time
import os
import sys

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace, EnvParams
from naive_joint_space_planner import NaivePlanner
import experiment_helpers as helpers


def piecewise_execution(planner: NaivePlanner, env: Environment,
                        exec_params_set,
                        load_saved=False, play_results=False,
                        res_fname="results"):
    if not load_saved:
        state_path, policy, node_path = planner.plan()
        np.savez("%s" % res_fname,
                 state_path=state_path, policy=policy, node_path=node_path)

    else:
        try:
            results = np.load("%s.npz" % res_fname, allow_pickle=True)
        except:
            print("Results: %s not found!" % res_fname)
            return

        policy = results["policy"]
        state_path = results["state_path"]
        node_path = results["node_path"]
        if len(policy) == 0:
            print("Empty Path! Skipping....")
            return

    # print(state_path)

    if play_results:
        bottle_pos = planner.bottle_pos_from_state(planner.start)
        init_joints = planner.joint_pose_from_state(planner.start)
        env.arm.reset(init_joints)
        bottle_ori = np.array([0, 0, 0, 1])

        bottle_goal = planner.bottle_pos_from_state(planner.goal)

        fall_count = 0
        success_count = 0
        for exec_i, exec_params in enumerate(exec_params_set):
            print("New Test with params: %s" % exec_params)
            cur_joints = init_joints
            cur_bottle_pos = bottle_pos.copy()
            cur_bottle_ori = bottle_ori.copy()
            is_fallen = False
            executed_traj = []

            for step in range(len(policy)):
                env.goal_line_id = env.draw_line(
                    lineFrom=bottle_goal,
                    lineTo=bottle_goal + np.array([0, 0, 1]),
                    lineColorRGB=[0, 0, 1], lineWidth=1,
                    replaceItemUniqueId=None,
                    lifeTime=0)
                step_is_fallen, is_collision, cur_bottle_pos, cur_bottle_ori, cur_joints = (
                    env.run_sim(policy[step], exec_params,
                                cur_joints, cur_bottle_pos, cur_bottle_ori)
                )
                env.goal_line_id = env.draw_line(
                    lineFrom=bottle_goal,
                    lineTo=bottle_goal + np.array([0, 0, 1]),
                    lineColorRGB=[0, 0, 1], lineWidth=1,
                    replaceItemUniqueId=None,
                    lifeTime=0)
                is_fallen |= step_is_fallen
                executed_traj.append(np.concatenate(
                    [cur_bottle_pos, cur_joints]))

            is_success = planner.reached_goal(cur_bottle_pos)
            print("Exec #%d: fell: %d, success: %d" %
                  (exec_i, is_fallen, is_success))
            print()
            success_count += is_success
            fall_count += is_fallen

            executed_traj = np.vstack(executed_traj)
            difference = np.linalg.norm(
                executed_traj[:, :3] - state_path[:, :3], axis=1)
            indices = np.where(difference > 0)[0]

        print("Fall Rate: %.2f, success rate: %.2f" % (
            fall_count / float(len(exec_params_set)),
            success_count / float(len(exec_params_set))
        ))


def piecewise_execution_replan_helper(planner: NaivePlanner, env: Environment, exec_params, res_fname, max_time):
    reached_goal = False
    is_fallen = False
    plan_count = 0
    final_state_path = []
    final_policy = []
    final_node_path = []
    goal = planner.goal

    if max_time == -1:
        max_time = sys.maxsize

    try:
        with helpers.time_limit(max_time):
            while not reached_goal and not is_fallen:
                # plan from current position
                plan_count += 1
                state_path, policy, node_path = planner.plan()
                final_state_path += state_path
                final_policy += policy
                final_node_path += node_path

                bottle_pos = planner.bottle_pos_from_state(planner.start)
                init_joints = planner.joint_pose_from_state(planner.start)
                env.arm.reset(init_joints)
                bottle_ori = np.array([0, 0, 0, 1])
                cur_joints = init_joints
                cur_bottle_pos = bottle_pos.copy()
                cur_bottle_ori = bottle_ori.copy()

                # execute found plan, terminate if bottle fell, replan if failed to reach goal
                for step in range(len(policy)):
                    step_is_fallen, is_collision, cur_bottle_pos, cur_bottle_ori, cur_joints = (
                        env.run_sim(policy[step], exec_params,
                                    cur_joints, cur_bottle_pos, cur_bottle_ori)
                    )
                    is_fallen |= step_is_fallen

                    if is_fallen:
                        break

                # check at end of execution if reached goal
                if not is_fallen:
                    reached_goal = planner.reached_goal(cur_bottle_pos)

    except helpers.TimeoutException:
        pass

    # save whatever plan was found, if any
    np.savez("%s" % res_fname,
             state_path=final_state_path, policy=final_policy, node_path=final_node_path)

    return reached_goal, is_fallen, plan_count


def piecewise_execution_replan(planner: NaivePlanner, env: Environment,
                               exec_params_set,
                               res_fname,
                               max_time):
    orig_start = planner.start
    orig_goal = planner.goal

    for exec_i, exec_params in enumerate(exec_params_set):
        planner.start = orig_start
        planner.goal = orig_goal
        piecewise_execution_replan_helper(planner, env,
                                          exec_params=exec_params,
                                          res_fname=res_fname + "_exec_%d" % exec_i,
                                          max_time=max_time)


def main():
    VISUALIZE = True
    REPLAY_RESULTS = True
    LOAD_SAVED = REPLAY_RESULTS
    LOGGING = False
    GRAVITY = -9.81

    max_time_s = 5 * 60  # -1 for no time limit
    fall_thresh = 0.1
    dx = dy = dz = 0.1
    dist_thresh = dx
    eps = 5
    da_rad = 8 * math.pi / 180.0
    num_sims_per_action = 10
    num_exec_tests = 10
    load_saved_params = True

    pi = 1
    sample_strat = "_sample_2"  # sample_1 is bimodal, sample_2 is unimodal centered at high values
    # sample_strat = ""
    if pi == 0:
        planner_folder = "results"
    else:
        planner_folder = "avg_results_%.2f%s" % (fall_thresh, sample_strat)
    if not os.path.exists(planner_folder):
        os.mkdir(planner_folder)

    # if REPLAY_RESULTS:
    #     sys.stdout = open("%s/results.txt" % planner_folder, "w")
    # else:
    #     sys.stdout = open("%s/output.txt" % planner_folder, "w")

    if VISUALIZE:
        p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, GRAVITY)
    planeId = p.loadURDF(Environment.plane_urdf_filepath,
                         basePosition=[0, 0, 0])
    kukaId = p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0])
    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, "avg_plan_success.mp4")

    if VISUALIZE:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, cameraYaw=145, cameraPitch=-10,
            cameraTargetPosition=[0, 0, 0])

    bottle_start_pos = np.array(
        [0.5, 0.5, 0]).astype(float)
    bottle_goal_pos = np.array([0.2, 0.6, 0]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)

    # starting end-effector pos, not base pos
    EE_start_pos = np.array([0.5, 0.3, 0.2])
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    arm = Arm(EE_start_pos=EE_start_pos,
              start_ori=base_start_ori,
              kukaId=kukaId)
    start_joints = arm.joint_pose

    env = Environment(arm, bottle, is_viz=VISUALIZE)
    # Normal distribution of internal bottle params
    # normal distrib for bottle friction
    env.min_fric = 0.15
    env.max_fric = 0.2
    # normal distrib for bottle fill proportion
    env.min_fill = env.bottle.min_fill
    env.max_fill = 1.0
    env.set_distribs()
    
    start = np.concatenate(
        [bottle_start_pos, start_joints])
    # goal joints are arbitrary and populated later in planner
    goal = np.concatenate(
        [bottle_goal_pos, [0] * arm.num_joints])
    xbounds = [-0.4, -0.9]
    ybounds = [-0.1, -0.9]

    # if  the below isn't true, you're expecting bottle to fall in exactly
    # the same state bin as the goal
    assert (dist_thresh <= dx)

    with open("filtered_start_goals.obj", "rb") as f:
        start_goals = pickle.load(f)

    # (startb, goalb, start_joints) = start_goals[5]
    # start_goals[5] = (startb, [0.40,
    #                            0.4732376897585052, 0.03803531856053188], start_joints)

    # (startb, goalb, start_joints) = start_goals[8]
    # start_goals[8] = (startb, [0.34, 0.46, 0], start_joints)
    # for i, pair in enumerate(start_goals):
    #     (startb, goalb, start_joints) = pair
    #     startb[-1] = bottle.PLANE_OFFSET
    #     start_goals[i] = (startb, goalb, start_joints)

    # ignore_list = {8, 3, 5}
    # new_start_goals = []
    # for i in range(len(start_goals)):
    #     if i not in ignore_list:
    #         new_start_goals.append(start_goals[i])
    # with open("filtered_start_goals.obj", "wb") as f:
    #     pickle.dump(new_start_goals, f)
    # for i, (startb, goalb, start_joints) in enumerate(start_goals):
    #     print(startb, goal)
    #     if np.linalg.norm(np.array(startb) - np.array(goalb)) <= dist_thresh:
    #         print("%d TOO EASY" % i)

    # exit()

    # change bottle with randomly sampled radius, height, fill, friction
    # radius = 0.025
    # height = 0.21
    # fill_prop = 0.3
    # fric = env.max_fric
    # new_bottle = Bottle(start_pos=bottle_start_pos,
    #                     start_ori=bottle_start_ori,
    #                     fill_prop=fill_prop,
    #                     fric=fric)
    # env.bottle = new_bottle

    with open("sim_params_set.obj", "rb") as f:
        exec_plan_params = pickle.load(f)
        exec_params_set = exec_plan_params["exec_params_set"]
        plan_params_sets = exec_plan_params["plan_params_sets"]

    # generate new planning env parameters, not exec
    if not load_saved_params:
        env.min_fric = 0.15
        env.max_fric = 0.2
        env.set_distribs()
        plan_params_sets = env.gen_random_env_param_set(
            num=int(num_sims_per_action))

        # env.min_fric = 0.05
        # env.max_fric = 0.1
        # env.set_distribs()
        # plan_params_sets += env.gen_random_env_param_set(
        #     num=int(num_sims_per_action//2))


    single_planner = NaivePlanner(start, goal, env, xbounds, ybounds,
                                  sim_params_set=plan_params_sets,
                                  dist_thresh=dist_thresh, eps=eps, da_rad=da_rad,
                                  dx=dx, dy=dy, dz=dz, visualize=VISUALIZE,
                                  sim_mode=NaivePlanner.SINGLE, fall_thresh=fall_thresh)
    single_planner.sim_params_set[single_planner.param_index] = EnvParams(0.70, 0.08, 0.33, 0.11)

    avg_planner = NaivePlanner(start, goal, env, xbounds, ybounds,
                               sim_params_set=plan_params_sets,
                               dist_thresh=dist_thresh, eps=eps, da_rad=da_rad,
                               dx=dx, dy=dy, dz=dz, visualize=VISUALIZE, sim_mode=NaivePlanner.AVG,
                               fall_thresh=fall_thresh)

    planner = single_planner if pi == 0 else avg_planner

    with open("%s/sim_params_set.obj" % planner_folder, "wb") as f:
        exec_plan_params = dict(exec_params_set=exec_params_set,
                                plan_params_sets=plan_params_sets)
        pickle.dump(exec_plan_params, f)

    # exec_params_set = plan_params_sets
    exec_params_set = exec_params_set[-1:]
    print("exec_params_set:")
    for v in exec_params_set:
        print(v)
    # avg_fric, avg_fill = 0, 0
    # for param in avg_planner.sim_params_set:
    #     print(param.bottle_fric)
    #     avg_fric += param.bottle_fric
    #     avg_fill += param.bottle_fill

    # print(avg_fric / 10.0)
    # print(avg_fill / 10.0)
    # exit()

    # pick which planner to use
    # use_single = False
    # if use_single:
    #     planner = single_planner
    #     planner_folder = "results"
    # else:
    #     planner = avg_planner
    #     planner_folder = "avg_results"

    # start_goal_idx = 10
    # res_fname = "%s/results_%d" % (planner_folder, start_goal_idx)
    # (startb, goalb, start_joints) = start_goals[start_goal_idx]

    # try:
    #     results = np.load("%s.npz" % res_fname, allow_pickle=True)
    # except:
    #     print("Results: %s not found!" % res_fname)
    #     return

    # policy = results["policy"]
    # state_path = results["state_path"]
    # idx = 23
    # startb, start_joints = state_path[idx, :3], state_path[idx, 3:]

    # start_state = helpers.bottle_EE_to_state(
    #     bpos=startb, arm=arm, joints=start_joints)
    # goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=arm)
    # planner.start = start_state
    # planner.goal = goal_state
    # # piecewise_execution
    # piecewise_execution(planner, env,
    #                     exec_params_set=planner.sim_params_set,
    #                     load_saved=LOAD_SAVED,
    #                     play_results=REPLAY_RESULTS,
    #                     res_fname=res_fname)

    plan_to_time = [0, 0]
    # for pi in range(1):
    print("Planner: %s" % planner_folder)

    # targets = list(range(5, 12))
    targets = list(range(0, 12))
    for start_goal_idx in targets:
        print("Start goal idx: %d" % start_goal_idx)
        res_fname = "%s/results_%d" % (planner_folder, start_goal_idx)
        (startb, goalb, start_joints) = start_goals[start_goal_idx]
        start_state = helpers.bottle_EE_to_state(
            bpos=startb, arm=arm, joints=start_joints)
        goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=arm)
        planner.start = start_state
        planner.goal = goal_state

        # results = np.load("%s.npz" % res_fname, allow_pickle=True)
        # for i, a in enumerate(results["policy"]):
        #     print('i: %d' % i)
        #     print(np.array2string(a[0], precision=2))
        # results = np.load("%s.npz" % res_fname, allow_pickle=True)
        # planner.start = results["node_path"][10].state
        # print(results["policy"][10])

        start_time = time.time()
        try:
            with helpers.time_limit(20*60):
                piecewise_execution(planner, env,
                                    exec_params_set=exec_params_set,
                                    load_saved=LOAD_SAVED,
                                    play_results=REPLAY_RESULTS,
                                    res_fname=res_fname)
        except helpers.TimeoutException:
            pass
        # max_time=max_time_s)

        time_taken = time.time() - start_time
        plan_to_time[pi] += time_taken
        print("time taken: %.3f" % time_taken, flush=True)

    print("Average time: ", plan_to_time)


if __name__ == "__main__":
    main()
