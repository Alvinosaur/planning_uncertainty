import pybullet as p
import pybullet_data
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import re
import time
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
            # print(bottle_pos, bottle_ori, flush=True)
            is_fallen = False
            executed_traj = []

            # if exec_i != 1:
            #     continue
            for step in range(len(policy)):
                # print(f"Step {step}, " + str(node_path[step]))
                # print(f"step: {step}")
                # print("state:")
                # print(np.concatenate([cur_bottle_pos, cur_joints]))
                # print(cur_bottle_ori)
                # action, sim_params: EnvParams, init_joints=None, bottle_pos=None, bottle_ori=None
                # print(node_path[step])
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
                # if step >= 10:
                #     state = node_path[step+1].state
                #     cur_bottle_pos = planner.bottle_pos_from_state(state)
                #     cur_bottle_ori = node_path[step+1].bottle_ori
                #     cur_joints = planner.joint_pose_from_state(state)
                env.goal_line_id = env.draw_line(
                    lineFrom=bottle_goal,
                    lineTo=bottle_goal + np.array([0, 0, 1]),
                    lineColorRGB=[0, 0, 1], lineWidth=1,
                    replaceItemUniqueId=None,
                    lifeTime=0)
                # env.draw_line(
                #     lineFrom=bottle_goal,
                #     lineTo=bottle_goal + np.array([0, 0, 1]),
                #     lineColorRGB=[0, 0, 1], lineWidth=1,
                #     lifeTime=0)
                # print("next state:")
                # print(np.concatenate([cur_bottle_pos, cur_joints]))
                # print(policy[step])
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
            # print(difference)
            indices = np.where(difference > 0)[0]
            # print(indices)
            # print(executed_traj[indices[0]-1:indices[0]+1, :])
            # print(state_path[indices[0]-1:indices[0]+1, :])
            # # ith policy led to ith state
            # print(policy[indices[0]])

            # fig, plots = plt.subplots(3, 4)
            # for i in range(planner.env.arm.num_joints):
            #     r = i // 4 + 1
            #     c = i % 4
            #     plots[r][c].plot(state_path[:, i+3], label="State traj")
            #     plots[r][c].plot(executed_traj[:, i+3], label="Executed traj")
            #     plots[r][c].set_title("Joint %d" % i)

            # xyz_labels = "xyz"
            # for i in range(3):
            #     r = 0
            #     c = i
            #     plots[r][c].plot(state_path[:, i], label="State traj")
            #     plots[r][c].plot(executed_traj[:, i], label="Executed traj")
            #     plots[r][c].set_title("Bottle pos %s" % xyz_labels[i])

            # plots[1][2].legend()
            # plt.show()

        print("Fall Rate: %.2f, success rate: %.2f" % (
            fall_count / float(len(exec_params_set)),
            success_count / float(len(exec_params_set))
        ))


def main():
    VISUALIZE = False
    REPLAY_RESULTS = True
    LOAD_SAVED = REPLAY_RESULTS
    LOGGING = False
    GRAVITY = -9.81
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

    # bottle
    # bottle_start_pos = np.array(
    #     [-0, -0.6, Bottle.INIT_PLANE_OFFSET]).astype(float)
    # bottle_goal_pos = np.array([-0.6, -0.2, 0]).astype(float)
    bottle_start_pos = np.array(
        [0.5, 0.5, 0]).astype(float)
    bottle_goal_pos = np.array([0.2, 0.6, 0]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)

    # starting end-effector pos, not base pos
    # NOTE: just temporarily setting arm to starting bottle position with some offset
    # offset = -np.array([0.05, 0, 0])
    # EE_start_pos = bottle_start_pos + offset
    EE_start_pos = np.array([0.5, 0.3, 0.2])
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    arm = Arm(EE_start_pos=EE_start_pos,
              start_ori=base_start_ori,
              kukaId=kukaId)
    start_joints = arm.joint_pose

    env = Environment(arm, bottle, is_viz=VISUALIZE)
    start = np.concatenate(
        [bottle_start_pos, start_joints])
    # goal joints are arbitrary and populated later in planner
    goal = np.concatenate(
        [bottle_goal_pos, [0] * arm.num_joints])
    xbounds = [-0.4, -0.9]
    ybounds = [-0.1, -0.9]
    dx = dy = dz = 0.1
    dist_thresh = dx
    # if  the below isn't true, you're expecting bottle to fall in exactly
    # the same state bin as the goal
    assert(dist_thresh <= dx)
    eps = 5
    da_rad = 8 * math.pi / 180.0

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

    # run planner and visualize result
    num_sims_per_action = 20
    plan_params_sets = env.gen_random_env_param_set(
        num=num_sims_per_action)
    num_exec_tests = 10
    exec_params_set = env.gen_random_env_param_set(
        num=num_exec_tests)

    load_saved_params = True
    if load_saved_params:
        with open("sim_params_set.obj", "rb") as f:
            exec_plan_params = pickle.load(f)
            exec_params_set = exec_plan_params["exec_params_set"]
            plan_params_sets = exec_plan_params["plan_params_sets"]
    else:
        with open("sim_params_set.obj", "wb") as f:
            exec_plan_params = dict(exec_params_set=exec_params_set,
                                    plan_params_sets=plan_params_sets)
            pickle.dump(exec_plan_params, f)

    single_planner = NaivePlanner(start, goal, env, xbounds,
                                  ybounds, dist_thresh, eps, da_rad=da_rad,
                                  dx=dx, dy=dy, dz=dz, visualize=VISUALIZE, sim_mode=NaivePlanner.SINGLE)
    avg_planner = NaivePlanner(start, goal, env, xbounds,
                               ybounds, dist_thresh, eps, da_rad=da_rad,
                               dx=dx, dy=dy, dz=dz, visualize=VISUALIZE, sim_mode=NaivePlanner.AVG,
                               num_rand_samples=num_sims_per_action, fall_thresh=0.1)

    # exec_params_set = plan_params_sets
    single_planner.sim_params_set = plan_params_sets
    avg_planner.sim_params_set = plan_params_sets
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
    for pi in range(1):
        if pi == 1:
            planner = single_planner
            planner_folder = "results"
        else:
            planner = avg_planner
            planner_folder = "avg_results"

        if REPLAY_RESULTS:
            sys.stdout = open("%s/results.txt" % planner_folder, "w")
        else:
            sys.stdout = open("%s/output.txt" % planner_folder, "w")
        print("Planner: %s" % planner_folder)

        for start_goal_idx in range(12):
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
            piecewise_execution(planner, env,
                                exec_params_set=exec_params_set,
                                load_saved=LOAD_SAVED,
                                play_results=REPLAY_RESULTS,
                                res_fname=res_fname)

            time_taken = time.time() - start_time
            plan_to_time[pi] += time_taken
            print("time taken: %.3f" % time_taken, flush=True)

    print("Average time: ", plan_to_time)


if __name__ == "__main__":
    main()
    # test_quaternion_discretization()
    # test_state_indexing()
