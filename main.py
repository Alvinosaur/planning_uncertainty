import pybullet as p
import pybullet_data
import math
import numpy as np
import pickle

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace, EnvParams
from naive_joint_space_planner import NaivePlanner
import experiment_helpers as helpers


def policy_to_full_traj(init_joints, policy):
    cur_joints = np.copy(init_joints)
    piecewise_trajs = []
    for (dq_vec, num_iters) in policy:
        target_joints = cur_joints + dq_vec
        traj = np.linspace(
            start=cur_joints, stop=target_joints, num=num_iters)
        piecewise_trajs.append(traj)
        cur_joints = target_joints
    try:
        full_arm_traj = np.vstack(piecewise_trajs)
    except:
        return []
    return full_arm_traj


def direct_plan_execution(planner: NaivePlanner, env: Environment,
                          exec_params_set,
                          load_saved=False, play_results=False,
                          res_fname="results"):
    if not load_saved:
        state_path, policy = planner.plan()
        np.savez("%s" % res_fname,
                 state_path=state_path, policy=policy)

    else:
        try:
            results = np.load("%s.npz" % res_fname, allow_pickle=True)
        except:
            print("Results: %s not found!" % res_fname)
            return

        policy = results["policy"]
        print("Policy:")
        for dq_vec, num_iters in policy:
            print(num_iters, planner.state_to_str(dq_vec))
        state_path = results["state_path"]

    if play_results:
        # print(policy)
        bottle_pos = planner.bottle_pos_from_state(planner.start)
        init_joints = planner.joint_pose_from_state(planner.start)
        env.arm.reset(init_joints)
        bottle_ori = np.array([0, 0, 0, 1])

        bottle_goal = planner.bottle_pos_from_state(planner.goal)
        env.goal_line_id = env.draw_line(
            lineFrom=bottle_goal,
            lineTo=bottle_goal + np.array([0, 0, 1]),
            lineColorRGB=[0, 0, 1], lineWidth=1,
            replaceItemUniqueId=env.goal_line_id,
            lifeTime=0)

        full_arm_traj = policy_to_full_traj(init_joints, policy)
        fall_count = 0
        success_count = 0
        for i, exec_params in enumerate(exec_params_set):
            print("New Test with params: %s" % exec_params)
            is_fallen, is_collision, new_bottle_pos, new_bottle_ori, new_joint_pos = (
                env.simulate_plan(joint_traj=full_arm_traj,
                                  bottle_pos=bottle_pos,
                                  bottle_ori=bottle_ori,
                                  sim_params=exec_params))
            is_success = planner.reached_goal(new_bottle_pos)
            print("Exec #%d: fell: %d, success: %d" %
                  (i, is_fallen, is_success))
            print()
            success_count += is_success
            fall_count += is_fallen

        print("Fall Rate: %.2f, success rate: %.2f" % (
            fall_count / float(len(exec_params_set)),
            success_count / float(len(exec_params_set))
        ))


def main():
    VISUALIZE = True
    REPLAY_RESULTS = False
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
                         basePosition=[0, 0, -0.01])
    kukaId = p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0])
    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, "avg_plan_success.mp4")

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

    for param in plan_params_sets:
        print(param)

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
                               num_rand_samples=num_sims_per_action)

    # exec_params_set = plan_params_sets[0]
    single_planner.sim_params_set = plan_params_sets
    avg_planner.sim_params_set = plan_params_sets

    # pick which planner to use
    use_single = False
    if use_single:
        planner = single_planner
        planner_folder = "results"
    else:
        planner = avg_planner
        planner_folder = "avg_results"

    start_goal_idx = 11
    (startb, goalb, start_joints) = start_goals[start_goal_idx]
    start_joints = [1.15, 1.48, 1.70, 1.03, 2.77, 2.09, 3.05]
    start_state = helpers.bottle_EE_to_state(
        bpos=startb, arm=arm, joints=start_joints)
    goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=arm)
    planner.start = start_state
    planner.goal = goal_state
    direct_plan_execution(planner, env,
                          exec_params_set=exec_params_set,
                          load_saved=LOAD_SAVED,
                          play_results=REPLAY_RESULTS,
                          res_fname="%s/results_%d" % (planner_folder, start_goal_idx))

    # for pi, planner in enumerate([avg_planner, single_planner]):
    #     if pi == 1:
    #         planner_folder = "results"
    #     else:
    #         planner_folder = "avg_results"

    #     for start_goal_idx in range(12):
    #         print("Start goal idx: %d" % start_goal_idx)
    #         (startb, goalb, start_joints) = start_goals[start_goal_idx]
    #         start_state = helpers.bottle_EE_to_state(
    #             bpos=startb, arm=arm, joints=start_joints)
    #         goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=arm)
    #         planner.start = start_state
    #         planner.goal = goal_state
    #         direct_plan_execution(planner, env,
    #                               exec_params_set=exec_params_set,
    #                               load_saved=LOAD_SAVED,
    #                               play_results=REPLAY_RESULTS,
    #                               res_fname="%s/results_%d" % (planner_folder, start_goal_idx))


if __name__ == "__main__":
    main()
    # test_quaternion_discretization()
    # test_state_indexing()
