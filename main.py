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
    full_arm_traj = np.vstack(piecewise_trajs)
    return full_arm_traj


def direct_plan_execution(planner: NaivePlanner, env: Environment,
                          exec_params_set,
                          replay_saved=False, visualize=False,
                          res_fname="results"):
    if not replay_saved:
        state_path, policy = planner.plan()
        np.savez("%s" % res_fname,
                 state_path=state_path, policy=policy)

    else:
        results = np.load("%s.npz" % res_fname, allow_pickle=True)
        policy = results["policy"]
        state_path = results["state_path"]

    if visualize:
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
        is_fallen, is_collision, bottle_pos, bottle_ori, joint_pos = (
            env.simulate_plan(joint_traj=full_arm_traj, bottle_pos=bottle_pos,
                              bottle_ori=bottle_ori,
                              sim_params=exec_params_set))

    elif not visualize and replay_saved:
        print("Trying to playback plan without visualizing!")
        exit()


def main():
    VISUALIZE = True
    REPLAY_RESULTS = True
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
            p.STATE_LOGGING_VIDEO_MP4, "single_plan.mp4")

    # bottle
    # bottle_start_pos = np.array(
    #     [-0, -0.6, Bottle.INIT_PLANE_OFFSET]).astype(float)
    # bottle_goal_pos = np.array([-0.6, -0.2, 0]).astype(float)
    bottle_start_pos = np.array(
        [0.5, 0.5, Bottle.INIT_PLANE_OFFSET]).astype(float)
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
    num_sims_per_action = 10
    plan_params_sets = env.gen_random_env_param_set(
        num=num_sims_per_action)

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

    exec_params_set = EnvParams(bottle_fill=1, bottle_fric=env.max_fric,
                                bottle_fill_prob=0, bottle_fric_prob=0)
    # exec_params_set = plan_params_sets[0]
    single_planner.sim_params_set = plan_params_sets
    avg_planner.sim_params_set = plan_params_sets

    (startb, goalb, start_joints) = start_goals[2]
    start_state = helpers.bottle_EE_to_state(
        bpos=startb, arm=arm, joints=start_joints)
    goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=arm)
    planner_folder = "results"
    single_planner.start = start_state
    single_planner.goal = goal_state
    direct_plan_execution(single_planner, env,
                          exec_params_set=exec_params_set,
                          replay_saved=REPLAY_RESULTS,
                          visualize=VISUALIZE,
                          res_fname="%s/results_%d" % (planner_folder, 2))


if __name__ == "__main__":
    main()
    # test_quaternion_discretization()
    # test_state_indexing()
