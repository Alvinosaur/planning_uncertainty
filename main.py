import pybullet as p
import pybullet_data
import math
import numpy as np
import pickle

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace
from naive_joint_space_planner import NaivePlanner
import experiment_helpers as helpers


def direct_plan_execution(planner, env, replay_saved=False, visualize=False):
    if not replay_saved:
        state_path, policy = planner.plan()
        np.savez("results", state_path=state_path, policy=policy)

    else:
        results = np.load("results.npz")
        policy = results["policy"]
        state_path = results["state_path"]

    if visualize:
        # print(policy)
        bottle_pos = planner.bottle_pos_from_state(planner.start)
        bottle_ori = np.array([0, 0, 0, 1])
        for dq in policy:
            # run deterministic simulation for now
            # init_joints not passed-in because current joint state
            # maintained by simulator
            # print(bottle_pos)
            # print(bottle_ori)
            trans_cost, bottle_pos, bottle_ori, _ = env.run_sim(
                action=dq, bottle_pos=bottle_pos, bottle_ori=bottle_ori)

            link_positions = env.arm.get_link_positions()
            EE_pos = np.array(link_positions[-1])
            print(EE_pos)

    elif not visualize and replay_saved:
        print("Trying to playback plan without visualizing!")
        exit()


# def interleaved_replanning_execution():
#     planner = NaivePlanner(start, goal, env, xbounds,
#                            ybounds, dist_thresh, eps)
#     state_path, policy = planner.plan()


def main():
    VISUALIZE = True
    REPLAY_RESULTS = False
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
            p.STATE_LOGGING_VIDEO_MP4, "cool.mp4")

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
    eps = 2
    da_rad = 8 * math.pi / 180.0

    # run planner and visualize result
    planner = NaivePlanner(start, goal, env, xbounds,
                           ybounds, dist_thresh, eps, da_rad=da_rad,
                           dx=dx, dy=dy, dz=dz, visualize=VISUALIZE)

    save_new_start_goals = False
    if save_new_start_goals:
        start_goals = helpers.generate_random_start_goals(
            arm=arm, bottle=bottle, num_pairs=50)
        with open("start_goals.obj", "wb") as f:
            pickle.dump(start_goals, f)
    else:
        with open("start_goals.obj", "rb") as f:
            start_goals = pickle.load(f)

    for (startb, goalb, start_EE) in start_goals:
        start_state = helpers.bottle_EE_to_state(
            bpos=startb, arm=arm, EE_pos=start_EE)
        goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=arm)
        planner.start = start_state
        planner.goal = goal_state

        direct_plan_execution(
            planner, env, replay_saved=REPLAY_RESULTS, visualize=VISUALIZE)
    # s1 = np.array([-0.50, -0.50, 0.04, 0.00, 0.00, -0.00, 1.00,
    #                0.51, 2.09, -0.11, 0.45, -0.14, 2.08, -0.91])
    # s2 = np.array([-0.50, -0.50, 0.04, -0.00, 0.00, -0.00, 1.00,
    #                0.51, 2.09, -0.11, 0.46, -0.14, 2.08, -0.93])
    # print(planner.state_to_key(s1))
    # print(planner.state_to_key(s2))


if __name__ == "__main__":
    main()
    # test_quaternion_discretization()
    # test_state_indexing()
