import pybullet as p
import pybullet_data
import math
import numpy as np

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace
from naive_joint_space_planner import NaivePlanner


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
        bottle_ori = planner.bottle_ori_from_state(planner.start)
        for dq in policy:
            # run deterministic simulation for now
            # init_joints not passed-in because current joint state
            # maintained by simulator
            # print(bottle_pos)
            # print(bottle_ori)
            trans_cost, bottle_pos, bottle_ori = env.run_sim(
                action=dq, bottle_pos=bottle_pos, bottle_ori=bottle_ori)

    elif not visualize and replay_saved:
        print("Trying to playback plan without visualizing!")
        exit()


# def interleaved_replanning_execution():
#     planner = NaivePlanner(start, goal, env, xbounds,
#                            ybounds, dist_thresh, eps)
#     state_path, policy = planner.plan()


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
            p.STATE_LOGGING_VIDEO_MP4, "fully_functional.mp4")

    # bottle
    bottle_start_pos = np.array(
        [0.5, 0.5, Bottle.INIT_PLANE_OFFSET]).astype(float)
    bottle_goal_pos = np.array([0.2, 0.6, 0]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)

    if VISUALIZE:
        p.addUserDebugLine(bottle_goal_pos,
                           bottle_goal_pos +
                           np.array([0, 0, 0.5]),
                           [0, 0, 1], 1,
                           0)

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

    N = 500
    env = Environment(arm, bottle, is_viz=VISUALIZE, N=N)
    start = np.concatenate(
        [bottle_start_pos,  bottle_start_ori, start_joints])
    # goal joints are arbitrary and populated later in planner
    goal = np.concatenate(
        [bottle_goal_pos,  bottle_start_ori, [0]*arm.num_joints])
    xbounds = [0.4, 0.9]
    ybounds = [0.1, 0.9]
    dist_thresh = 1e-1
    eps = 1

    # run planner and visualize result
    planner = NaivePlanner(start, goal, env, xbounds,
                           ybounds, dist_thresh, eps)
    direct_plan_execution(
        planner, env, replay_saved=REPLAY_RESULTS, visualize=VISUALIZE)


if __name__ == "__main__":
    main()
    # test_quaternion_discretization()
    # test_state_indexing()
