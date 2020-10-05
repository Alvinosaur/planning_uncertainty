import pybullet as p
import pybullet_data
import math
import numpy as np
import pickle

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace
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
                          replay_saved=False, visualize=False,
                          res_fname="results"):
    if not replay_saved:
        state_path, policy = planner.plan()
        np.savez("results/%s" % res_fname,
                 state_path=state_path, policy=policy)

    else:
        results = np.load("results/%s.npz" % res_fname, allow_pickle=True)
        policy = results["policy"]
        state_path = results["state_path"]

    if visualize:
        # print(policy)
        bottle_pos = planner.bottle_pos_from_state(planner.start)
        init_joints = planner.joint_pose_from_state(planner.start)
        env.arm.reset(init_joints)
        bottle_ori = np.array([0, 0, 0, 1])

        full_arm_traj = policy_to_full_traj(init_joints, policy)
        is_fallen, is_collision, bottle_pos, bottle_ori, joint_pos = (
            env.simulate_plan(joint_traj=full_arm_traj, bottle_pos=bottle_pos,
                              bottle_ori=bottle_ori))

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
    eps = 5
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
            # print(start_goals)

    for i, (startb, goalb, start_EE) in enumerate(start_goals):
        if not i >= 2:
            continue

        start_state = helpers.bottle_EE_to_state(
            bpos=startb, arm=arm, EE_pos=start_EE)
        goal_state = helpers.bottle_EE_to_state(bpos=goalb, arm=arm)
        planner.start = start_state
        planner.goal = goal_state

        direct_plan_execution(planner, env,
                              replay_saved=REPLAY_RESULTS,
                              visualize=VISUALIZE,
                              res_fname="results_%d" % i)
        print("FOUND PLAN FOR %d" % i)


if __name__ == "__main__":
    main()
    # test_quaternion_discretization()
    # test_state_indexing()
