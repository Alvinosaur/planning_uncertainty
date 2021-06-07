import pybullet as p
import pybullet_data
import math
import numpy as np
import pickle
import os

from sim_objects import Bottle, Arm
from environment import Environment, ActionSpace, EnvParams, StateTuple

DEG_TO_RAD = math.pi / 180.0


def generate_random_start_goals(env: Environment, bottle: Bottle, num_pairs=50):
    max_horiz_dist = 0.7 * env.arm.MAX_HORIZ_REACH
    min_horiz_dist = 0.2 * env.arm.MAX_HORIZ_REACH
    empty_action = (np.zeros(env.arm.num_joints), 50)

    min_start_goal_dist = 0.4

    # arbitrary
    single_param = env.gen_random_env_param_set(num=1)[0]

    start_goal_pairs = []
    count = 0
    while count < num_pairs:
        goal_bottle_angle = np.random.uniform(
            low=30, high=80)
        start_bottle_angle = np.random.uniform(
            low=30, high=80)
        arm_EE_psi = np.random.uniform(
            low=-5, high=90)

        # needs to behind bottle generally facing towards goal due to simple
        # heuristic
        max_offset = 30
        arm_EE_theta = np.random.uniform(
            low=start_bottle_angle - max_offset,
            high=start_bottle_angle + max_offset)
        rand_joints = np.random.uniform(
            low=env.arm.ll, high=env.arm.ul) % (2 * math.pi)
        rand_joints[0] = (arm_EE_theta * DEG_TO_RAD) % (2 * math.pi)
        rand_joints[1] = (arm_EE_psi * DEG_TO_RAD) % (2 * math.pi)

        goal_dist = np.random.uniform(
            low=0.3, high=1.0) * max_horiz_dist
        start_dist = np.random.uniform(
            low=0.6, high=1.0) * max_horiz_dist

        start_x = start_dist * math.cos(start_bottle_angle * DEG_TO_RAD)
        start_y = start_dist * math.sin(start_bottle_angle * DEG_TO_RAD)
        goal_x = goal_dist * math.cos(goal_bottle_angle * DEG_TO_RAD)
        goal_y = goal_dist * math.sin(goal_bottle_angle * DEG_TO_RAD)

        dist = ((start_x - goal_x) ** 2 + (start_y - goal_y) ** 2) ** 0.5
        if dist < min_start_goal_dist:
            continue
        else:
            # make sure initial arm configuration is stable
            start = [start_x, start_y] + [bottle.PLANE_OFFSET]
            goal = [goal_x, goal_y] + [0]
            state_tuple = StateTuple(bottle_pos=start, bottle_ori=[0, 0, 0, 1],
                                     joints=rand_joints)
            res = env.run_sim(action=empty_action, state=state_tuple, sim_params=single_param)
            start_goal_pairs.append((start, goal, res.joint_pose))
            count += 1

    return start_goal_pairs


if __name__ == "__main__":
    fname = "test_start_goals.obj"
    p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(Environment.plane_urdf_filepath, basePosition=[0, 0, 0])

    # Create Arm, Bottle and set up Environment
    bottle = Bottle()
    arm = Arm(kuka_id=p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0]))
    arm.set_general_max_reach(all_contact_heights=np.linspace(start=0.2, stop=1.01, num=5) * bottle.height)
    env = Environment(arm, bottle, is_viz=False)

    start_goals = generate_random_start_goals(env, bottle, num_pairs=50)
    with open(os.path.join(fname), "wb") as outfile:
        pickle.dump(start_goals, outfile)
