import pybullet as p
import pybullet_data
import math
import numpy as np
from contextlib import contextmanager
import signal

from sim_objects import Bottle, Arm
from environment import Environment

DEG_TO_RAD = math.pi / 180.0


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Enforces max allotted time for planning, execution, and replanning
    Args:
        seconds ([type]): [description]
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    if seconds != -1:
        signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def calc_EE_from_direction(max_reach, theta, psi, z_offset):
    z = z_offset + max_reach * math.sin(psi)
    r = max_reach * math.cos(psi)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return np.array([x, y, z])


def bottle_EE_to_state(bpos, arm: Arm, joints=None, EE_pos=None):
    # find arm joint angles that correspond to this EE pose
    if EE_pos is None and joints is None:
        joints = arm.default_joint_pose
    if joints is None:
        joints = arm.get_target_joints(target_EE_pos=EE_pos, angle=0)

    state = np.hstack([bpos, joints])
    return state


def generate_random_start_goals(env: Environment, bottle: Bottle, num_pairs=50):
    max_horiz_dist = 0.9 * env.arm.MAX_HORIZ_REACH
    min_horiz_dist = 0.2 * env.arm.MAX_HORIZ_REACH
    empty_action = (np.zeros(env.arm.num_joints), 50)

    min_start_goal_dist = 0.3

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
            res = env.run_sim(action=empty_action, init_joints=rand_joints)
            rand_joints = res[-1]

            start = [start_x, start_y] + [bottle.INIT_PLANE_OFFSET]
            goal = [goal_x, goal_y] + [0]
            start_goal_pairs.append((start, goal, rand_joints))
            count += 1

    return start_goal_pairs
