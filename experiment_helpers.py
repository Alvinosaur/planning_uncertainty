import pybullet as p
import pybullet_data
import math
import numpy as np


from sim_objects import Bottle, Arm

DEG_TO_RAD = math.pi / 180.0


def calc_EE_from_direction(max_reach, theta, psi, z_offset):
    z = z_offset + max_reach * math.sin(psi)
    r = max_reach * math.cos(psi)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return np.array([x, y, z])


def bottle_EE_to_state(bpos, arm: Arm, EE_pos=None):
    # find arm joint angles that correspond to this EE pose
    if EE_pos is None:
        joints = arm.default_joint_pose
    else:
        joints = arm.get_target_joints(target_EE_pos=EE_pos, angle=0)
    state = np.hstack([bpos, joints])
    print(bpos, joints)
    return state


def generate_random_start_goals(arm: Arm, bottle: Bottle, num_pairs=50):
    valid_bottle_positions = []
    max_horiz_dist = arm.MAX_HORIZ_REACH
    min_horiz_dist = 0.2 * arm.MAX_HORIZ_REACH
    # focus on quadrant 1, all other quadrants are just reflections
    for x in np.linspace(start=min_horiz_dist, stop=max_horiz_dist, num=10):
        for y in np.linspace(start=min_horiz_dist, stop=max_horiz_dist, num=10):
            valid_bottle_positions.append([x, y])

    start_goal_pairs = []
    goal_bottle_angles = np.random.uniform(
        low=75, high=120, size=num_pairs)
    start_bottle_angles = np.random.uniform(
        low=30, high=65, size=num_pairs)

    # horizontal angle
    # vertical angle
    arm_EE_psis = np.random.uniform(
        low=-5, high=60, size=num_pairs)

    for i in range(num_pairs):
        start_bottle_angle = start_bottle_angles[i]
        goal_bottle_angle = goal_bottle_angles[i]
        # needs to behind bottle generally facing towards goal due to simple heuristic
        arm_EE_theta = np.random.uniform(
            low=-10, high=start_bottle_angle - 10, size=1)
        arm_EE_psi = arm_EE_psis[i]

        goal_dist = np.random.uniform(
            low=0.3, high=0.8) * max_horiz_dist
        start_dist = np.random.uniform(
            low=0.3, high=0.8) * max_horiz_dist

        EE_pos = calc_EE_from_direction(
            arm.MAX_HORIZ_REACH, theta=arm_EE_theta * DEG_TO_RAD, psi=arm_EE_psi * DEG_TO_RAD,
            z_offset=arm.max_straight_pos[-1])

        start_x = start_dist * math.cos(start_bottle_angle * DEG_TO_RAD)
        start_y = start_dist * math.sin(start_bottle_angle * DEG_TO_RAD)
        goal_x = start_dist * math.cos(goal_bottle_angle * DEG_TO_RAD)
        goal_y = start_dist * math.sin(goal_bottle_angle * DEG_TO_RAD)

        start = [start_x, start_y] + [bottle.INIT_PLANE_OFFSET]
        goal = [goal_x, goal_y] + [0]
        start_goal_pairs.append((start, goal, EE_pos))

    return start_goal_pairs
