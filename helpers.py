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

