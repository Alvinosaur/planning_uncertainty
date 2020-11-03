import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def plot_distrib(horiz_bins, vert_bins, xlabel, ylabel, title):
    x_ind = horiz_bins
    plt.xticks(horiz_bins)
    plt.yticks(vert_bins)
    plt.bar(horiz_bins, vert_bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.close()


def find_nearest_bin(dist, targets):
    diff = abs(targets - dist)
    return np.argmin(diff)


def check_is_fallen(bottle_ori):
    z_axis = np.array([0, 0, 1])
    rot_mat = R.from_quat(bottle_ori).as_matrix()
    new_z_axis = rot_mat @ z_axis
    angle = math.acos(z_axis @ new_z_axis / 
        (np.linalg.norm(z_axis) * np.linalg.norm(new_z_axis)))
    return abs(angle) > (85 * math.pi / 180)  # when z-axis rotation is 90deg


def euc_dist_horiz(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


def calc_joints_from_pos(L1, L2, goal_x, goal_y):
    # while desired x, y is out of reach of arm
    # check if hypotenuse of triangle formed by x and y is > combined arm length
    theta1 = math.acos((goal_x**2 + goal_y**2 - L1**2 - L2**2) /
                       (2*L1*L2))
    theta0 = math.atan2(goal_y, goal_x) - (
                math.atan2(L2*math.sin(theta1),
                           L1 + L2*math.cos(theta1)))
    # simply invert conversion to get radians to degree
    return (theta0, theta1)


def get_target_joint_pos(arm, contact_heights, L1, L2, base_link_l):
    joint_poses = []
    for contact_height in contact_heights:
        target_z = contact_height - base_link_l # subtracted base link length
        theta1, theta2 = None, None
        possible_y = np.arange(start=(L1 + L2), stop=0, step=-1*(L1+L2)/20)
        for target_y in possible_y:
            try:
                theta1, theta2 = calc_joints_from_pos(L1, L2, goal_x=target_y, goal_y=target_z)
                break
            except Exception:
                continue
        new_pos = arm.rp
        new_pos[1] = math.pi/2 - theta1
        new_pos[3] = theta2
        joint_poses.append(copy.copy(new_pos))

    return joint_poses