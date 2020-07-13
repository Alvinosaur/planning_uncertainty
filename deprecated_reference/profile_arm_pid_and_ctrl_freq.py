
import math
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pybullet as p
import pybullet_data

from environment import ActionSpace, Environment
from sim_objects import Arm, Bottle

RAD_TO_DEG = 180.0 / math.pi


class TestEnvironment(Environment):
    def __init__(self, arm: Arm, bottle: Bottle, da, is_viz=True, use_3D=True, min_iters=10, max_iters=150):
        super().__init__(arm, bottle, is_viz=is_viz, use_3D=use_3D,
                         min_iters=min_iters, max_iters=max_iters)
        self.da = da
        self.A = ActionSpace(num_DOF=arm.num_DOF, da_rad=self.da)

    def compare_forces(self, num_actions=1):
        forces = np.linspace(0, 500, num=50 + 1)
        colors = pl.cm.jet(np.linspace(
            start=0, stop=1, num=len(forces)))
        init_joints = self.arm.init_joints

        # plot x, y, z for different forces
        ts = np.linspace(start=0, stop=self.min_iters, num=self.min_iters)
        EEfig, EEaxs = pl.subplots(3)
        EEfig.suptitle("EE Position with %d iters" % self.min_iters)
        titles = ["x", "y", "z"]
        for i in range(3):
            EEaxs[i].set_title(titles[i])

        # plot joint value trajectories
        jvalue_fig, jvalue_axs = pl.subplots(self.arm.num_DOF)
        jvalue_fig.suptitle("Joint Values(Deg) with %d iters" % self.min_iters)

        # plot joint velocity trajectories
        jvel_fig, jvel_axs = pl.subplots(self.arm.num_DOF)
        jvel_fig.suptitle(
            "Joint Velocities(Deg/s) with %d iters" % self.min_iters)

        # plot joint torque trajectories
        jtor_fig, jtor_axs = pl.subplots(self.arm.num_DOF)
        jtor_fig.suptitle("Joint Torques with %d iters" % self.min_iters)

        for i in range(self.arm.num_DOF):
            jvalue_axs[i].set_title("Joint %d" % i)
            jvel_axs[i].set_title("Joint %d" % i)
            jtor_axs[i].set_title("Joint %d" % i)

        count = 0
        for ai in self.A.action_ids:
            if count > num_actions:
                break
            count += 1
            # action defined as an offset of joint angles of arm
            dq = self.A.get_action(ai)
            print("Action: ", dq * 180.0 / math.pi)

            for fi, force in enumerate(forces):
                self.arm.force = force
                (actual_EE_traj, actual_joint_traj,
                 actual_joint_vel, actual_joint_tor) = self.run_sim(
                    action=dq, init_joints=init_joints,
                    bottle_pos=self.bottle.start_pos, bottle_ori=self.bottle.start_ori)

                # convert all angles to degrees for easier understanding
                actual_joint_traj *= RAD_TO_DEG
                actual_joint_vel *= RAD_TO_DEG

                color = colors[fi, :]
                for i in range(3):
                    EEaxs[i].plot(ts, actual_EE_traj[:, i],
                                  c=color, label=force)

                for i in range(self.arm.num_DOF):
                    jvalue_axs[i].plot(ts, actual_joint_traj[:, i],
                                       c=color, label=force)
                    jvel_axs[i].plot(ts, actual_joint_vel[:, i],
                                     c=color, label=force)
                    jtor_axs[i].plot(ts, actual_joint_tor[:, i],
                                     c=color, label=force)

            # create legends after all data has been populated
            handles, labels = EEaxs[-1].get_legend_handles_labels()
            EEfig.legend(handles, labels, loc="upper right", labelspacing=0)
            jvel_fig.legend(handles, labels, loc="upper right", labelspacing=0)
            handles, labels = jtor_axs[-1].get_legend_handles_labels()
            jtor_fig.legend(handles, labels, loc="upper right", labelspacing=0)

            # have joint values y-axis show 1-degree  ticks
            for i in range(self.arm.num_DOF):
                start, end = jvalue_axs[i].get_ylim()
                jvalue_axs[i].yaxis.set_ticks(np.arange(start, end, 1.0))
            handles, labels = jvalue_axs[-1].get_legend_handles_labels()
            jvalue_fig.legend(
                handles, labels, loc="upper right", labelspacing=0)
            handles, labels = jvel_axs[-1].get_legend_handles_labels()

            pl.show()
            pl.close()

    def compare_min_iters(self, i=0):
        """Since min_iters = 1 / freq for arm control, we only test that here, not max_iters.
        """
        all_min_iters = np.linspace(start=10, stop=50, num=5).astype(int)
        self.min_iters = all_min_iters[i]
        self.compare_forces()

    def run_sim(self, action, init_joints=None, bottle_pos=None, bottle_ori=None):
        """Deterministic simulation where all parameters are already set and
        known.

        Arguments:
            action {np.ndarray} -- offset in joint space, generated in ActionSpace
        """
        if init_joints is None:  # use arm's current joint state
            init_joints = np.array(self.arm.joint_pose)
        else:
            init_joints = np.array(init_joints)

        target_joint_pose = np.array(init_joints) + action
        print(init_joints * RAD_TO_DEG)
        print(target_joint_pose * RAD_TO_DEG)
        joint_traj = np.linspace(init_joints,
                                 target_joint_pose, num=2)

        return self.simulate_plan(joint_traj=joint_traj, bottle_pos=bottle_pos, bottle_ori=bottle_ori)

    def simulate_plan(self, joint_traj, bottle_pos, bottle_ori):
        """Run simulation with given joint-space trajectory. Returns debug information showing the following:
            - desired v.s actual joint trajectories
            - actual EE position trajectory
            - joint velocities
            - joint torques

        The core simulation format itself is identical to parent Environment()
        apart from extra debugging info returned instead of cost/next state
        """
        self.arm.reset(joint_traj[0, :])
        init_arm_pos = np.array(p.getLinkState(
            self.arm.kukaId, self.arm.EE_idx)[4])
        prev_arm_pos = np.copy(init_arm_pos)

        # create new bottle object with parameters set beforehand
        if bottle_pos is not None:
            self.bottle.create_sim_bottle(bottle_pos, ori=bottle_ori)
            prev_bottle_pos = bottle_pos
        else:
            self.bottle.create_sim_bottle(ori=bottle_ori)
            prev_bottle_pos = self.bottle.start_pos
        bottle_vert_stopped = False
        bottle_horiz_stopped = False
        bottle_stopped = bottle_vert_stopped and bottle_horiz_stopped

        # iterate through action
        pos_change = 0  # init arbitrary
        thresh = 0.001
        EE_error = 0

        # EXTRA debugging info
        actual_EE_traj = np.zeros((self.min_iters, 3))  # x, y, z
        # M-iters x Num-dof
        actual_joint_traj = np.zeros((self.min_iters, self.arm.num_DOF))
        actual_joint_vel = np.zeros_like(actual_joint_traj)
        actual_joint_tor = np.zeros_like(actual_joint_traj)

        iter = 0
        # print(joint_traj[-1, :] * 180 / math.pi)
        while iter < self.min_iters or (iter < self.max_iters and not bottle_stopped):
            # set target joint pose
            if iter < self.min_iters:
                next_joint_pose = joint_traj[-1, :]
                for ji, jval in enumerate(next_joint_pose):
                    p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                            jointIndex=ji,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=jval,
                                            force=self.arm.force,
                                            positionGain=self.arm.position_gain)
            # run one sim iter
            p.stepSimulation()
            self.arm.update_joint_pose()

            # record debug info
            ls = p.getLinkState(self.arm.kukaId, self.arm.EE_idx)
            joint_states = p.getJointStates(
                self.arm.kukaId, range(self.arm.num_DOF))
            EE_pos = ls[4]
            if iter < self.min_iters:
                actual_EE_traj[iter, :] = EE_pos
                actual_joint_traj[iter, :] = [state[0]
                                              for state in joint_states]
                actual_joint_vel[iter, :] = [state[1]
                                             for state in joint_states]
                actual_joint_tor[iter, :] = [state[3]
                                             for state in joint_states]

            # get feedback and vizualize trajectories
            if self.is_viz and prev_arm_pos is not None:
                ls = p.getLinkState(self.arm.kukaId, self.arm.EE_idx)
                arm_pos = ls[4]
                # Uncomment below to visualize lines of target and actual trajectory
                # also slows down simulation, so only run if trying to visualize
                # p.addUserDebugLine(prev_target, next_target, [0, 0, 0.3], 1, 1)
                # p.addUserDebugLine(arm_pos, prev_arm_pos, [1, 0, 0], 1,
                #                    self.trail_dur)
                prev_arm_pos = arm_pos
                # time.sleep(self.SIM_VIZ_FREQ)

            # check status of other objects to possibly terminate sim early
            bottle_pos, bottle_ori = p.getBasePositionAndOrientation(
                self.bottle.bottle_id)
            bottle_vert_stopped = math.isclose(
                bottle_pos[2] - prev_bottle_pos[2],
                0.0, abs_tol=1e-05)
            bottle_horiz_stopped = math.isclose(
                np.linalg.norm(
                    np.array(bottle_pos)[:2] - np.array(prev_bottle_pos)[:2]),
                0.0, abs_tol=1e-05)
            bottle_stopped = bottle_vert_stopped and bottle_horiz_stopped
            prev_bottle_pos = bottle_pos

            iter += 1

        return actual_EE_traj, actual_joint_traj, actual_joint_vel, actual_joint_tor


def main():
    VISUALIZE = False
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

    # bottle
    bottle_start_pos = np.array(
        [0.5, 0.5, Bottle.INIT_PLANE_OFFSET]).astype(float)
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

    da = 15 * math.pi / 180.0
    env = TestEnvironment(arm, bottle, da, is_viz=VISUALIZE)
    env.compare_min_iters(2)


if __name__ == "__main__":
    main()
