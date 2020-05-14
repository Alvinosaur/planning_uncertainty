import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time

from sim_objects import Arm, Bottle


class Action():
    def __init__(self, angle_offset, velocity, height):
        self.angle_offset = angle_offset
        self.velocity = velocity
        self.height = height

    def __repr__(self):
        return "a: %d, v: %.3f, h: %.3f" % (
            int(self.angle_offset * 180 / math.pi),
            self.velocity,
            self.height
        )

class Environment(object):
    # pybullet_data built-in models
    plane_urdf_filepath = "plane.urdf"
    arm_filepath = "kuka_iiwa/model.urdf"
    table_filepath = "table/table.urdf"
    gripper_path = "kuka_iiwa/kuka_with_gripper.sdf"
    def __init__(self, arm, bottle,  is_viz=True, gravity=-9.81, N=500, 
            run_full_mdp=True):
        self.arm = arm
        self.bottle = bottle
        # angle to hit bottle
        self.main_angle = math.atan2(bottle.start_pos[1], bottle.start_pos[0])
        self.is_viz = is_viz
        self.gravity = gravity
        self.stop_sim_thresh = 0.001  # if arm can't move any further, stop sim
        self.MIN_ITERS = 50  # sim doesn't prematurely stop if arm too slow
        self.MAX_ITERS = 700  # number of sim steps per action
        self.trail_dur = 1  # visulize arm trajectory
        self.target_bottle_pos = np.array([0.8, 0.6, 0.1])
        self.FALL_COST = 100
        self.dist_cost_scale = 10
        self.SIM_VIZ_FREQ = 1/240.
        self.run_full_mdp = run_full_mdp
        self.target_thresh = 0.08

        
        # varying unobservable parameters of bottle
        if run_full_mdp:
            self.min_fric = 0.1
            self.max_fric = 0.2
            self.fric_step = 0.05
            self.min_fill = 0.1
            self.max_fill = 1.0
            self.fill_step = 0.3
        else:
            self.min_fric = 0.1
            self.max_fric = 0.2
            self.fric_step = 0.05 
            self.min_fill = 0.1
            self.max_fill = 1.0
            self.fill_step = 0.3

    # def create_sim(self):
    #     if self.is_viz: p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    #     else: p.connect(p.DIRECT)
    #     p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #     p.setGravity(0,0,self.gravity)
    #     planeId = p.loadURDF(Environment.plane_urdf_filepath)

    # def restart_sim(self, is_viz=None):
    #     # restart simulator with option of changing visualization mode
    #     try:
    #         p.disconnect()
    #     except p.error:
    #         print("No sim to disconnect from, creating new one as normal...")
    #     if is_viz is not None: self.is_viz = is_viz
    #     self.create_sim()

    def eval_cost(self, is_fallen, bottle_pos):
        dist = np.linalg.norm(self.target_bottle_pos[:2] - bottle_pos[:2])
        # Pure euclidean distance from goal
        # return self.dist_cost_scale*dist + self.FALL_COST*is_fallen
        # reach target within some radius
        return (self.dist_cost_scale*(dist >= self.target_thresh) + 
                self.FALL_COST*is_fallen)


    def change_bottle_pos(self, new_pos, target_type="extend"):
        self.bottle.start_pos = new_pos
        if target_type == "extend":
            self.target_bottle_pos = np.array(new_pos) * 2
            self.target_bottle_pos[2] = new_pos[2]  # keep z position same
        elif target_type == "const":
            self.target_bottle_pos = np.array([0.65, 0.55, 0.1])
        else:
            print("Invalid type of bottle change: %s" % target_type)
            assert(False)
        self.main_angle = math.atan2(new_pos[1], new_pos[0])


    def run_sim_stochastic(self, action: Action):
        """Different from normal run_sim by iterating over unknown parameters of bottle friction, slight deviations in bottle location, bottle mass

        Arguments:
            action {[type]} -- [description]
        """
        total_cost = 0
        lat_frics = np.arange(
            start=self.min_fric, 
            stop=(self.max_fric+self.fric_step), 
            step=self.fric_step)
        fill_props = np.arange(
            start=self.min_fill, 
            stop=(self.max_fill+self.fill_step), 
            step=self.fill_step)
        total_iters = float(len(fill_props) * len(lat_frics))
        # sim_iter = 0
        # iterate through possible env factors to account for unknwon parameters
        # and get expected cost
        avg_bottle_final_pos = np.zeros((2,))
        # avg = 0.0
        pos_count = 0
        for fill_prop in fill_props:
            self.bottle.set_fill_proportion(fill_prop)
            for lat_fric in lat_frics:
                # sim_iter += 1
                self.bottle.lat_fric = lat_fric
                start = time.time()
                cost, bottle_pos = self.run_sim(action)
                end = time.time()
                
                # avg += (end-start)
                # only include avg bottle pos if didn't fall
                # since simulator sometimes produces strange behavior when bottle falls
                if cost < self.FALL_COST:
                    avg_bottle_final_pos += bottle_pos
                    pos_count += 1
                total_cost += cost
        
        # print("Time(s) to run one sim: %.3f"  % (avg / total_iters))
        expected_cost = total_cost / total_iters
        if pos_count > 0:
            avg_bottle_final_pos = avg_bottle_final_pos / pos_count
        else:
            avg_bottle_final_pos = self.bottle.start_pos[:2]
        return expected_cost, avg_bottle_final_pos


    def run_sim(self, action: Action):
        """Deterministic simulation where all parameters are already set and 
        known.

        Arguments:
            action {Action} -- target position, force
        """
        # action composed of desired angle to move arm towards and height
        da = action.angle_offset
        angle = self.main_angle + da
        dir_vec = np.array([math.cos(angle), math.sin(angle), 0])
        const_height_vec = np.array([0, 0, action.height])

        # generate arm EE trajectory
        dt = 0.1
        T = 100.
        init_pos = 0.5 * np.array(self.bottle.start_pos) + np.array([0, 0, 0.2])
        target_pos = self.arm.MAX_REACH*dir_vec + const_height_vec
        num_iters = int(min(T/action.velocity, self.MAX_ITERS))
        # num_iters = int(T/action.velocity)
        xy_traj = np.linspace(init_pos[:2], target_pos[:2], num=num_iters)

        # find iter where arm makes contact with arm, which should be minimizing horiz dist with bottle position
        iters_till_contact = np.argmin(
            np.linalg.norm(xy_traj-self.bottle.start_pos[:2], axis=1) )
        z_traj = np.linspace(init_pos[2], target_pos[2], num=iters_till_contact)

        pos = np.copy(init_pos)
        prevPose = np.copy(pos)
        prevPose1 = init_pos

        # create new bottle object with parameters set beforehand
        self.bottle.create_sim_bottle()
        prev_bottle_pos = self.bottle.start_pos
        bottle_vert_stopped = True
        bottle_horiz_stopped = True
        
        self.arm.reset(target_pos=init_pos, angle=angle)

        # iterate through action
        pos_change = 0  # init arbitrary
        iter = 0
        t = 0
        dist = np.linalg.norm(init_pos - target_pos)
        thresh = 0.001

        while (pos_change > thresh or iter < num_iters-1 or 
                not (bottle_vert_stopped and bottle_horiz_stopped)) and (
                    iter < self.MAX_ITERS): 
        # for iter in range(num_iters):
            iter += 1
            pos_i = min(iter, num_iters-1)
            pos = xy_traj[pos_i,:]
            z = z_traj[min(iter, iters_till_contact-1)]
            pos = np.append(pos, z)

            # set target joint positions of arm
            joint_poses = self.arm.get_target_joints(pos, angle)
            for i in range(self.arm.num_joints):
                p.setJointMotorControl2(bodyIndex=self.arm.kukaId,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_poses[i],
                                        targetVelocity=0,
                                        force=self.arm.force,
                                        positionGain=self.arm.position_gain,
                                        velocityGain=self.arm.velocity_gain)
            # run one sim iter
            p.stepSimulation()

            # get feedback and vizualize trajectories
            ls = p.getLinkState(self.arm.kukaId, self.arm.EE_idx)
            # Uncomment below to visualize lines of target and actual trajectory
            # p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, self.trail_dur)
            # p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, self.trail_dur)
            pos_change = np.linalg.norm(prevPose1 - np.array(ls[4]))
            # dist = np.linalg.norm(np.array(ls[4]) - target_pos)

            prevPose = np.copy(pos)
            prevPose1 = np.array(ls[4])

            bottle_pos, bottle_ori = p.getBasePositionAndOrientation(
                self.bottle.bottle_id)
            bottle_vert_stopped = math.isclose(
                bottle_pos[2] - prev_bottle_pos[2], 
                0.0, abs_tol=1e-05)
            bottle_horiz_stopped = math.isclose(
                np.linalg.norm(
                    np.array(bottle_pos)[:2] - np.array(prev_bottle_pos)[:2]), 
                0.0, abs_tol=1e-05)
            prev_bottle_pos = bottle_pos

            # if self.is_viz: time.sleep(self.SIM_VIZ_FREQ)

        # stop simulation if bottle and arm stopped moving
        is_fallen = self.bottle.check_is_fallen()
        # remove bottle object, can't just reset pos since need to change params each iter
        p.removeBody(self.bottle.bottle_id)

        return self.eval_cost(is_fallen, bottle_pos), np.array(bottle_pos[:2])
        
        # # print(bottle_vert_stopped)