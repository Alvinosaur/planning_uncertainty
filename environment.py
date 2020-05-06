import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np

from sim_objects import Arm, Bottle


class Environment(object):
    # pybullet_data built-in models
    plane_urdf_filepath = "plane.urdf"
    arm_filepath = "kuka_iiwa/model.urdf"
    table_filepath = "table/table.urdf"
    gripper_path = "kuka_iiwa/kuka_with_gripper.sdf"
    def __init__(self, arm, bottle,  is_viz=True, gravity=-9.81, N=500):
        self.arm = arm
        self.bottle = bottle
        self.is_viz = is_viz
        self.gravity = gravity
        self.stop_sim_thresh = 0.001  # if arm can't move any further, stop sim
        self.MIN_ITERS = 50  # sim doesn't prematurely stop if arm too slow
        self.N = N  # number of sim steps per action
        self.trail_dur = 3  # visulize arm trajectory
        self.target_bottle_pos = np.array([0.8, 0.6, 0.1])
        self.FALL_COST = 100
        self.dist_cost_scale = 10

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
        dist = np.linalg.norm(self.target_bottle_pos - bottle_pos)
        return self.dist_cost_scale*dist + self.FALL_COST*is_fallen


    def run_sim_stochastic(self, action):
        """Different from normal run_sim by iterating over unknown parameters of bottle friction, slight deviations in bottle location, bottle mass

        Arguments:
            action {[type]} -- [description]
        """
        total_cost = 0
        lat_frics = np.arange(start=0.1, stop=(0.4+0.1), step=0.1)
        fill_props = np.arange(start=0, stop=(1+0.1), step=0.3)
        total_iters = float(len(fill_props) * len(lat_frics))
        sim_iter = 0
        # iterate through possible env factors to account for unknwon parameters
        # and get expected cost
        for fill_prop in fill_props:
            self.bottle.set_fill_proportion(fill_prop)
            for lat_fric in lat_frics:
                sim_iter += 1
                self.bottle.lat_fric = lat_fric
                cost = self.run_sim(action)
                total_cost += cost
                
        expected_cost = total_cost / total_iters
        return expected_cost


    def run_sim(self, action):
        """Deterministic simulation where all parameters are already set and 
        known.

        Arguments:
            action {Action} -- target position, force
        """
        # action composed of desired angle to move arm towards and height
        (angle, const_height) = action
        dir_vec = np.array([math.cos(angle), math.sin(angle), 0])
        velocity = 0.01

        # create new bottle object with parameters set beforehand
        self.bottle.create_sim_bottle()
        self.arm.reset()

        # set initial target end-effector pos of arm
        pos = velocity * dir_vec + np.array([0, 0, const_height])
        prevPose = np.copy(pos)
        prevPose1 = np.array([0, 0, 0])

        # iterate through action
        pos_change = 0  # init arbitrary
        iter = 0
        t = 0
        while iter < self.N and (
                iter < self.MIN_ITERS or pos_change > self.stop_sim_thresh):
            iter += 1
            # update target
            pos += velocity * dir_vec
            
            # confine target position within reach of arm
            target_dist = np.linalg.norm(pos - self.arm.EE_start_pos)
            if target_dist >  self.arm.MAX_REACH:
                pos = self.arm.MAX_REACH*dir_vec + np.array([0,0,const_height])

            # set target joint positions of arm
            joint_poses = self.arm.get_target_joints(pos)
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
            p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, self.trail_dur)
            p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, self.trail_dur)
            pos_change = np.linalg.norm(prevPose1 - np.array(ls[4]))

            prevPose = np.copy(pos)
            prevPose1 = np.array(ls[4])

        # stop simulation if bottle and arm stopped moving
        is_fallen = self.bottle.check_is_fallen()
        bottle_pos, bottle_ori = p.getBasePositionAndOrientation(
            self.bottle.bottle_id)
        # remove bottle object, can't just reset pos since need to change params each iter
        p.removeBody(self.bottle.bottle_id)
        return self.eval_cost(is_fallen, bottle_pos)
        # bottle_vert_stopped = math.isclose(bottle_pos[2] - prev_pos[2], 0.0, abs_tol=1e-05)
        # # print(bottle_vert_stopped)
        # bottle_horiz_stopped = math.isclose(helpers.euc_dist_horiz(bottle_pos, prev_pos), 0.0, abs_tol=1e-05)


def main():
    # initialize simulator environment
    VISUALIZE = True
    LOGGING = False
    GRAVITY = -9.81
    if VISUALIZE: p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    else: p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,GRAVITY)
    planeId = p.loadURDF(Environment.plane_urdf_filepath)
    kukaId = p.loadURDF(Environment.arm_filepath, [0, 0, 0])
    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "sim_run.mp4")

    # starting end-effector pos, not base pos
    EE_start_pos = np.array([0.2, 0.2, 0.3]).astype(float)
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    arm = Arm(EE_start_pos=EE_start_pos, start_ori=base_start_ori, 
        kukaId=kukaId)

    # bottle
    bottle_start_pos = np.array([0.6, 0.6, 0.1]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)
    bottle.col_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=bottle.radius, 
        height=bottle.height)
    
    env = Environment(arm, bottle)

    # Action space
    dh = 5
    contact_heights = np.arange(
        start=bottle.height/dh, 
        stop=bottle.height + bottle.height/dh, 
        step=bottle.height/dh)
    angles = np.arange(start=math.pi/4, stop=math.pi+math.pi/4, step=math.pi/4)

    # run through all possible actions at a given state
    for angle in angles:
        for h in contact_heights:
            action = (angle, h)
            expected_cost = env.run_sim(action)
            # expected_cost = env.run_sim_stochastic(action)

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

if __name__=='__main__':
    main()