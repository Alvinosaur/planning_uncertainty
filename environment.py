import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time

from sim_objects import Arm, Bottle


class Action():
    def __init__(self, angle_offset, velocity, height, reach_p):
        self.angle_offset = angle_offset
        self.velocity = velocity
        self.height = height
        self.reach_p = reach_p

    def __repr__(self):
        return "a: %d, v: %.3f, h: %.3f, rp: %.1f" % (
            int(self.angle_offset * 180 / math.pi),
            self.velocity,
            self.height,
            self.reach_p
        )

class Environment(object):
    # pybullet_data built-in models
    plane_urdf_filepath = "plane.urdf"
    arm_filepath = "kuka_iiwa/model.urdf"
    table_filepath = "table/table.urdf"
    gripper_path = "kuka_iiwa/kuka_with_gripper.sdf"
    INF = 1e10
    def __init__(self, arm, bottle,  is_viz=True, gravity=-9.81, N=500, 
            run_full_mdp=True, cost_based=True):
        self.arm = arm
        self.bottle = bottle
        # angle to hit bottle
        self.main_angle = math.atan2(bottle.start_pos[1], bottle.start_pos[0])
        self.is_viz = is_viz
        self.gravity = gravity
        self.stop_sim_thresh = 0.001  # if arm can't move any further, stop sim
        self.MIN_ITERS = 50  # sim doesn't prematurely stop if arm too slow
        self.MAX_ITERS = 150  # number of sim steps per action
        # number of random samples of internal params for stochastic simulation
        self.sim_samples = 10  
        self.trail_dur = 1  # visulize arm trajectory
        self.target_bottle_pos = np.zeros((3,))
        self.FALL_COST = Environment.INF
        self.dist_cost_scale = 10
        self.SIM_VIZ_FREQ = 1/240.
        self.run_full_mdp = run_full_mdp
        self.target_thresh = 0.08
        self.init_reach_p = 0.7
        self.target_line = None
        self.cost_based = cost_based

        # Normal distribution of internal bottle params
        self.min_fric = 0.1
        self.max_fric = 0.2
        self.min_fill = 0.1
        self.max_fill = 1.0
        self.mean_friction = (self.min_fric + self.max_fric) / 2.
        # want min and max to be at 3 std deviations
        self.std_friction = (self.max_fric - self.mean_friction) / 3.
        self.mean_fillp = (self.min_fill + self.max_fill) / 2.
        self.std_fillp = (self.max_fill - self.mean_fillp) / 3.


    def eval_cost(self, is_fallen, bottle_pos):
        dist = np.linalg.norm(self.target_bottle_pos[:2] - bottle_pos[:2])
        # Pure euclidean distance from goal
        if self.cost_based:
            return dist + self.FALL_COST*is_fallen
        else:  # reward-based
            # reach target within some radius
            return (10*(dist < self.target_thresh) - 
                    self.FALL_COST*is_fallen)


    def change_bottle_pos(self, new_pos, target_type="extend"):
        self.bottle.start_pos = new_pos
        if target_type == "extend":
            self.target_bottle_pos = np.array(new_pos) * 2
            self.target_bottle_pos[2] = new_pos[2]  # keep z position same
        elif target_type == "const":
            const_target = np.array([0.65, 0.55, 0.1])
            
            if not np.allclose(self.target_bottle_pos, const_target, atol=1e-5):
                vert_offset_target = const_target + np.array([0,0,1])
                p.addUserDebugLine(const_target, 
                    vert_offset_target, [1, 0, 0], 4, 0)
                
            self.target_bottle_pos = const_target
        else:
            print("Invalid type of bottle change: %s" % target_type)
            assert(False)

        self.main_angle = math.atan2(new_pos[1], new_pos[0])


    def run_sim_stochastic(self, action: Action):
        """
        Randomly sample internal(unobservable to agent) bottle parameters.
        """
        expected_cost = 0
        expected_next_state = np.zeros((2,))
        for sim_iter in range(self.sim_samples):
            rand_fill = np.random.normal(
                loc=self.mean_fillp, scale=self.std_fillp)
            rand_fric = np.random.normal(
                loc=self.mean_friction, scale=self.std_friction)

            # ensure actually feasible parameters by enforcing bounds
            if rand_fill < self.min_fill: rand_fill = self.min_fill
            elif rand_fill > self.max_fill: rand_fill = self.max_fill
            if rand_fric < self.min_fric: rand_fric = self.min_fric
            elif rand_fric > self.max_fric: rand_fric = self.max_fric

            self.bottle.set_fill_proportion(rand_fill)
            self.bottle.lat_fric = rand_fric
            cost, ns = self.run_sim(action)
            expected_cost += cost
            expected_next_state += ns

        return (expected_cost / float(self.sim_samples), 
            expected_next_state / float(self.sim_samples))


    def run_sim(self, action: Action):
        """Deterministic simulation where all parameters are already set and 
        known.

        Arguments:
            action {Action} -- target position, force
        """
        # action composed of desired angle to move arm towards and height
        da = action.angle_offset
        angle = self.main_angle + da
        dir_vec = np.array([math.cos(angle), math.sin(angle)])
        const_height_vec = np.array([0, 0, action.height])

        # generate arm EE trajectory
        T = 10.

        bottle_dist = np.linalg.norm(
            self.bottle.start_pos[:2] - self.arm.base_pos[:2])
        # max x,y defined by arm's max reach, z is contact height
        max_horiz_dist = self.arm.calc_max_horiz_dist(action.height)
        max_pos = max_horiz_dist*dir_vec
        max_pos = np.append(max_pos, action.height)

        # baseline is with minimum dist target with reach_p=0
        baseline_target = bottle_dist*dir_vec
        baseline_target = np.append(baseline_target, action.height)

        # initial position is some small proportion away from baseline
        # acts like a "warm start" for IK solver
        init_pos = self.init_reach_p * baseline_target
        init_pos[2] = action.height  # don't scale contact height

        # some position in between bottle pos and max reach
        # ensure arm at least comes into contact with bottle
        target_pos = (action.reach_p*(max_pos-baseline_target) + 
            baseline_target)
        
        num_iters = int(T/action.velocity)
        # num_iters = int(T/action.velocity)
        traj = np.linspace(init_pos, target_pos, num=num_iters)

        return self.simulate_plan(traj, angle)


    def simulate_plan(self, traj, angle=0):
        """Run simulation with given end-effector position trajectory and 
        desired yaw angle of EE.

        Arguments:
            traj {[type]} -- [description]

        Keyword Arguments:
            angle {int} -- [description] (default: {0})

        Returns:
            [type] -- [description]
        """
        next_target = traj[0]
        prev_target = traj[0]
        prev_arm_pos = traj[0]
        self.arm.reset(target_pos=traj[0], angle=angle)

        if isinstance(traj, np.ndarray):
            num_iters = traj.shape[0]
        else: num_iters = len(traj)

        # create new bottle object with parameters set beforehand
        self.bottle.create_sim_bottle()
        prev_bottle_pos = self.bottle.start_pos
        bottle_vert_stopped = False
        bottle_horiz_stopped = False

        # iterate through action
        pos_change = 0  # init arbitrary
        iter = 0
        thresh = 0.001
        EE_error = 0

        while (pos_change > thresh or iter < num_iters-1 or 
                not (bottle_vert_stopped and bottle_horiz_stopped)) and (
                    iter < self.MAX_ITERS): 
            iter += 1
            pos_i = min(iter, num_iters-1)
            next_target = traj[pos_i]

            # set target joint positions of arm
            joint_poses = self.arm.get_target_joints(next_target, angle)
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
            arm_pos = np.array(ls[4])
            # Uncomment below to visualize lines of target and actual trajectory
            # also slows down simulation, so only run if trying to visualize
            p.addUserDebugLine(prev_target, next_target, [0, 0, 0.3], 1, 1)
            p.addUserDebugLine(arm_pos, prev_arm_pos, [1, 0, 0], 1, 
               self.trail_dur)
            
            EE_error += np.linalg.norm(next_target - arm_pos)
            
            pos_change = np.linalg.norm(arm_pos - prev_arm_pos)
            prev_arm_pos = arm_pos

            prev_target = np.copy(next_target)

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

        return self.eval_cost(is_fallen, bottle_pos), np.array(bottle_pos[:2]), EE_error
