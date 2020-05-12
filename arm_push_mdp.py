import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np

from sim_objects import Bottle, Arm
from environment import Environment, Action

def main():
    # initialize simulator environment
    VISUALIZE = False
    LOGGING = False
    GRAVITY = -9.81
    if VISUALIZE: p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    else: p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,GRAVITY)
    planeId = p.loadURDF(Environment.plane_urdf_filepath)
    kukaId = p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0])
    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "sim_run.mp4")

    # starting end-effector pos, not base pos
    EE_start_pos = np.array([0.2, 0.2, 0.3]).astype(float)
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    arm = Arm(EE_start_pos=EE_start_pos, start_ori=base_start_ori, 
        kukaId=kukaId)

    # bottle
    bottle_start_pos = np.array([0.7, 0.6, 0.1]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)
    
    N = 700
    env = Environment(arm, bottle, is_viz=VISUALIZE, N=N)

    A = init_action_space(env.bottle)
    solver = MDP(env)
    solver.solve_mdp()
    # test(env, A[0])
    # test_state_to_idx(mdp=solver)

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

    # to deal with the issue of bottle being pushed out of range of arm and reaching a state with some unknown value since we can't even reach, just leave V(out of bounds states) = 0, but set every state's reward to be euclidean distance from some goal

def init_action_space(bottle):
    A = []  # need to maintain order
    dh = 2  # 5
    contact_heights = np.arange(
        start=bottle.height/dh, 
        stop=bottle.height + bottle.height/dh, 
        step=bottle.height/dh)
    
    # velocities = np.arange(start=0.005, stop=0.01, step=0.001)
    velocities = np.arange(start=0.1, stop=0.51, step=0.1)

    da = math.pi/80
    # angle_offsets = np.arange(start=-3*da, stop=4*da, step=da)
    angle_offsets = np.arange(start=-da, stop=2*da, step=da)
    for h in contact_heights:
        for v in velocities:
            for a in angle_offsets:
                action = Action(angle_offset=a, velocity=v, height=h)
                A.append(action)
            
    return A

class MDP():
    def __init__(self, env):
        self.max_iters = 10
        self.gamma = 0.8
        # action space
        self.A = init_action_space(env.bottle)
        self.env = env

        # full state space
        self.dx = self.dy = 0.1
        self.xlim, self.ylim = 1.5, 1.5
        self.X = np.arange(start=-self.xlim, stop=self.xlim, step=self.dx)
        self.Y = np.arange(start=-self.ylim, stop=self.ylim, step=self.dy)
        self.H, self.W = len(self.Y), len(self.X)

        # filtering out states out of bounuds
        self.valid_states = []
        for x in self.X:
            for y in self.Y:
                dist_from_base = np.linalg.norm(
                    np.array([x,y]) - self.env.arm.base_pos[:2])
                if (dist_from_base < self.env.arm.min_dist or 
                    dist_from_base > env.arm.MAX_REACH):
                    continue
                else: self.valid_states.append((x,y))

        self.V = np.zeros((self.H, self.W))  # 1D array
        # self.P = np.zeros((self.H * self.W, len(A)))  # policy as probabilities
        self.P = [[None]*self.W for i in range(self.H)]
        
    def state_to_idx(self, state):
        (x, y) = state
        xi = int((x + self.xlim)/self.dx)
        yi = int((y + self.ylim)/self.dy)
        return (xi, yi)

    def solve_mdp(self):
        for iter in range(self.max_iters):
            print(iter)
            self.update_policy()
            self.evaluate_policy()

    # run through all possible actions at a given state
    
            # expected_cost = env.run_sim_stochastic(action)


    def evaluate_policy(self):
        """Use current policy to estimate new value function
        """
        new_V = np.zeros_like(self.V)  # synchronous update for now
        for (x,y) in self.valid_states:
            self.env.change_bottle_pos(
                new_pos=[x, y, 0.1],
                target_type="extend")
            (xi, yi) = self.state_to_idx((x,y))
            best_actions = self.P[yi][xi]
            expected_cost = 0
            prob = 1./float(len(best_actions))  # uniform distb for best actions
            for action in best_actions:
                cost, ns = self.env.run_sim_stochastic(action)
                try:
                    (nxi, nyi) = self.state_to_idx(ns)
                    expected_future_cost = self.V[nyi, nxi]
                except IndexError:
                    expected_future_cost = self.V[yi, xi]
                    # expected_future_cost = 0
                expected_cost += prob * (
                    cost + self.gamma*expected_future_cost)
                
            # synchronous update
            new_V[yi, xi] = expected_cost
        
        print(self.V)
        self.V = new_V


    def update_policy(self):
        """Use current value function to estimate new policy.
        """
        # for each state, find best action(s) to take
        for (x,y) in self.valid_states:
            self.env.change_bottle_pos(
                new_pos=[x, y, 0.1],
                target_type="extend")
            (xi, yi) = self.state_to_idx((x,y))
            min_cost = 0
            best_actions = []
            for ai, action in enumerate(self.A):
                cost, ns = self.env.run_sim_stochastic(action)
                try:
                    (nxi, nyi) = self.state_to_idx(ns)
                    expected_future_cost = self.V[nyi, nxi]
                except IndexError:
                    # just use value at current state if next state is out of  bounds
                    expected_future_cost = self.V[yi, xi]
                    # expected_future_cost = 0

                total_cost = cost + self.gamma*expected_future_cost
                if len(best_actions) == 0 or total_cost < min_cost:
                    min_cost = total_cost
                    best_actions = [ai]
                elif math.isclose(total_cost, min_cost, abs_tol=1e-6):
                    best_actions.append(ai)
                
            self.P[yi][xi] = best_actions


def test_state_to_idx(mdp: MDP):
    # self.dx = self.dy = 0.1
    # self.xlim, self.ylim = 1.5, 1.5
    # self.X = np.arange(start=-self.xlim, stop=self.xlim, step=self.dx)
    # self.Y = np.arange(start=-self.ylim, stop=self.ylim, step=self.dy)
    x, y = -mdp.xlim, -mdp.ylim
    print(mdp.state_to_idx((x,y)))
    x, y = mdp.xlim, mdp.ylim
    print(mdp.state_to_idx((x,y)))
    x, y = 0, 0
    print(mdp.state_to_idx((x,y)))
    x, y = mdp.xlim + mdp.dx*0.9, mdp.ylim + mdp.dy*0.9
    print(mdp.state_to_idx((x,y)))


def test(env, action):
    X = np.arange(start=0, stop=1.5, step=0.5)
    Y = np.arange(start=-0.5, stop=1.5, step=0.1)
    velocities = np.arange(start=0.005, stop=0.01, step=0.001)
    for x in X:
        for y in Y:
            for v in velocities:
                action.velocity = v
                dist_from_base = np.linalg.norm(
                    np.array([x,y]) - env.arm.base_pos[:2])
                if (dist_from_base < env.arm.min_dist or 
                    dist_from_base > env.arm.MAX_REACH):
                    continue
                env.change_bottle_pos([x, y, 0.1])
                expected_cost = env.run_sim(action)

if __name__=='__main__':
    main()