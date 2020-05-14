import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import time
import matplotlib.pyplot as plt

from sim_objects import Bottle, Arm
from environment import Environment, Action

class MDP():
    def __init__(self, env, run_full_mdp=True, target_type="const"):
        if run_full_mdp: self.max_iters = 100
        else: self.max_iters = 10
        self.gamma = 0.8
        # action space
        self.A = init_action_space(env.bottle, run_full_mdp)
        self.env = env
        self.target_type = target_type

        # self.sim_log = dict()  # (x,y,action) -> [start, end]

        # full state space
        if run_full_mdp:
            self.dx = self.dy = 0.1
            self.xmin, self.xmax = 0, 1.5
            self.ymin, self.ymax = 0, 1.5
        else:
            # only consider one quadrant 
            self.dx = self.dy = 0.1
            self.xmin, self.xmax = 0, 0.8
            self.ymin, self.ymax = 0, 0.8
        self.X = np.arange(
            start=self.xmin, 
            stop=self.xmax+self.dx, 
            step=self.dx)
        self.Y = np.arange(
            start=self.ymin, 
            stop=self.ymax+self.dy, 
            step=self.dy)
        self.H, self.W = len(self.Y), len(self.X)
        self.OUT_OF_BOUNDS_COST = 20

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

        self.V = np.ones((self.H, self.W)) * self.OUT_OF_BOUNDS_COST # 1D array
        # self.P = np.zeros((self.H * self.W, len(A)))  # policy as probabilities
        self.P = [[0]*self.W for i in range(self.H)]
        self.all_Vs = [self.V]
        self.all_Ps = [self.P]
        
    def state_to_idx(self, state):
        (x, y) = state
        xi = int(round((x - self.xmin)/self.dx))
        yi = int(round((y - self.ymin)/self.dy))

        # usually leave out-of-bounds as-is, but for special case of rounding
        # up to bound, just keep within index range
        if xi == self.W: xi -= 1
        if yi == self.H: yi -= 1
        return (xi, yi)

    def solve_mdp(self):
        np.savez("value_policy_iter_0", V=self.V, P=self.P)
        for iter in range(self.max_iters):
            self.update_policy()
            self.evaluate_policy()
            np.savez("results/value_policy_iter_%d" % (iter+1), V=self.V, P=self.P)
            print("Percent complete: %.3f" % (iter / float(self.max_iters)))
            
    def evaluate_policy(self):
        """Use current policy to estimate new value function
        """
        print("Evaluting Policy to Update Value Function...")
        new_V = np.zeros_like(self.V)  # synchronous update for now
        num_states = float(len(self.valid_states))
        start = time.time()
        for (x,y) in self.valid_states:
            self.env.change_bottle_pos(
                new_pos=[x, y, 0.1],
                target_type=self.target_type)
            (xi, yi) = self.state_to_idx((x,y))
            best_actions = self.P[yi][xi]
            expected_cost = 0
            prob = 1./float(len(best_actions))  # uniform distb for best actions
            for ai in best_actions:
                action = self.A[ai]
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
        
        self.all_Vs.append(new_V)
        self.V = new_V

        end = time.time()
        print("Total Runtime of Eval Policy: %.3f" % (end-start))


    def update_policy(self):
        """Use current value function to estimate new policy.
        """
        print("Updating Policy...")
        num_states = float(len(self.valid_states))
        total_time = 0
        
        # for each state, find best action(s) to take
        for (x,y) in self.valid_states:
            start = time.time()
            self.env.change_bottle_pos(
                new_pos=[x, y, 0.1],
                target_type=self.target_type)
            (xi, yi) = self.state_to_idx((x,y))
            min_cost = 0
            best_actions = []
            # sim_log = []
            for ai, action in enumerate(self.A):
                cost, ns = self.env.run_sim(action)
                # sim_log.append((action, (x,y), ns))
                
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

            # self.plot_sim_results(sim_log)
                
            self.P[yi][xi] = best_actions
            end = time.time()
            print("Time(s) for one state: %.3f" % (end - start))
            total_time += (end-start)
        self.all_Ps.append(self.P)
        print("Total Runtime of Update Policy: %.3f" % total_time)

    def plot_sim_results(self, sim_log):
        # n = len(sim_log)
        # color_vals = np.random.randint(0, 0xFFFFFF, size=n)  # +1 for target
        # colors = [('#%06X' % v) for v in color_vals]
        for (_, start, end) in sim_log:
            print(start, end)
            plt.plot([start[0], end[0]], [start[1], end[1]])
        
        plt.legend([str(action) for action in self.A], loc='upper left')
        plt.show()

    def view_state_space(self):
        # visualize 
        action= self.A[0]
        action.velocity = 0.3
        for x in self.X:
            for y in self.Y:
                dist_from_base = np.linalg.norm(
                    np.array([x,y]) - self.env.arm.base_pos[:2])
                if (dist_from_base < self.env.arm.min_dist or 
                    dist_from_base > self.env.arm.MAX_REACH):
                    continue
                self.env.change_bottle_pos([x, y, 0.1], target_type=self.target_type)
                cost, ns = self.env.run_sim(action)
                dist = np.linalg.norm(ns - self.env.target_bottle_pos[:2])
                print(cost, dist, self.env.target_thresh, ns, self.env.target_bottle_pos[:2])

def main():
    # initialize simulator environment
    VISUALIZE = False
    LOGGING = False
    GRAVITY = -9.81
    RUN_FULL_MDP = False
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
    env = Environment(arm, bottle, is_viz=VISUALIZE, N=N, 
        run_full_mdp=RUN_FULL_MDP)

    solver = MDP(env, RUN_FULL_MDP)
    solver.solve_mdp()
    # solver.view_state_space()
    # check_results(env, mdp=solver)
    # test_bottle_dynamics(env)
    # A = init_action_space(env.bottle, RUN_FULL_MDP)
    # test(env, A[0])
    # test_state_to_idx(mdp=solver)

    if LOGGING and VISUALIZE:
        p.stopStateLogging(log_id)

    # to deal with the issue of bottle being pushed out of range of arm and reaching a state with some unknown value since we can't even reach, just leave V(out of bounds states) = 0, but set every state's reward to be euclidean distance from some goal

def check_results(env: Environment, mdp: MDP):
    path = "results/value_policy_iter_%d.npz"
    print("Value Functions:")
    policies = []
    for iter in range(1, mdp.max_iters+1):
        data = np.load(path % iter, allow_pickle=True)
        V, P = data["V"], data["P"]
        policies.append(P)
        print(V)

    print("Policies:")
    for P in policies: print(P)
        

def init_action_space(bottle, run_full_mdp):
    A = []  # need to maintain order
    da = math.pi/80
    if run_full_mdp:
        dh = 5
        velocities = np.arange(start=0.1, stop=0.31, step=0.1)
        angle_offsets = np.arange(start=-2*da, stop=3*da, step=da)
    else:
        dh = 3
        velocities = np.arange(start=0.1, stop=0.31, step=0.1)
        angle_offsets = np.arange(start=-da, stop=2*da, step=da)

    contact_heights = np.arange(
        start=bottle.height/dh, 
        stop=bottle.height + bottle.height/dh, 
        step=bottle.height/dh)
    
    for h in contact_heights:
        for v in velocities:
            for a in angle_offsets:
                action = Action(angle_offset=a, velocity=v, height=h)
                A.append(action)
            
    return A


def test_state_to_idx(mdp: MDP):
    # self.dx = self.dy = 0.1
    # self.xlim, self.ylim = 1.5, 1.5
    # self.X = np.arange(start=-self.xlim, stop=self.xlim, step=self.dx)
    # self.Y = np.arange(start=-self.ylim, stop=self.ylim, step=self.dy)
    x, y = mdp.xmin, mdp.ymin
    print(mdp.state_to_idx((x,y)))
    x, y = mdp.xmax, mdp.ymax
    print(mdp.state_to_idx((x,y)))
    x, y = 0, 0
    print(mdp.state_to_idx((x,y)))
    x, y = mdp.xmax + mdp.dx*0.9, mdp.ymax + mdp.dy*0.9
    print(mdp.state_to_idx((x,y)))


def test_bottle_dynamics(env):
    # [-0.6999999999999993, -0.29999999999999893, 0.1]
    fill_prop, lat_fric = 1.0, 0.1
    env.change_bottle_pos(new_pos=[-0.7, -0.3, 0.1])
    env.bottle.set_fill_proportion(fill_prop)
    env.bottle.lat_fric = lat_fric
    action = Action(0, 0.01, env.bottle.height / 2)
    env.run_sim(action)


def test(env, action):
    X = [0.5, 1.5]
    Y = np.arange(start=-0.5, stop=1.5, step=0.1)
    velocities = np.arange(start=0.1, stop=0.31, step=0.1)
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



# Two Ideas to fix:
# make  goal actually lie within reach of arm, but what about cost at each state?
# fix velocity by trying set max number of iters and just increasing by velocity and cap max reach

