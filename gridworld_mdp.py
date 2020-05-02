import numpy as np
import sys
from pprint import pprint

# Concepts: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf

# State space
R = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0],
    [0, -1, 1, -1, 10],
    [0, 0, 0, 0, 0],
    [-10, -10, -10, -10, -10]
]).astype(float)
V = np.zeros_like(R)
H, W = R.shape

# x,y
SET_STATES = set([(2,2), (2,4), (4,0), (4,1), (4,2), (4,3), (4,4)])
OBSTACLES = set([(1,1), (2,1), (2,3)])
NON_STATES = SET_STATES.union(OBSTACLES)

# Action space
A = [np.array(a).astype(float) for a in [(1,0), (-1,0), (0,1), (0,-1)]]
strA = ["Down", "Up", "Right", "Left"]

# Hyperparameters
noise = 0.1  # for stochastic environment with constant uncertainty
gamma = 0.9  # discount rate

def change_action(action, direction):
    if direction == 'left':
        rot = np.array([
            [0, -1],  # cos(90), -sin(90)
            [1, 0]    # sin(90), cos(90)
        ])
    else:  # right
        rot = np.array([
            [0, 1],  # cos(-90), -sin(-90)
            [-1, 0]    # sin(-90), cos(-90)
        ])
    return rot @ action

def is_legal(state):
    global V
    y, x = state
    H, W = V.shape
    return (0 <= x < W) and (0 <= y < H) and (
        tuple(state.astype(int)) not in OBSTACLES)

def take_action(state, action):
    return state + action

def print_policy(policy):
    H, W = len(policy), len(policy[0])
    policy_str = [[""]*W for h in range(H)]
    for y in range(H):
        for x in range(W):
            if (y, x) in NON_STATES:
                policy_str[y][x] = "None"
            else:
                action = policy[y][x]
                match_i = next(i for i,a in enumerate(A) 
                    if np.array_equal(a, action))
                policy_str[y][x] = strA[match_i]

    pprint(policy_str)

# def take_action_stochastic(state, action):
#     # constant stochastic environment: 80% chance move in desired dir
#     # 10% turn left, 10% turn right
#     p = np.random.random()
#     if p > 2*noise: return state + action
#     elif noise < p <= 2*noise: 
#         new_action = change_action(action, direction='left')
#     else:
#         new_action = change_action(action, direction='right')
#     return state + new_action

def get_value_and_reward(state, action, V):
    global R, gamma
    new_state = take_action(state, action)
    if not is_legal(new_state): 
        new_state = state  # if go out of bounds, no change
    y, x = new_state.astype(int)
    return R[y][x] + (gamma * V[y][x])

def update(V, state):
    global A
    best_value = 0
    best_action = None
    for action in A:
        # in gridworld, discrete action space so can just define
        # all other possible actions and transitions
        left_action = change_action(action, direction='left')
        right_action = change_action(action, direction='right')

        value = sum([
            (1-noise) * get_value_and_reward(state, action, V),
            noise * get_value_and_reward(state, right_action, V),
            noise * get_value_and_reward(state, left_action, V)
        ])

        if value > best_value or best_action is None:
            best_value = value
            best_action = action

    assert(best_action is not None)
    return best_value, best_action


# main loop
max_iters = 100
policy = [[None]*W for h in range(H)]
print("Reward mapping:")
print(R)
is_converged = False
min_change = 0.5
for iter in range(max_iters):
    total_change = 0
    # in normal value iteration, don't update on the fly
    oldV = np.copy(V)

    # loop through all states to update their V-value
    for y in range(H):
        for x in range(W):
            state = np.array([y,x])
            # don't update values for obstacles or set states
            if (y, x) in NON_STATES: continue
            V[y][x], policy[y][x] = update(oldV, state)
            total_change += abs(V[y][x] - oldV[y][x])
    if iter > 2 and total_change < min_change: break

print("Final values:")
print(V)
print("---------------------")
print("Final policy:")
print_policy(policy)