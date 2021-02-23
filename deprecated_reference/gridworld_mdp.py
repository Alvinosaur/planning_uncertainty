import numpy as np
import sys
from pprint import pprint
import heapq

# Concepts: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf

# State space
C = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, -1, 1, -10],
    [0, 0, 0, 0, 0],
    [10, 10, 10, 10, 10]
]).astype(float)
H, W = C.shape

# (y, x), SET_STATES are states that should not be evaluated since they have fixed value
# and should not have a defined action
# SET_STATES = set([(2, 2), (2, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)])
OBSTACLES = set([(1, 1), (2, 1), (2, 3)])
NON_STATES = OBSTACLES

# Action space
A = [np.array(a).astype(float) for a in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
strA = ["Down", "Up", "Right", "Left"]

# Hyperparameters
noise = 0.1  # for stochastic environment with constant uncertainty
gamma = 0.9  # discount rate


def change_action(action, direction):
    if direction == 'left':
        rot = np.array([
            [0, -1],  # cos(90), -sin(90)
            [1, 0]  # sin(90), cos(90)
        ])
    else:  # right
        rot = np.array([
            [0, 1],  # cos(-90), -sin(-90)
            [-1, 0]  # sin(-90), cos(-90)
        ])
    return rot @ action


def hash(state):
    # gridworld: get rid of trailing .0 decimal
    # continuous world: discretize using some dx, dy
    return tuple(state.astype(int))


def is_legal(state):
    y, x = state
    return (0 <= x < W) and (0 <= y < H)


def take_action(state, action):
    return state + action


def print_policy(policy):
    H, W = len(policy), len(policy[0])
    policy_str = [[""] * W for h in range(H)]
    for y in range(H):
        for x in range(W):
            action = policy[y][x]
            try:
                match_i = next(i for i, a in enumerate(A)
                           if np.array_equal(a, action))
                policy_str[y][x] = strA[match_i]
            except StopIteration:
                policy_str[y][x] = "None"

    pprint(policy_str)


def get_value_and_reward(state, action, V):
    new_state = take_action(state, action)
    if not is_legal(new_state):
        new_state = state
    y, x = hash(new_state)
    return C[y][x] + (gamma * V[y][x])


def heuristic(s1, s2):
    # Manhattan distance
    return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])


def stochastic_execution(state, action):
    p = np.random.random()  # sample from uniform [0, 1)
    if p < noise:
        action = change_action(action, direction='left')

    elif noise <= p < 2 * noise:
        action = change_action(action, direction='right')

    new_state = take_action(state, action)
    return new_state


def estimate_value(state, action, V):
    left_action = change_action(action, direction='left')
    right_action = change_action(action, direction='right')

    value = sum([
        (1 - 2 * noise) * get_value_and_reward(state, action, V),
        noise * get_value_and_reward(state, right_action, V),
        noise * get_value_and_reward(state, left_action, V)
    ])
    return value


def update(V, state):
    best_value = 0
    best_action = None
    for action in A:
        # in gridworld, discrete action space so can just define
        # all other possible actions and transitions
        value = estimate_value(state, action, V)
        if not is_legal(take_action(state, action)):
            continue

        # looking for minimum cost
        if value < best_value or best_action is None:
            best_value = value
            best_action = action

    assert (best_action is not None)
    return best_value, best_action


def value_iteration(online=False):
    # main loop
    V = np.zeros_like(C)
    max_iters = 500
    policy = [[None] * W for h in range(H)]
    print("Cost mapping:")
    print(C)
    is_converged = False
    min_change = 0.01
    # import ipdb
    # ipdb.set_trace()
    for iter in range(max_iters):
        total_change = 0
        # in normal value iteration, don't update on the fly
        oldV = np.copy(V)

        # loop through all states to update their V-value
        for y in range(H):
            for x in range(W):
                state = np.array([y, x])
                # don't update values for obstacles or set states
                if (y, x) in NON_STATES: continue
                if online:
                    V[y][x], policy[y][x] = update(V, state)
                else:
                    V[y][x], policy[y][x] = update(oldV, state)
                total_change += abs(V[y][x] - oldV[y][x])
        if iter > 2 and total_change < min_change: break

        print("Iter %d" % iter)
        print(V + C)

    print("Number of iterations: %d" % iter)
    print("Final values:")
    print(V + C)
    print("---------------------")
    print("Final policy:")
    print_policy(policy)


class Node(object):
    def __init__(self, state, value):
        self.state = state
        self.value = value

    def __lt__(self, other):
        return self.value > other.value

    def __repr__(self):
        return "[%.2f] (%d, %d)" % (self.value, self.state[0], self.state[1])


def RTDP(start, goal, online):
    # main loop
    V = np.zeros_like(C)

    # initialize value table using heuristic
    # if state-space too large, just use dictionary and use heuristic if no entry
    for row in range(H):
        for col in range(W):
            V[row][col] = heuristic((row, col), goal)

    max_iters = 500
    policy = [[None] * W for h in range(H)]
    print("Cost mapping:")
    print(C)
    is_converged = False
    min_change = 0.01
    # import ipdb
    # ipdb.set_trace()
    for iter in range(max_iters):
        import ipdb
        ipdb.set_trace()
        # in normal value iteration, don't update on the fly
        oldV = np.copy(V)

        # initialize state = start, then follow greedy policy until reach goal
        state = start
        visited = {hash(start)}
        while True:
            (y, x) = hash(state)
            print(state)

            # find best action and expected value
            if online:
                V[y][x], policy[y][x] = update(V, state)
            else:
                V[y][x], policy[y][x] = update(oldV, state)

            # execute this best action, emulate "stochastic" by just sampling action
            new_state = stochastic_execution(state, policy[y][x])

            if is_legal(new_state) and hash(new_state) not in NON_STATES and hash(new_state) not in visited:
                state = new_state
                visited.add(hash(state))

            if hash(new_state) == hash(goal):
                break

        # check total value change
        total_change = 0
        for state in visited:
            (y, x) = state
            total_change += abs(V[y][x] - oldV[y][x])

        if iter > 2 and total_change < min_change: break

        print("Iter %d" % iter)
        print(V + C)

    print("Number of iterations: %d" % iter)
    print("Final values:")
    print(V + C)
    print("---------------------")
    print("Final policy:")
    print_policy(policy)


if __name__ == "__main__":
    online = False
    # value_iteration(online=online)  # False -> 17 iters, True -> 12 iters

    start = np.array((3, 0))
    goal = np.array((2, 4))
    RTDP(start=start, goal=goal, online=online)
