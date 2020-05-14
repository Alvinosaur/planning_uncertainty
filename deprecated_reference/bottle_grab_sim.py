
# Source: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_kuka_diverse_object_grasping.py

import os, inspect
import numpy as np
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

# To use RL model, follow source in above link
# possibly create subclass of KukaGymEnv to define our own simulator functions
# maybe change the reward function that is being used

# relevant links
# https://github.com/bulletphysics/bullet3/blob/ed11038362b7f79c8062b033d4e57f6e3b6f0db3/examples/pybullet/gym/pybullet_envs/bullet/kuka.py#L101
# https://github.com/bulletphysics/bullet3/blob/ed11038362b7f79c8062b033d4e57f6e3b6f0db3/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py#L24

def main():
    env = KukaGymEnv(renders=True, isDiscrete=True)
    
    while True:
        # kuka gym manipulates a box with arm and gripper
        # observation = [boxX_wrt_gripper, boxY_wrt_gripper, boxOrn_wrt_gripper]
        dx = 0.1
        dy = 0
        da = -0.05
        fingerAngle = math.pi/4
        # action = 6 # [0, 6]
        # env.render()
        # for Discrete mode, distinct number of actions to take
        # dx = [0, -dv, dv, 0, 0, 0, 0][action]
        # dy = [0, 0, 0, -dv, dv, 0, 0][action]
        # da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
        # pass action index into step()
        action = [0, 0, 0, da, fingerAngle]
        obs, rew, done, _ = env.step2(action)
        # input()
        # obs = env.reset()

        # or define your own motion with step2()
        # da = d(endEffectorAngle)
        # motorCommands = [dx, dy, -0.002, da, fingerAngle]
        # can change other parts of finger: maxForce, 
        # print("Episode reward", episode_rew)


if __name__ == '__main__':
  main()