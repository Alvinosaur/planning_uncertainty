How to interpret these graphs:
Legend of the different colors shows different max_force levels allotted for
pybullet's setJointMotorControl2() function. The goal was to find the minimum
amount of force needed to still have decent performance of reaching target joint
values. The large zigzag blue line for each plot shows behavior with force=0,
which is undefined and basically means the arm just falls down.

x-axis is timestep for the simulation run, dictated by min_iters allotted for
the simulation. The tradeoff here is having small number of iterations for fast
simulation and thus fast planning, but also have enough iters to have realistic
behavior and allow arm complete action.

From the  plots, it looks like velocity is smooth, but torque/accel is
piecewise, so jerk is discontinuous.
