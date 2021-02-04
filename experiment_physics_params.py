import pybullet as p
import pybullet_data
import numpy as np
import math

from sim_objects import Bottle, Arm
from environment import Environment, EnvParams, StateTuple
import matplotlib.pyplot as plt

# constants
GRAVITY = -9.81
dtheta = 8  # degrees

# params
VISUALIZE = False

fill = 1.0
friction = 0.08
sim_params = EnvParams(bottle_fill=fill, bottle_fric=friction,
                       bottle_fill_prob=1.0, bottle_fric_prob=1.0)

if VISUALIZE:
    p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5, cameraYaw=145, cameraPitch=-10,
        cameraTargetPosition=[0, 0, 0])
else:
    p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, GRAVITY)
p.loadURDF(Environment.plane_urdf_filepath, basePosition=[0, 0, 0])

# Create Arm, Bottle and set up Environment
bottle = Bottle()
arm = Arm(kuka_id=p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0]))
env = Environment(arm, bottle, is_viz=VISUALIZE)

bottle_pos = np.array([0.17, 0.57, 0.10])
joints = np.array([1.66433097, 1.65043968, 2.36681825, 0.86160736,
                   0.89415987, 2.09438084, 0.03126315])
state = StateTuple(bottle_pos=bottle_pos, bottle_ori=[0, 0, 0, 1.0],
                   joints=joints)
action = (np.array([-0.16755161, -0., -0., -0., -0., -0., -0.]),
          200)

# frictions = [0.2]
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_ylim([0.45, 0.6])
ax.set_xlim([0.15, 0.4])
fills = np.linspace(start=0.2, stop=1.0, endpoint=True, num=9)  # 0.2, 0.3, ... 1.0
# frics = [0.05, 0.09, 0.12, 0.14, 0.15, 0.151, 0.152]
frics = []
frics += np.linspace(start=0.06, stop=0.12, num=16).tolist()
fric_counts = np.zeros_like(frics)
# for fill in fills:
for i, fric in enumerate(frics):
    sim_params.bottle_fric = fric
    sim_params.bottle_fill = fill
    results = env.run_sim(action=action, sim_params=sim_params, state=state)
    x, y, z = results.bottle_pos
    fric_counts[i] += results.is_fallen
    print("fric(%.5f), x(%.3f), y(%.3f), fall(%d)" % (fric, x, y, results.is_fallen))
    ax.scatter([x], [y], s=10, marker="o", label='%.5f' % fric)

# print(frics)
# print(fric_counts / len(fills))

plt.title("Pos vs bottle fric @ fill=%.2f" % sim_params.bottle_fill)
plt.legend(loc='upper left')
plt.show()

# No noticeable change from different bottle fill parameters at low friction
# fills = [0.2, 0.4, 0.6, 0.8, 1.0]
# [0.26415718 0.53237922 0.12825468]
# [0.26555587 0.53066559 0.12824613]
# [0.26363408 0.53215477 0.12824886]
# [0.2632767  0.53188981 0.12824921]
# [0.26412209 0.53170024 0.12824733]
# BUT at higher friction values, choice of fill seems to matter much more
# Also seems like having a mean of 0.146 friction is best
# 2nd std dev should be +/- 0.08


"""
Frictions:
[0.13, 0.132, 0.134, 0.136, 
0.138, 0.14, 0.142, 0.144, 
 0.146, 0.148, 0.15, 0.152, 
 0.154, 0.156, 0.158, 0.16]
 
Their respective average failure counts varied across different bottle fills:
[0.         0.         0.         0.         
0.         0.         0.         0.         
0.33333333 0.44444444 0.55555556 0.66666667
 0.66666667 0.77777778 0.77777778 0.77777778]
"""
