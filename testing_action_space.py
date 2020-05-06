import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np

# clid = p.connect(p.SHARED_MEMORY)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
  #p.connect(p.SHARED_MEMORY_GUI)

p.loadURDF("plane.urdf", [0, 0, -0.3])
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints != 7):
  exit()

#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

p.setGravity(0, 0, 0)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1

useOrientation = 1
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 0
ikSolver = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 3

# set initial position
N = 500
init_pos = np.array([0.2, 0.2, 0.3])
pos = init_pos
init_joints = p.calculateInverseKinematics(kukaId,
    kukaEndEffectorIndex,
    pos,
    lowerLimits=ll,
    upperLimits=ul,
    jointRanges=jr,
    restPoses=rp)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
for i in range(numJoints):
    p.resetJointState(kukaId, i, init_joints[i])

threshold = 0.001
MAX_REACH = 1.2
TARGET_HEIGHT = 0.1
MIN_ITERS = 50
    
# loop through directions for arm to move towards
angles = np.arange(start=0, stop=math.pi+math.pi/4, step=math.pi/4)
for angle in angles:
    dir_vec = np.array([math.cos(angle), math.sin(angle), 0])
    velocity = 0.01
    pos = velocity * dir_vec + np.array([0, 0, TARGET_HEIGHT])
    prevPose = np.copy(pos)

    # reset arm state
    p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
    for i in range(numJoints):
        p.resetJointState(kukaId, i, init_joints[i])

    # iterate through action
    pos_change = 100000
    iter = 0
    while iter < N and not (iter > MIN_ITERS and pos_change < threshold):
        print(pos_change, threshold)
        iter += 1
        t = t + 0.01
        pos += velocity * dir_vec
        if np.linalg.norm(pos - init_pos) >  MAX_REACH:
            pos = MAX_REACH * dir_vec + np.array([0, 0, TARGET_HEIGHT])

        # jointPoses = p.calculateInverseKinematics(kukaId,
        #     kukaEndEffectorIndex,
        #     pos,
        #     lowerLimits=ll,
        #     upperLimits=ul,
        #     jointRanges=jr,
        #     restPoses=rp)
        jointPoses = p.calculateInverseKinematics(kukaId,
            kukaEndEffectorIndex,
            pos,
            solver=ikSolver)
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                              jointIndex=i,
                              controlMode=p.POSITION_CONTROL,
                              targetPosition=jointPoses[i],
                              targetVelocity=0,
                              force=500,
                              positionGain=0.03,
                              velocityGain=1)
    
        p.stepSimulation()
        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, 
            trailDuration)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
        pos_change = np.linalg.norm(prevPose1 - np.array(ls[4]))
        # print(pos_change)

        prevPose = np.copy(pos)
        prevPose1 = np.array(ls[4])
p.disconnect()