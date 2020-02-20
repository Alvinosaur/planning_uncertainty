import pybullet as p
import time
import pybullet_data

GRAVITY = -9.81
N = 10000  # simulation iterations
plane_urdf_filepath = "plane.urdf"
box_urdf_filepath = "r2d2.urdf"

# p.setTimeOut(max_timeout_sec)

physics_client = p.connect(p.GUI)  # or p.DIRECT for nongraphical version

# allows you to use pybullet_data package's existing URDF models w/out actually having them
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,GRAVITY)
planeId = p.loadURDF(plane_urdf_filepath)
cube_start_pos = [0, 0, 1]
cube_start_ori = p.getQuaternionFromEuler([0, 0, 0])
box_id = p.loadURDF(box_urdf_filepath, cube_start_pos, cube_start_ori)

for _ in range(N):
    ## Check connection maintained
    # status = p.getConnectionInfo(physics_client)
    # assert(status['isConnected'])
    p.stepSimulation()
    time.sleep(1./240.)

cube_pos, cube_ori = p.getBasePositionAndOrientation(box_id)
p.disconnect()
