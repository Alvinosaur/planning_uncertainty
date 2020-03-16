import math

# water bottle 
class Bottle:
    def __init__(self, table_height, bottle_r, bottle_h):
        self.radius = bottle_r  # 0.03175  # m
        self.height = bottle_h  # 0.1905   # m
        self.mass = 0.5        # kg
        self.start_pos = [0.5, 0, table_height+.1]
        self.start_ori = [0, 0, 0, 1]
        self.col_id = None
        self.default_fric = 0.1  # plastic-wood dynamic friction

class Arm:
    def __init__(self, start_pos, start_ori):
        # NOTE: taken from example: 
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        #lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        # self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        self.rp = [math.pi/4, (90 + 15)*math.pi/180, 0, 0, 0, 0, 0]
        #joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.max_force = 100000  # allow instantenous velocity = target velocity
        self.max_vel = 2*math.pi/16  # angular velocity
        self.rot_vel = self.max_vel

        self.start_pos = start_pos
        self.start_ori = start_ori
