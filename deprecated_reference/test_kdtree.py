import kdtree
import numpy as np


class AngleKDNode(kdtree.KDNode):
    def axis_dist(self, point, axis):
        """
        Changed to be shortest abs distance btwn two angles (int degrees)
        """
        diff = abs(self.data[axis] - point[axis]) % 360
        # This is either the distance or 360 - distance
        if diff > 180:
            return 360 - diff
        else:
            return diff


def create_angle_kdtree(dimensions):
    sel_axis = (lambda prev_axis: (prev_axis + 1) % dimensions)
    return AngleKDNode(sel_axis=sel_axis, axis=0, dimensions=dimensions)


tree = create_angle_kdtree(dimensions=3)
tree.add(np.array([45, 45, 45]))
tree.add(np.array([45, 45, 46]))
tree.add(np.array([42, 45, 45]))
tree.add(np.array([30, 45, 60]))
tree.add(np.array([35, 45, 55]))

for i in range(1, 5 + 1):
    neighbors = tree.search_knn((45, 45, 45), k=i)
    for (node, dist) in neighbors:
        print(node, dist)
