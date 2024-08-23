import numpy as np
import pybullet as p
import pybullet_data


class LoadObjects:
    def __init__(self, con) -> None:
        self.con = con
        self.load_objects()

    def load_urdf(self, urdf_name, start_pos=[0, 0, 0], start_orientation=[0, 0, 0], fix_base=True, radius=None):
        orientation = self.con.getQuaternionFromEuler(start_orientation)
        if radius is None:
            objectId = self.con.loadURDF(urdf_name, start_pos, orientation, useFixedBase=fix_base)
        else:
            objectId = self.con.loadURDF(urdf_name, start_pos, globalScaling=radius, useFixedBase=fix_base)
            p.changeVisualShape(objectId, -1, rgbaColor=[0, 1, 0, 1]) 
        return objectId

    def load_objects(self):
        self.planeId = self.load_urdf("plane.urdf")

        self.start_x = 0.5
        self.start_y = 1
        self.prune_point_0_pos = [self.start_x, self.start_y, 1.55] 
        self.prune_point_1_pos = [self.start_x, self.start_y - 0.05, 1.1] 
        self.prune_point_2_pos = [self.start_x, self.start_y + 0.05, 0.55] 
        self.radius = 0.05 

        self.leader_branchId = self.load_urdf("./urdf/leader_branch.urdf", [0, self.start_y, 1.6/2])
        self.top_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, self.start_y, 1.5], [0, np.pi / 2, 0])
        self.mid_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, self.start_y, 1], [0, np.pi / 2, 0])
        self.bottom_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, self.start_y, 0.5], [0, np.pi / 2, 0])
        self.collision_objects = [self.leader_branchId, self.top_branchId, self.mid_branchId, self.bottom_branchId, self.planeId]

        self.prune_point_0 = self.load_urdf("sphere2.urdf", self.prune_point_0_pos, radius=self.radius)
        self.prune_point_1 = self.load_urdf("sphere2.urdf", self.prune_point_1_pos, radius=self.radius)
        self.prune_point_2 = self.load_urdf("sphere2.urdf", self.prune_point_2_pos, radius=self.radius)
