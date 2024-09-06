import numpy as np
import time
import sys
import os

# Get the file from the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot
from sample_approach_points import prune_arc, sample_hemisphere_suface_pts, hemisphere_orientations


class AlignHemisphere:
    def __init__(self, robot_urdf_path, robot_home_pos, renders: bool) -> None:
        self.pyb = PybUtils(self, renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)
        self.robot = LoadRobot(self.pyb.con, robot_urdf_path, [0.6, 0.2, 0], self.pyb.con.getQuaternionFromEuler([0, 0, 0]), robot_home_pos)

    def test(self):
        # Target position and orientation for the end-effector
        # look_at_point = [0, 0.8, 1.2]    # Point where the end-effector should face
        look_at_point = self.object_loader.prune_point_1_pos
        look_at_point_offset = 0.0
        num_points = [25, 25]

        # Visualize the look_at_point as a sphere
        # look_at_sphere = self.pyb.con.loadURDF("sphere2.urdf", look_at_point, globalScaling=0.05, useFixedBase=True)
        # self.pyb.con.changeVisualShape(look_at_sphere, -1, rgbaColor=[0, 1, 0, 1]) 

        hemisphere_pts = sample_hemisphere_suface_pts(look_at_point, look_at_point_offset, 0.1, num_points)
        hemisphere_oris = hemisphere_orientations(look_at_point, hemisphere_pts)

        # Iterate through position and orientation pairs
        iteration = 0

        manips = []
        best_joint_config = []
        best_ee_point = []

        max_manipulability = 0
        best_config = None
        for target_position, target_orientation in zip(hemisphere_pts, hemisphere_oris):
            print(iteration)
            # if iteration == 34:
            # Add a debug line between the two points
            line_id = self.pyb.con.addUserDebugLine(lineFromXYZ=target_position,
                                        lineToXYZ=look_at_point,
                                        lineColorRGB=[1, 0, 0],  # Red color
                                        lineWidth=2)


            poi_id = self.pyb.con.loadURDF("sphere2.urdf", target_position, globalScaling=0.041, useFixedBase=True)
            self.pyb.con.changeVisualShape(poi_id, -1, rgbaColor=[1, 0, 0, 1]) 

            joint_angles = self.robot.inverse_kinematics(target_position, target_orientation)

            self.robot.set_joint_positions(joint_angles)

            manipulability = self.robot.calculate_manipulability(joint_angles, planar=False)

            if manipulability > max_manipulability:
                # max_manipulability = manipulability
                best_config = joint_angles
                # best_ee_point = target_position

            manips.append(manipulability)
            best_joint_config.append(joint_angles)

            # print(f"Distance between hemisphere point and target: {np.linalg.norm(target_position - look_at_point)}")

            # Step simulation and render
            for _ in range(240):  # Adjust number of simulation steps as needed
                self.pyb.con.stepSimulation()
                time.sleep(1./240.)  # Sleep to match real-time

            # # Pause to view the result before moving to the next target
            # input("Press Enter to view the next target...")

            if iteration == 37:
                poi_id = self.pyb.con.loadURDF("sphere2.urdf", target_position, globalScaling=0.05, useFixedBase=True)
                self.pyb.con.changeVisualShape(poi_id, -1, rgbaColor=[1, 1, 1, 1]) 

            iteration += 1
        
        best_manip = np.argsort(manips)[-3:][::-1]
        print(best_manip)
        best_idx = best_manip[0]

        self.robot.reset_joint_positions(best_config)
        self.robot.set_joint_positions(best_config)

        while True:
            self.pyb.con.stepSimulation()


if __name__ == '__main__':
    render = True
    robot_home_pos = [0, -np.pi/2, 0, -np.pi/2, 0, 0]
    path_cache = AlignHemisphere(robot_urdf_path="./urdf/ur5e/ur5e_cart.urdf", robot_home_pos=robot_home_pos, renders=render)

    path_cache.test()
