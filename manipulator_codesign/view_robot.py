import os
import argparse
import numpy as np
import time
import pybullet as p
from scipy.spatial.transform import Rotation as R
from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_objects import LoadObjects
from pybullet_robokit.load_robot import LoadRobot
from pybullet_robokit.motion_planners import KinematicChainMotionPlanner
import manipulator_codesign.orchard_workspace as orchard_ws
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.urdf_to_decision_vector import urdf_to_decision_vector, encode_seed


def get_urdf_path(user_input, default_dir, default_file):
    """
    Determines the correct URDF file path.
    
    Args:
        user_input (str): The user-provided URDF path or filename.
        default_dir (str): The default directory where URDF files are stored.
        default_file (str): The default URDF filename.
    
    Returns:
        str: The resolved URDF file path.
    """
    if os.path.isabs(user_input) and os.path.isfile(user_input):
        return user_input
    
    potential_path = os.path.join(default_dir, user_input)
    if os.path.isfile(potential_path):
        return potential_path
    
    return os.path.join(default_dir, default_file)


class ViewRobot:
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.01, renders=True, ee_link_name='ee_link'):
        """
        Initialize the ViewRobot class.

        Args:
            robot_urdf_path (str): Path to the URDF file of the robot.
            robot_home_pos (list): Home position of the robot joints.
            ik_tol (float, optional): Tolerance for inverse kinematics. Defaults to 0.01.
            renders (bool, optional): Whether to visualize the robot in the PyBullet GUI. Defaults to True.x
        """
        self.pyb = PybUtils(renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)

        self.robot_urdf_path = robot_urdf_path

        self.ik_tol = ik_tol

        # robot_to_amiga_translation = [0, 0, 1.025]
        robot_to_amiga_translation = [0, 0, 0]
        amiga_to_robot_translation = [0, -0.3, 0]
        self.robot_system_translation = [-4.0, -2.0, 0]

        self.robot_translation = np.add(robot_to_amiga_translation, self.robot_system_translation)
        self.robot = LoadRobot(self.pyb.con, 
                               robot_urdf_path, 
                               self.robot_translation, 
                               self.pyb.con.getQuaternionFromEuler([0, 0, 0]), 
                               robot_home_pos, 
                               collision_objects=self.object_loader.collision_objects,
                               ee_link_name=ee_link_name)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_dir = os.path.join(script_dir, 'urdf')
        flags = 0 
        # self.amiga_id = self.object_loader.load_urdf(os.path.join(urdf_dir, "robots/amiga.urdf"),
        #                                 start_pos=np.add(amiga_to_robot_translation, self.robot_system_translation), 
        #                                 start_orientation=[0, 0, 0], 
        #                                 fix_base=True,
        #                                 flags=flags)
        # self.object_loader.collision_objects.append(self.amiga_id)

        pretty_mesh = None
        if pretty_mesh:
            filename = "/manipulator_codesign/manipulator_codesign/meshes/Pair01_before_mesh.obj"
            # Load tree collision shape
            collision_shape = self.robot.con.createCollisionShape(
                shapeType=self.robot.con.GEOM_MESH,
                fileName=filename,
                flags=self.robot.con.GEOM_FORCE_CONCAVE_TRIMESH,
            )
            # Create a visual shape with light brown color (RGBA)
            visual_shape = self.robot.con.createVisualShape(
                shapeType=self.robot.con.GEOM_MESH,
                fileName=filename,
                rgbaColor=[0.71, 0.40, 0.16, 1.0]  # light brown, alpha=1.0
            )
            # Create your body using this multi‐hull collision shape
            body_id = self.robot.con.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[-5.7, -1.905, 0]
            )
            self.object_loader.collision_objects.append(body_id)
        elif pretty_mesh == False:
            filename = "/manipulator_codesign/manipulator_codesign/meshes/before_mesh_transformed.obj"
            # Load tree collision shape
            collision_shape = self.robot.con.createCollisionShape(
                shapeType=self.robot.con.GEOM_MESH,
                fileName=filename,
                flags=self.robot.con.GEOM_FORCE_CONCAVE_TRIMESH,
            )
            # Create a visual shape with light brown color (RGBA)
            visual_shape = self.robot.con.createVisualShape(
                shapeType=self.robot.con.GEOM_MESH,
                fileName=filename,
                rgbaColor=[0.71, 0.40, 0.16, 1.0]  # light brown, alpha=1.0
            )
            # Create your body using this multi‐hull collision shape
            body_id = self.robot.con.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0,0,0]
            )
            self.object_loader.collision_objects.append(body_id)
        else:
            pass

    def cartesian_path_test(self):
        # Euler angles to point the end-effector in the y-direction
        orientation = list(self.pyb.con.getQuaternionFromEuler([-np.pi/2, 0, 0]))

        # Define start and end positions
        start_pos = [0.5, 0.75, 0.5]
        end_pos = [-0.5, 0.75, 0.5]

        # Create start and end poses as (position, orientation)
        start_pose = (start_pos, orientation)
        end_pose = (end_pos, orientation)

        print("Start Pose:", start_pose)
        print("End Pose:", end_pose)

        # Interpolate a list of waypoint poses between start_pose and end_pose
        num_points = 10
        waypoints = []
        for t in np.linspace(0, 1, num_points):
            interp_pos = (np.array(start_pos) * (1 - t) + np.array(end_pos) * t).tolist()

            # Orientation remains constant
            waypoints.append((interp_pos, orientation))

        # Call the plan_cartesian_motion_path function with the list of waypoint poses
        joint_path = self.robot.plan_cartesian_motion_path(waypoints, max_iterations=10000)

        if joint_path is None:
            print("Cartesian path planning failed.")
            return

        # Execute the planned joint path
        self.robot.set_joint_path(joint_path)

    def test_resolved_rate_motion_control(self):
        # input("Press Enter to start the resolved rate motion control test...")
        target_pos = [-4, -1, 1.5]

        target_point_id = self.object_loader.load_urdf(
            "sphere2.urdf",
            start_pos=target_pos,
            start_orientation=[0, 0, 0],
            fix_base=True,
            radius=0.05
        )

        motion_planner = KinematicChainMotionPlanner(self.robot)

        target_orientations = [
            np.array([180, 0, 90]), # top-down (-z)

            np.array([90, 0, 180]), # front-back (+y)

            np.array([0, 0, -90]), # bottom-up (+z)

            np.array([90, 0, -90]), # right-left (-x)

            np.array([90, 0, 180]), # front-back (+y)

            np.array([90, 0, 90]), # left-right (+x)
        ]

        for target_ori in target_orientations:
            target_ori = R.from_euler('xyz', target_ori, degrees=True).as_quat()
            
            q_final, manip_score, delta_joint_score, pose_error = motion_planner.resolved_rate_control(
                                                                        (target_pos, target_ori), 
                                                                        max_steps=400,
                                                                        plot_manipulability=False, 
                                                                        alpha=0.75,
                                                                        beta=0.75,
                                                                        damping_lambda=0.15, 
                                                                        manipulability_gain=0.1, 
                                                                        stall_vel_threshold=0.1, 
                                                                        stall_patience=10)
            print("\nManipulability score:", manip_score)
            print("Delta joint score:", delta_joint_score)
            print("Position error:", pose_error[0])
            print("Orientation error:", pose_error[1])
            print()

            # if q_final:
            #     self.robot.set_joint_configuration(q_final)
        
        print("Test complete. Press Ctrl+C to exit.")
        while True:
            self.pyb.con.stepSimulation()

    def rrt_path_test(self):
        # Define the window position and size for filtering prune points and point cloud data
        window_x_pos = self.robot_system_translation[0]
        window_size = 3

        # Start configuration is the robots home position
        start_config = self.robot.home_config

        # Get IK to the target position
        # target_pos = [-4, 1.25, 1.5]
        # target_ori = np.array([90, 0, 180])
        # target_ori = R.from_euler('xyz', target_ori, degrees=True).as_quat()
        # target_pose = (target_pos, target_ori)
        pos=[-3.9119461,  -0.7460511,  1.4021448 ]
        quat=[0.3495689,  0.04118999, 0.67365077, 0.64984584]

        target_pose = (pos, quat)
        target_config = self.robot.inverse_kinematics(target_pose, pos_tol=self.ik_tol, max_iter=1000, resample=True, num_resample=10)

        target_point_id = self.object_loader.load_urdf(
                "sphere2.urdf",
                start_pos=pos,
                start_orientation=[0, 0, 0],
                fix_base=True,
                radius=0.05
            )

        # Initialize the motion planner
        motion_planner = KinematicChainMotionPlanner(self.robot)

        # Pass the start and target configurations to the RRT planner
        joint_path = motion_planner.rrt_path(start_config, target_config, rrt_iter=100000, collision_objects=self.object_loader.collision_objects, steps=None)

        # Check if the path is valid
        if joint_path is None:
            print("\nRRT path planning failed.\n")
            return
        else:
            print("\nRRT path planning succeeded.\n")

        print(len(joint_path), "steps in the path")

        # Execute the planned joint path
        self.robot.set_joint_path(joint_path)

        print("Test complete. Press Ctrl+C to exit.")
        while True:
            self.pyb.con.stepSimulation()

    def test_collisions(self):
        # Define joint configuration for the robot
        # joint_config = [0, 0, 0, 0, 0, 0]
        joint_config = self.robot.home_config
        # joint_config = [0.0, 0.95]

        self.robot.set_joint_configuration(joint_config)
        self.robot.reset_joint_positions(joint_config)

        self.robot.detect_all_self_collisions(self.robot.robotId)
        self.robot.print_robot_environment_contacts(self.object_loader.collision_objects)

        # Check for collisions
        print('Self-collision check:')
        print('AABB: ', self.robot.check_collision_aabb(self.robot.robotId, self.robot.robotId))
        print('General: ', self.robot.collision_check(self.robot.robotId, collision_objects=self.object_loader.collision_objects))
        print('Self: ', self.robot.check_self_collision(self.robot.home_config))

        print("\nTest complete. Press Ctrl+C to exit.\n")
        while True:
            self.pyb.con.stepSimulation()

    def get_filtered_prune_points(self, window_size=2.5):
        # Load the prune points from the YAML file
        pose_data_results = orchard_ws.get_prune_poses_from_yaml(
            yaml_path="manipulator_codesign/prune_data/all_branches_info.yaml",
            robot_base=self.robot_translation,
            window_size=window_size,
            min_y=None,
            max_y=0
        )
        target_poses, target_offset_poses = orchard_ws.package_poses(pose_data_results)

        # Remove the robot if it was previously loaded (planning to use the chain version)
        self.pyb.con.removeBody(self.robot.robotId)

        raw_robot_params = urdf_to_decision_vector(self.robot_urdf_path)
        x = encode_seed(raw_robot_params, min_joints=5, max_joints=7)
        n, types, axes, lengths = decode_decision_vector(x, min_joints=5, max_joints=7)
        chain = KinematicChainPyBullet(self.pyb.con, self.robot_translation, n, types, axes, lengths, collision_objects=self.object_loader.collision_objects)
        chain.build_robot()
        chain.load_robot()

        # if target_poses has a None type, print the index and remove it
        # target_poses = [pose for pose in target_poses if pose is not None]
        poses_collision_free = chain.sample_collision_free_poses(target_poses)
        return poses_collision_free, chain
    
    def test_prune_point_orientation(self):
        self.pyb.con.removeBody(self.robot.robotId)

        poses_collision_free, chain = self.get_filtered_prune_points(window_size=2.5)

        # create a red sphere visual
        target_point_id = None
        radius = 0.05
        red = [1, 0, 0, 1]  # RGBA
        visual_shape_id = self.robot.con.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=red
        )
        # Iterate through the filtered prune points and visualize them
        for pose in poses_collision_free:
            input("Press Enter to visualize the next prune point...")

            if target_point_id is not None:
                # Remove the previous target point after reaching it
                self.pyb.con.removeBody(target_point_id) 

            # Spawn a little marker at that prune point
            target_point_id = p.createMultiBody(
                baseMass=0,  # 0 = static
                baseVisualShapeIndex=visual_shape_id,
                basePosition=pose[0]  # x, y, z
            )

            # Reset robot to home, then move to target via IK
            chain.robot.set_joint_configuration(chain.robot.home_config)
            joint_config = chain.robot.inverse_kinematics(
                pose,
                pos_tol=self.ik_tol,
                max_iter=1000,
                resample=False
            )
            chain.robot.reset_joint_positions(joint_config)
            chain.robot.set_joint_configuration(joint_config)

            # Step simulation and render
            for _ in range(240):
                # self.pyb.con.stepSimulation()
                time.sleep(1.0 / 240.0)

    def basic_prune_point_viz(self):
        # create a red sphere visual
        radius = 0.05
        red = [1, 0, 0, 1]  # RGBA
        visual_shape_id = self.robot.con.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=red
        )

        self.pyb.con.removeBody(self.robot.robotId)
        self.pyb.con.removeBody(self.amiga_id)

        prune_pts = [
            (-3.1, -0.51, 0.35), 
            (-3.5, -1.9, 2.85), 
            (-4.825, -0.3, 0.95), 
            (-4.125, -0.4, 0.65), 
            (-5.0, -0.14, 1.22), 
            (-4.6, -1.45, 3.0), 
            (-3.4, -0.175, 0.5), 
        ]

        for pt in prune_pts:
            # create a multibody to actually place it in the sim
            sphere_id = p.createMultiBody(
                baseMass=0,  # 0 = static
                baseVisualShapeIndex=visual_shape_id,
                basePosition=pt  # x, y, z
            )

        # Step simulation and render
        while True:
            self.pyb.con.stepSimulation()

    def command_to_joint_positions(self, joint_positions):
        """
        Command the robot to move to the specified joint positions.

        Args:
            joint_positions (list): List of joint positions to move the robot to.
        """
        # if len(joint_positions) != self.robot.num_joints:
        #     raise ValueError(f"Expected {self.robot.num_joints} joint positions, got {len(joint_positions)}")
        print(len(self.robot.controllable_joint_idx))
        self.robot.set_joint_configuration(joint_positions)
        self.robot.reset_joint_positions(joint_positions)

        # Step simulation and render
        while True:
            self.pyb.con.stepSimulation()

    def main(self):
        target_positions = np.random.uniform(low=[0, 0, 0], high=[2.0, 2.0, 2.0], size=(20, 3)).tolist()
        target_point_id = None

        for i, position in enumerate(target_positions):
            input("Press Enter to continue...")

            if target_point_id is not None:
                # Remove the previous target point after reaching it
                self.pyb.con.removeBody(target_point_id) 
        
            target_point_id = self.object_loader.load_urdf("sphere2.urdf", 
                                        start_pos=position, 
                                        start_orientation=[0, 0, 0], 
                                        fix_base=True, 
                                        radius=0.05)
            
            # Set the robot to its home position
            self.robot.set_joint_configuration(self.robot.home_config)
        
            # Move arm to the target pose using inverse kinematics
            joint_config = self.robot.inverse_kinematics(position, pos_tol=self.ik_tol, max_iter=1000, resample=False)
            self.robot.reset_joint_positions(joint_config)
            self.robot.set_joint_configuration(joint_config)

            # Step simulation and render
            for _ in range(240):  # Adjust number of simulation steps as needed
                self.pyb.con.stepSimulation()
                time.sleep(1./240.)  # Sleep to match real-time
            
        while True:
            self.pyb.con.stepSimulation()

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a robot in PyBullet simulation.")
    parser.add_argument("-u", "--urdf_path", type=str, default="example_6dof_manipulator.urdf", 
                        help="URDF file path or name (default: example_6dof_manipulator.urdf)")
    parser.add_argument("--render", action="store_true", help="Enable rendering in PyBullet")
    parser.add_argument("--no-render", action="store_false", dest="render", help="Disable rendering in PyBullet")
    
    parser.set_defaults(render=True)  # Default to True
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf_dir = os.path.join(script_dir, 'urdf', 'robots')
    default_urdf_file = os.path.join(default_urdf_dir, "example_6dof_manipulator.urdf")
    
    robot_urdf_path = get_urdf_path(args.urdf_path, default_urdf_dir, default_urdf_file)
    ee_link_name = 'end_effector' if "best_chain" or 'test_robot' in robot_urdf_path else 'gripper_link'
    # ee_link_name = 'tool0'
    
    robot_home_pos = None

    print(f"\nLoading robot from: {robot_urdf_path}\n")
    
    view_robot = ViewRobot(robot_urdf_path=robot_urdf_path, 
                           renders=args.render,
                           robot_home_pos=robot_home_pos,
                           ik_tol=0.01,
                           ee_link_name=ee_link_name)
    
    # view_robot.test_resolved_rate_motion_control()
    # view_robot.main() 
    # view_robot.rrt_path_test()
    # view_robot.test_collisions()
    # view_robot.test_prune_point_orientation()
    # view_robot.basic_prune_point_viz()
    # view_robot.command_to_joint_positions([0.25, -0.25, 0.5, -1.2, -0.5, 0.2, -1.0, 0.5, -1.5]) # For gen_seed_6
    # view_robot.command_to_joint_positions([-2, 0.5, -0.3, 0.2, 1.0, 0.5, -0.5]) # For gen_seed_0
    # view_robot.command_to_joint_positions([0, -0.25]) # For test_robot_7
    # view_robot.command_to_joint_positions([0, -0.25, 0, 0]) # For test_robot_8
    # view_robot.command_to_joint_positions([0, 0.5]) # For test_robot_9
    view_robot.command_to_joint_positions([0.25, 0.1]) # For test_robot_10