import numpy as np
from tabulate import tabulate
from scipy.linalg import svdvals
# import roboticstoolbox as rtb
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R
from manipulator_codesign.urdf_gen import URDFGen
from pybullet_robokit.load_robot import LoadRobot
from pybullet_robokit.motion_planners import KinematicChainMotionPlanner


# Base class with shared parameters and methods
class KinematicChainBase:
    def __init__(self, num_joints, joint_types, joint_axes, link_lengths, 
                 robot_name='silly_robot', save_urdf_dir=None,
                 joint_limit_prismatic=(-0.5, 0.5), joint_limit_revolute=(-2*np.pi, 2*np.pi)):
        """
        Initialize the kinematic chain for the robot.

        Args:
            num_joints (int): Number of joints in the kinematic chain.
            joint_types (list of int): Types of joints (0 for prismatic, 1 for revolute).
            joint_axes (list of list of float): Axes of each joint.
            link_lengths (list of float): Lengths of each link in the kinematic chain.
            robot_name (str, optional): Name of the robot. Defaults to 'silly_robot'.
            save_urdf_dir (str, optional): Directory to save the URDF file. Defaults to None.
            joint_limit_prismatic (tuple of float, optional): Joint limits for prismatic joints. Defaults to (-0.5, 0.5).
            joint_limit_revolute (tuple of float, optional): Joint limits for revolute joints. Defaults to (-np.pi, np.pi).
        """
        self.num_joints = num_joints
        self.joint_types = joint_types
        self.joint_axes = joint_axes
        self.link_lengths = link_lengths
        self.robot_name = robot_name
        self.joint_limit_prismatic = joint_limit_prismatic
        self.joint_limit_revolute = joint_limit_revolute
        self.joint_limits = [
            self.joint_limit_prismatic if jt == 0 else self.joint_limit_revolute 
            for jt in joint_types
        ]

        self.save_urdf_dir = save_urdf_dir
        self.urdf_gen = URDFGen(self.robot_name, self.save_urdf_dir)

    def create_urdf(self):
        """
        Generates a URDF (Unified Robot Description Format) representation of the manipulator.

        This method uses the URDF generator to create a URDF file for the manipulator based on
        the provided joint axes, joint types, link lengths, and joint limits.
        """
        self.urdf_gen.create_manipulator(self.joint_axes, self.joint_types, self.link_lengths, self.joint_limits, collision=True)
    
    def save_urdf(self, filename):
        """
        Save the URDF (Unified Robot Description Format) file.

        Args:
            filename (str): The name of the file where the URDF will be saved.
        """
        self.urdf_gen.save_urdf(filename)

    def describe(self):
        """
        Prints a description of the kinematic chain, including the number of joints,
        joint types, joint axes, and link lengths.
        """
        print("\nKinematic Chain Description:")
        headers = ["Joint Number", "Joint Type", "Joint Axis", "Link Length"]
        joint_types_mapped = [self.urdf_gen.map_joint_type(joint_types) for joint_types in self.joint_types]
        table_data = zip(list(range(1, self.num_joints + 1)), joint_types_mapped, self.joint_axes, np.round(self.link_lengths, 3))
        print(tabulate(table_data, headers=headers, tablefmt="grid", colalign=("center", "center", "center", "center")))

    def compute_fitness(self, target):
        """
        Compute the error between the chain’s end-effector and a target position.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


# # --- Robotics Toolbox Implementation ---
# class KinematicChainRTB(KinematicChainBase):
#     def __init__(self, num_joints, joint_types, joint_axes, link_lengths, **kwargs):
#         super().__init__(num_joints, joint_types, joint_axes, link_lengths, **kwargs)
#         # Build the robot using the DH convention (RTB)
#         links = []
#         for i in range(self.num_joints):
#             if self.joint_types[i] == 1:  # Revolute joint
#                 links.append(rtb.RevoluteDH(
#                     a=self.link_lengths[i] if self.joint_axes[i] in ['x', 'y'] else 0,
#                     d=self.link_lengths[i] if self.joint_axes[i] == 'z' else 0,
#                     alpha=0, offset=0,
#                     qlim=self.joint_limits[i]
#                 ))
#             else:  # Prismatic joint
#                 links.append(rtb.PrismaticDH(
#                     a=self.link_lengths[i] if self.joint_axes[i] in ['x', 'y'] else 0,
#                     theta=0, alpha=0, offset=0,
#                     qlim=(0, self.link_lengths[i])
#                 ))
#         self.robot = rtb.DHRobot(links, name=self.robot_name)

#     def compute_fitness(self, target):
#         # Use RTB’s inverse kinematics method.
#         ik_solution = self.robot.ikine_LM(SE3(target), tol=0.01, slimit=100, ilimit=10, joint_limits=True)
#         return ik_solution.residual  # Lower is better


# --- PyBullet Implementation ---
class KinematicChainPyBullet(KinematicChainBase):
    def __init__(self, pyb_con, start_position, num_joints, joint_types, joint_axes, link_lengths, ee_link_name='end_effector', collision_objects=[], **kwargs):
        """
        pyb_con: A connection object from your PyBullet utilities.
        """
        super().__init__(num_joints, joint_types, joint_axes, link_lengths, **kwargs)
        self.pyb_con = pyb_con
        self.start_position = start_position
        self.ee_link_name = ee_link_name
        self.urdf_path = None
        self.robot = None
        self.collision_objects = collision_objects

        self.mean_pose_error = None
        self.mean_torque = None
        self.global_conditioning_index = None
        self.target_joint_positions = None
        self.mean_rrt_path_cost = None
        self.mean_manip_score_rrmc = None
        self.mean_delta_joint_score_rrmc = None
        self.mean_pos_error_rrmc = None
        self.mean_ori_error_rrmc = None

        self.is_built = False
        self.is_loaded = False

        self.default_joint_config = [0.0] * self.num_joints

    def build_robot(self):
        # Create a URDF for this chain.
        self.create_urdf()
        # Save a temporary URDF file to load the robot.
        self.urdf_path = self.urdf_gen.save_temp_urdf()
        self.is_built = True
    
    def load_robot(self):
        # Load the robot into PyBullet.
        self.robot = LoadRobot(self.pyb_con, 
                               self.urdf_path, 
                               start_pos=self.start_position, 
                               start_orientation=self.pyb_con.getQuaternionFromEuler([0, 0, 0]),
                               home_config=self.default_joint_config,
                               ee_link_name=self.ee_link_name,
                               collision_objects=self.collision_objects)
        self.is_loaded = True

        # Initialize motion planner
        self.motion_planner = KinematicChainMotionPlanner(self.robot)

    def compute_chain_metrics(self, targets, targets_offset):
        # Compute the mean pose error and mean torque for the given targets.
        pose_errors, self.target_joint_positions = zip(*[self.compute_pose_fitness(target) for target in targets_offset])
        self.mean_pose_error = np.mean(pose_errors)

        # Compute the rrt path cost for the target joint positions with the tree collision mesh.
        rrt_path_costs = [self.compute_rrt_path_cost(joint_positions, collision_objects=self.collision_objects) for joint_positions in self.target_joint_positions]
        self.mean_rrt_path_cost = np.mean(rrt_path_costs)

        # Remove the last collision object (assumed to be the tree mesh). This is necessary to get global metrics of torque, GCI, and manipulability.
        self.pyb_con.removeBody(self.collision_objects[-1])  

        self.mean_torque = np.mean([self.compute_gravity_torque_magnitute(joint_positions) for joint_positions in self.target_joint_positions])

        # Compute the Global Conditioning Index (GCI) for the kinematic chain.
        self.global_conditioning_index = self.compute_global_conditioning_index(num_samples=50)

        # Compute the manipulability score and delta joint score using resolved-rate motion control.
        final_configs, manip_scores, delta_joint_scores, pose_errors_rrmc = zip(*[self.compute_resolved_rate_motion_control_fitness(target) for target in targets])
        self.mean_manip_score_rrmc = np.mean(manip_scores)
        self.mean_delta_joint_score_rrmc = np.mean(delta_joint_scores)
        self.mean_pos_error_rrmc = np.mean(pose_errors_rrmc[0])
        self.mean_ori_error_rrmc = np.mean(pose_errors_rrmc[1])

    def compute_pose_fitness(self, target_pose):
        """
        Compute the fitness of the robot's configuration by solving the inverse kinematics (IK) problem.

        This function attempts to find a joint configuration that achieves the specified target position
        and orientation using the robot's inverse kinematics solver. It then computes the error between
        the achieved end-effector position/orientation and the target. The error is used as the fitness
        value, with lower values indicating better fitness.

        Parameters:
        target (list or tuple): The desired target position and/or orientation. If the length of the target
                    is 1, it is assumed to be a position. Otherwise, it is assumed to be a 
                    combination of position and orientation.

        Returns:
        float: The computed fitness value. A lower value indicates a better fit. If an error occurs during
               IK computation, a large fitness value (1e6) is returned.
        """
        # Step 4: Compute IK
        joint_config = self.robot.inverse_kinematics(target_pose, pos_tol=0.01, rest_config=self.robot.home_config, max_iter=200, resample=True)
        
        # drive the robot to that config (for measurement)
        self.robot.set_joint_configuration(joint_config)
        ee_pos, ee_ori = self.robot.get_link_state(self.robot.end_effector_index)

        # compute error
        if isinstance(target_pose, tuple) and len(target_pose) == 2:
            # return self.compute_pose_error(target_pose, (ee_pos,ee_ori), weight_position=2.0, weight_orientation=0.25), joint_config
            target_pos, target_quat = target_pose
            _, _, pos_err_norm, _, ori_err_angle = self.robot.check_pose_within_tolerance(
            current_pos=ee_pos,
            current_ori=ee_ori,
            target_pos=target_pos,
            target_ori=target_quat,
            tol=0.0
            )
            # e.g. weight orientation half as much as position
            total_error = pos_err_norm + ori_err_angle
            return total_error, joint_config
        else:
            return np.linalg.norm(np.array(target_pose) - np.array(ee_pos)), joint_config
        
    def compute_rrt_path_cost(self, target_config, home_config=None, collision_objects=[]):
        """
        Plan an RRT path from home_config → q_goal and
        return its joint‐space length (or a big penalty if no path).
        """
        home = home_config or self.robot.home_config

        path = self.motion_planner.rrt_path(home, target_config, collision_objects, rrt_iter=500)
        if path is None:
            return 1e3   # no collision-free path found

        # path cost = sum of successive L2 distances
        cost = 0.0
        for a, b in zip(path[:-1], path[1:]):
            cost += np.linalg.norm(np.array(a) - np.array(b))
        return cost
    
    def compute_motion_plan_fitness(self, pose_waypoints):
        """
        Compute the fitness of a motion plan by solving it in PyBullet.

        This function plans a Cartesian motion path for the robot to follow the given
        pose waypoints. It then computes the fitness of the motion plan based on the
        error between the desired and actual end-effector poses.

        Args:
            pose_waypoints (list of tuple): A list of desired end-effector poses, where each
                pose is represented as a tuple (position, orientation). Position is a tuple
                of (x, y, z) coordinates, and orientation is a tuple of quaternion (x, y, z, w).

        Returns:
            float: The computed fitness value. A lower value indicates a better motion plan.
                If the motion plan cannot be solved, a high fitness value of 1e6 is returned.
        """
        # Compute fitness by solving a motion plan in PyBullet.
        joint_path = self.robot.plan_cartesian_motion_path(pose_waypoints, max_iterations=10000)
        if joint_path is None:
            return 1e6
        else:
            error = 0
            for i, joint_config in enumerate(joint_path):
                self.robot.reset_joint_positions(joint_config)
                ee_pos, ee_ori = self.robot.get_link_state(self.robot.end_effector_index)
                ee_pose = (ee_pos, ee_ori)
                error += self.compute_pose_error(pose_waypoints[i], ee_pose)
            return error
    
    def compute_global_conditioning_index(self, num_samples=100, epsilon=1e-6):
        """
        Computes the Global Conditioning Index (GCI) for the kinematic chain.

        The GCI is calculated as the average of the inverse of the condition number 
        of the Jacobian over a set of sampled joint configurations.

        Args:
            num_samples (int): Number of random joint configurations to sample.
            epsilon (float): Small value to replace near-zero singular values.

        Returns:
            float: The computed GCI value. Higher values indicate better conditioning.
        """
        gci_values = np.zeros(num_samples)

        for i in range(num_samples):
            # Generate a random valid joint configuration within limits
            random_config = np.array([
                np.random.uniform(*self.joint_limits[i]) for i in range(self.num_joints)
            ])

            # Set the robot to this configuration
            self.robot.set_joint_configuration(list(random_config))

            # Compute the Jacobian
            J = np.array(self.robot.get_jacobian(list(random_config)))

            if J.shape[0] != 6:  # Ensure the Jacobian properly accounts for all 6 DOFs
                continue  # Skip if Jacobian computation fails or is incorrect

            # Compute singular values (axes lengths of the manipulability ellipsoid)
            # _, singular_values, _ = np.linalg.svd(J)
            singular_values = svdvals(J)  # More efficient than full SVD

            # Replace near-zero singular values with epsilon to avoid division issues
            singular_values = np.maximum(singular_values, epsilon)

            # Compute the condition number (k = sigma_max / sigma_min) - Could also use np.linalg.cond(J)
            cond_num = np.max(singular_values) / np.min(singular_values)
            # cond_num = np.linalg.cond(J, p=2) # Compute condition number using 2-norm

            # Compute GCI contribution from this sample (square of the inverse of condition number)
            # Note: Not squaring this value is also acceptable. Squaring can simplify algebra, but might not be necessary here
            gci_values[i] = (1.0 / cond_num) ** 2

        return np.mean(gci_values[gci_values > 0]) if np.any(gci_values > 0) else 0.0
    
    def compute_gravity_torque_magnitute(self, joint_positions):
        """
        Compute the magnitude of the gravity torque for a given joint configuration.

        Returns:
            float: The magnitude of the gravity torque.
        """
        # Set the joint positions
        self.robot.set_joint_configuration(joint_positions)

        # Compute the gravity torque
        gravity_torque = self.robot.inverse_dynamics(joint_positions)

        # Compute the magnitude of the gravity torque
        gravity_torque_magnitude = np.linalg.norm(gravity_torque)

        return gravity_torque_magnitude
    
    def compute_resolved_rate_motion_control_fitness(self, target_pose, max_steps=400, alpha=0.75, manipulability_gain=0.1, stall_vel_threshold=0.1, stall_patience=10):
        """
        Compute the fitness of a resolved-rate motion control plan.

        Args:
            target_pose (list): The desired target poses.
            max_steps (int): Maximum number of simulation steps.
            alpha (float): Weight for the manipulability term.
            manipulability_gain (float): Gain for the manipulability term.
            stall_vel_threshold (float): Threshold for stall velocity.
            stall_patience (int): Number of steps to wait before considering a stall.

        Returns:
            tuple: Final joint configuration and fitness metrics.
        """
        target_pos, target_orientation = target_pose

        # TODO: Add smart orientation selection based on target point. Currently only suited for approaches in positive y direction
        # Compute the target pose (position and orientation)
        target_orientations = [
            np.array([180, 0, 90]), # top-down (-z)
            np.array([90, 0, 180]), # front-back (+y)
            np.array([0, 0, -90]), # bottom-up (+z)
            np.array([90, 0, -90]), # right-left (-x)
            np.array([90, 0, 180]), # front-back (+y)
            np.array([90, 0, 90]), # left-right (+x)
        ]

        # Package poses
        target_poses = [(target_pos, R.from_euler('xyz', target_ori, degrees=True).as_quat()) for target_ori in target_orientations]

        # Check if the target pose is reachable via IK
        results = [self.robot.is_pose_reachable(target_pose) for target_pose in target_poses]
        reachabilities, joint_configs = zip(*results)

        # Set the initial joint configuration (in front-back [+y] orientation)
        self.robot.set_joint_configuration(joint_configs[0])

        # Initialize variables to store the final results
        q_final = np.zeros((len(target_poses), self.num_joints))
        manip_score = np.zeros(len(target_poses))
        delta_joint_score = np.zeros(len(target_poses))
        pose_error = np.zeros((len(target_poses), 2))
        for i, reachable in enumerate(reachabilities):
            q_final[i, :], manip_score[i], delta_joint_score[i], pose_error[i, :] = self.motion_planner.resolved_rate_control(
                                                                        target_poses[i], 
                                                                        max_steps=max_steps,
                                                                        plot_manipulability=False, 
                                                                        alpha=alpha, 
                                                                        manipulability_gain=manipulability_gain, 
                                                                        stall_vel_threshold=stall_vel_threshold, 
                                                                        stall_patience=stall_patience)
        return q_final, manip_score, delta_joint_score, pose_error

    @staticmethod        
    def compute_pose_error(target_pose, actual_pose, weight_position=1.0, weight_orientation=1.0):
        """
        Compute a combined error between target and actual poses.
        
        Each pose is a tuple: (position, quaternion)
        - position: a 3-element array
        - quaternion: a 4-element array in [x, y, z, w] format
        
        Args:
            target_pose (tuple): (position, quaternion) for the target.
            actual_pose (tuple): (position, quaternion) for the actual pose.
            weight_position (float): Weight for the position error.
            weight_orientation (float): Weight for the orientation error.
            
        Returns:
            float: The weighted error.
        """
        target_pos, target_quat = target_pose
        actual_pos, actual_quat = actual_pose

        # Compute position error (Euclidean distance)
        pos_error = np.linalg.norm(np.array(target_pos) - np.array(actual_pos))
        
        # Normalize quaternions to be safe
        target_quat = np.array(target_quat) / np.linalg.norm(target_quat)
        actual_quat = np.array(actual_quat) / np.linalg.norm(actual_quat)
        
        # Compute orientation error as angular difference (in radians)
        # Ensure the dot product is positive to get the smallest angle
        dot_prod = np.abs(np.dot(target_quat, actual_quat))
        # Clamp dot_prod to the valid range [-1, 1] to avoid numerical issues
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        ang_error = 2 * np.arccos(dot_prod)

        # NOTE: BELOW METHOD NOT TESTED
        # # Compute quaternion difference more efficiently
        # relative_rotation = R.from_quat(actual_quat) * R.from_quat(target_quat).inv()
        # ang_error = 2 * np.arccos(np.clip(relative_rotation.as_quat()[-1], -1.0, 1.0))
        
        # Combine errors using the specified weights
        total_error = weight_position * pos_error + weight_orientation * ang_error
        return total_error