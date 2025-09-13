import os
import argparse
import numpy as np
from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_objects import LoadObjects
import manipulator_codesign.orchard_workspace as orchard_ws
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.urdf_to_decision_vector import urdf_to_decision_vector, encode_seed
from manipulator_codesign.moo_decoder import decode_decision_vector


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


class BenchMarkRobot:
    def __init__(self, robot_urdf_path: str, robot_home_pos=None, ik_tol=0.01, max_ik_iter=200, renders=True, ee_link_name='end_effector', ur5e=False, pretty_mesh=False):
        """
        Initialize the ViewRobot class.

        Args:
            robot_urdf_path (str): Path to the URDF file of the robot.
            robot_home_pos (list): Home position of the robot joints.
            ik_tol (float, optional): Tolerance for inverse kinematics. Defaults to 0.01.
            renders (bool, optional): Whether to visualize the robot in the PyBullet GUI. Defaults to True.
        """
        self.pyb = PybUtils(renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)
        
        self.robot_urdf_path = robot_urdf_path
        self.robot_home_pos = robot_home_pos
        self.ik_tol = ik_tol
        self.max_ik_iter = max_ik_iter
        self.ee_link_name = ee_link_name
        self.ur5e = ur5e
        self.pretty_mesh = pretty_mesh

        self.robot_to_amiga_translation = [0, 0, 1.025] 
        self.amiga_to_robot_translation = [0, -0.3, 0]
        self.robot_system_translation = [-1.5, -2.25, 0]
        self.robot_translation = np.add(self.robot_to_amiga_translation, self.robot_system_translation)

        self.stage_env()

    def stage_env(self):      
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_dir = os.path.join(script_dir, 'urdf')
        flags = 0 
        self.amiga_id = self.object_loader.load_urdf(os.path.join(urdf_dir, "robots/amiga.urdf"),
                                        start_pos=np.add(self.amiga_to_robot_translation, self.robot_system_translation), 
                                        start_orientation=[0, 0, 0], 
                                        fix_base=True,
                                        flags=flags)
        self.object_loader.collision_objects.append(self.amiga_id)

        if self.pretty_mesh:
            filename = "/manipulator_codesign/manipulator_codesign/meshes/Pair01_before_mesh.obj"
            # Load tree collision shape
            collision_shape = self.pyb.con.createCollisionShape(
                shapeType=self.pyb.con.GEOM_MESH,
                fileName=filename,
                flags=self.pyb.con.GEOM_FORCE_CONCAVE_TRIMESH,
            )
            # Create a visual shape with light brown color (RGBA)
            visual_shape = self.pyb.con.createVisualShape(
                shapeType=self.pyb.con.GEOM_MESH,
                fileName=filename,
                rgbaColor=[0.78, 0.65, 0.53, 1.0]  # light brown, alpha=1.0
            )
            # Create your body using this multi‐hull collision shape
            body_id = self.pyb.con.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[-5.7, -1.905, -2]
            )
            self.object_loader.collision_objects.append(body_id)
        elif self.pretty_mesh == False:
            filename = "/manipulator_codesign/manipulator_codesign/meshes/before_mesh_transformed.obj"
            # Load tree collision shape
            collision_shape = self.pyb.con.createCollisionShape(
                shapeType=self.pyb.con.GEOM_MESH,
                fileName=filename,
                flags=self.pyb.con.GEOM_FORCE_CONCAVE_TRIMESH,
            )
            # Create a visual shape with light brown color (RGBA)
            visual_shape = self.pyb.con.createVisualShape(
                shapeType=self.pyb.con.GEOM_MESH,
                fileName=filename,
                rgbaColor=[0.78, 0.65, 0.53, 1.0]  # light brown, alpha=1.0
            )
            # Create your body using this multi‐hull collision shape
            body_id = self.pyb.con.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0,0,0],

            )
            self.object_loader.collision_objects.append(body_id)
        else:
            pass

    def load_prune_poses(self, window_size=2.5):
        yaml_path = "manipulator_codesign/prune_data/all_branches_info.yaml"

        # Load the prune points from the YAML file
        pose_data_results = orchard_ws.get_prune_poses_from_yaml(
            yaml_path=yaml_path,
            robot_base=self.robot_translation,
            window_size=window_size,
            min_y=None,
            max_y=0,
            downsample=False,
            downsample_threshold=0.1
        )
        target_poses, target_offset_poses = orchard_ws.package_poses(pose_data_results)
        prune_points = [res['prune_point'] for res in pose_data_results]

        return prune_points, target_poses, target_offset_poses

    def load_optimized_manip(self, target_poses, target_offset_poses):
        """
        Load an optimized manipulator from a URDF file.

        Args:
            urdf_path (str): Path to the URDF file.
            robot_home_pos (list, optional): Home position of the robot joints. Defaults to None.
            ee_link_name (str, optional): Name of the end-effector link. Defaults to 'ee_link'.
        """
        # decode and build kinematic chain
        raw_robot_params = urdf_to_decision_vector(self.robot_urdf_path, ee_link_name=self.ee_link_name)
        x = encode_seed(raw_robot_params, min_joints=5, max_joints=7)
        joint_count, joint_types, joint_axes, link_lengths = decode_decision_vector(x, min_joints=5, max_joints=7)

        ch = KinematicChainPyBullet(
            self.pyb.con, self.robot_translation,
            joint_count, joint_types, joint_axes, link_lengths,
            collision_objects=self.object_loader.collision_objects,
            ee_link_name=self.ee_link_name, 
            ik_tol=self.ik_tol, max_ik_iter=self.max_ik_iter
        )
        if not ch.is_built and not self.ur5e:
            ch.build_robot()
        elif self.ur5e:
            ch.urdf_path = self.robot_urdf_path
            ch.ee_link_name = self.ee_link_name
        ch.load_robot()

        target_poses = ch.sample_collision_free_poses(target_poses)
        target_offset_poses = ch.sample_collision_free_poses(target_offset_poses)

        ch.compute_chain_metrics(target_poses, target_offset_poses)
        ch.compute_chain_metric_stats()

        return {
            'pose_error':                 ch.mean_pose_error,
            'pose_error_std':            ch.std_pose_error,

            'rrt_path_cost':             ch.mean_rrt_path_cost,
            'rrt_path_cost_std':         ch.std_rrt_path_cost,

            'torque':                    ch.mean_torque,
            'torque_std':                ch.std_torque,

            'joint_count':               ch.num_logical_joints,
            'conditioning_index':        ch.global_conditioning_index,

            'delta_joint_score_rrmc':    ch.mean_delta_joint_score_rrmc,
            'delta_joint_score_rrmc_std': ch.std_delta_joint_score_rrmc,

            'pos_error_rrmc':            ch.mean_pos_error_rrmc,
            'pos_error_rrmc_std':        ch.std_pos_error_rrmc,

            'num_reachable_targets':      ch.num_reachable_targets,
            'reachable_fraction':         ch.reachable_fraction,
        }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a robot in PyBullet simulation.")
    parser.add_argument("-u", "--urdf_path", type=str, default="example_6dof_manipulator.urdf", 
                        help="URDF file path or name (default: example_6dof_manipulator.urdf)")
    parser.add_argument("--render", action="store_true", help="Enable rendering in PyBullet")
    parser.add_argument("--no-render", action="store_false", dest="render", help="Disable rendering in PyBullet")
    parser.add_argument("--ur5e", action="store_true", help="Run benchmark on ur5e")
    
    parser.set_defaults(render=True)  # Default to True
    parser.set_defaults(ur5e=False)  # Default to False
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf_dir = os.path.join(script_dir, 'urdf', 'robots')
    default_urdf_file = os.path.join(default_urdf_dir, "example_6dof_manipulator.urdf")
    
    robot_urdf_path = get_urdf_path(args.urdf_path, default_urdf_dir, default_urdf_file)

    ur5e = args.ur5e
    ee_link_name = 'end_effector'
    
    robot_home_pos = None

    print(f"\nLoading robot from: {robot_urdf_path}\n")
    
    bench_mark = BenchMarkRobot(robot_urdf_path=robot_urdf_path, 
                           renders=args.render,
                           robot_home_pos=robot_home_pos,
                           ik_tol=0.05,
                           max_ik_iter=250,
                           ee_link_name=ee_link_name,
                           ur5e=ur5e)
    prune_points, target_poses, target_offset_poses = bench_mark.load_prune_poses(window_size=2.5)
    
    results = bench_mark.load_optimized_manip(
                # target_poses=target_poses[len(target_poses) // 6 * 5 :], # use the last quarter of locations
                # target_offset_poses=target_offset_poses[len(target_poses) // 6 * 5 :], # use the last quarter of locations
                target_poses=target_poses,
                target_offset_poses=target_offset_poses
                )
    
    print("\nBenchmark Results:")
    # print(f"Total target poses evaluated: {len(target_poses[len(target_poses) // 6 * 5 :])}")
    print(f"Total target poses evaluated: {len(target_poses)}")
    print("------------------")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")  
    print("------------------\n")  