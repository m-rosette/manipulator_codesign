import numpy as np
import os
import xml.etree.ElementTree as ET

from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_robot import LoadRobot
from manipulator_codesign.urdf_gen import URDFGen


def parse_cylinder_lengths(urdf_path):
    """
    Returns a dict mapping link-name → true cylinder length (float)
    """
    tree = ET.parse(urdf_path)
    cyls = {}
    for link in tree.getroot().findall('link'):
        cyl = link.find('visual/geometry/cylinder')
        if cyl is not None:
            cyls[link.attrib['name']] = float(cyl.attrib['length'])
    return cyls

def get_link_length(robot, joint_idx, cylinder_lengths):
    """
    Returns the true cylinder length for the child link of `joint_idx`, 
    or 2x the distance between joint and child link if not present.
    """
    joint_pos = robot.con.getJointState(robot.robotId, joint_idx)[0]
    child_pos = robot.con.getJointState(robot.robotId, joint_idx if robot.con.getJointInfo(robot.robotId, joint_idx)[2] == robot.con.JOINT_FIXED else joint_idx)[0]
    child_name = robot.con.getJointInfo(robot.robotId, joint_idx)[12].decode()

    if child_name in cylinder_lengths:
        return cylinder_lengths[child_name]
    else:
        return 2.0 * np.linalg.norm(np.array(child_pos) - np.array(joint_pos))

def get_logical_joints(robot):
    """
    Returns a list of either
        - int          (rev/prism joints)
        - [ix,iy,iz]  (spherical triplets)
    in the order of controllable joints.
    """
    seen = set()
    logical = []
    for j in robot.controllable_joint_idx:
        if j in seen:
            continue

        if j in robot.spherical_groups:
            trip = next(t for t in robot.spherical_joint_idx if t[0] == j)
            logical.append(trip)
            seen.update(trip)
        else:
            logical.append(j)
            seen.add(j)
    return logical

def urdf_to_decision_vector(urdf_path):
    # 1) load robot and let it detect spherical groups
    pyb = PybUtils(renders=False)
    robot = LoadRobot(
        pyb.con,
        urdf_path,
        [0, 0, 0],
        pyb.con.getQuaternionFromEuler([0, 0, 0]),
        home_config=None,
        collision_objects=[],
        ee_link_name='end_effector',
    )

    # 2) parse URDF once for true cylinder lengths
    cylinder_lengths = parse_cylinder_lengths(urdf_path)

    # 3) build logical sequence in URDF/joint order
    logical_joints = get_logical_joints(robot)

    # 3) extract properties for each logical joint
    types, axes, lengths = [], [], []
    for item in logical_joints:
        if isinstance(item, list):
            # spherical
            types.append('spherical')
            # collect each axis vector
            axes.append([tuple(pyb.con.getJointInfo(robot.robotId, j)[13]) for j in item])
            # length is on the fixed joint after the triplet
            fixed = robot.spherical_groups[item[0]]
            lengths.append(get_link_length(robot, fixed, cylinder_lengths))

        else:
            # revolute or prismatic
            info = pyb.con.getJointInfo(robot.robotId, item)
            types.append('revolute' if info[2] == pyb.con.JOINT_REVOLUTE else 'prismatic')
            axes.append(tuple(info[13]))
            lengths.append(get_link_length(robot, item, cylinder_lengths))

    # Replace any list of tuples, spherical logical joint, with (0, 0, 0)
    axes = [(0.0, 0.0, 0.0) if isinstance(item, list) and all(isinstance(subitem, tuple) for subitem in item) else item
                for item in axes]

    return len(types), types, axes, lengths

def encode_seed(dec_vec, min_joints=4, max_joints=7):
    """
    Turn the output of urdf_to_decision_vector:
       (n_joints, types, axes, lengths)
    into a 1D numpy array [n_joints, type0, axis0, len0, …],
    padded out to max_joints with a safe “fallback” design.
    """
    n_joints, types, axes, lengths = dec_vec

    vec = []
    # 1) number of joints
    vec.append(float(n_joints))

    # 2) encode each real joint
    for t, a, L in zip(types, axes, lengths):
        # joint-type code via centralized inverse
        t_id = URDFGen.map_joint_type_inverse(t)

        # spherical has its own code
        if t == 'spherical':
            a_id = 3
        else:
            # turn e.g. (1,0,0) into "1 0 0"
            axis_str = f"{int(round(a[0]))} {int(round(a[1]))} {int(round(a[2]))}"
            a_id = URDFGen.map_axis_inverse(axis_str)

        vec.extend([float(t_id), float(a_id), float(L)])

    # 3) pad out inactive slots (ensure we have 1 + 3 * max_joints in decision vector)
    pad_t = URDFGen.map_joint_type_inverse('prismatic')
    pad_axis_str = URDFGen.map_axis_input('x')  # default to x-axis
    pad_a = URDFGen.map_axis_inverse(pad_axis_str)

    while len(vec) < 1 + 3 * max_joints:
        vec.extend([
            float(pad_t),  # prismatic
            float(pad_a),  # x-axis
            0.1            # minimal length
        ])

    return np.asarray(vec, dtype=float)

def load_seeds(urdf_dir, max_joints=7):
    """
    Load and encode all URDF seed files in `urdf_dir`.
    """
    urdfs = [os.path.join(urdf_dir, f) for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
    raw = [urdf_to_decision_vector(u) for u in urdfs]
    return [encode_seed(s, max_joints=max_joints) for s in raw]


if __name__ == "__main__":
    urdf_path = "manipulator_codesign/urdf/robots/nsga2_seeds/gen_seed_9.urdf"
    total_joint_count, joint_types, joint_axes, link_lengths = urdf_to_decision_vector(
        urdf_path
    )
    # Replace any list of tuples with (0, 0, 0)
    joint_axes = [(0.0, 0.0, 0.0) if isinstance(item, list) and all(isinstance(subitem, tuple) for subitem in item) else item
                for item in joint_axes]
    print("Total joints in the system:", total_joint_count)
    joint_types = [URDFGen.map_joint_type_inverse(jt) for jt in joint_types]
    print("Joint types:", joint_types)
    joint_axes = [' '.join(map(str, map(int, ja))) for ja in joint_axes]
    joint_axes = [URDFGen.map_axis_inverse(ja) for ja in joint_axes]
    print("Joint axes:", joint_axes)
    print("Link lengths:", link_lengths)
