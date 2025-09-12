"""
manipulator_codesign/urdf_to_decision_vector.py

Convert URDF --> decision vector used by the NSGA2 optimizer (and helper functions).
This module intentionally encodes only controllable joints (revolute/prismatic/spherical).
Fixed helper joints produced by URDFGen are not represented in the optimizer decision vector.
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple

# These imports assume your pybullet_robokit helpers are on PYTHONPATH
from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_robot import LoadRobot

from manipulator_codesign.urdf_gen import URDFGen

# --------------------------
# Optimizer <-> semantic maps
# --------------------------
# Stable mapping used by NSGA2 decision vectors:
#  0 -> revolute
#  1 -> prismatic
#  2 -> spherical
OPT_CODE_TO_NAME = {0: "revolute", 1: "prismatic", 2: "spherical"}
NAME_TO_OPT_CODE = {v: k for k, v in OPT_CODE_TO_NAME.items()}

# Axis code mapping (optimizer integer -> URDF axis string)
AXIS_CODE_TO_STR = {0: "1 0 0", 1: "0 1 0", 2: "0 0 1"}
STR_TO_AXIS_CODE = {v: k for k, v in AXIS_CODE_TO_STR.items()}


# --------------------------
# URDF helpers
# --------------------------

def parse_cylinder_lengths(urdf_path: str) -> dict:
    """
    Parse URDF and return {link_name: cylinder_length} for every <visual><geometry><cylinder length="..."> found.
    """
    cyls = {}
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception:
        return cyls

    for link in root.findall('link'):
        name = link.attrib.get('name')
        if not name:
            continue
        visual = link.find('visual')
        if visual is None:
            continue
        geom = visual.find('geometry')
        if geom is None:
            continue
        cyl = geom.find('cylinder')
        if cyl is not None and 'length' in cyl.attrib:
            try:
                cyls[name] = float(cyl.attrib['length'])
            except Exception:
                # skip malformed
                pass
    return cyls


def _safe_get_joint_info(con, robot_id: int, joint_idx: int):
    """
    Wrapper for pybullet getJointInfo; returns None on error.
    """
    try:
        return con.getJointInfo(robot_id, joint_idx)
    except Exception:
        return None


def get_link_length(robot, joint_idx: int, cylinder_lengths: dict, fallback: float = 0.25) -> float:
    """
    Estimate the length of the child link of joint_idx.

    Priority:
      1) If joint is prismatic and limits exist (upper != lower), return span upper - lower.
      2) If the child link name has a cylinder length parsed from URDF, use that.
      3) Otherwise return fallback.
    """
    con = robot.con
    info = _safe_get_joint_info(con, robot.robotId, joint_idx)
    if info is None:
        return float(fallback)

    # pybullet getJointInfo fields: see pybullet docs; indices used below are conventional:
    # info[2] = jointType, info[8]=lower limit, info[9]=upper limit, info[12]=childLinkName, info[13]=axis
    joint_type = info[2]
    try:
        child_name = info[12].decode() if isinstance(info[12], (bytes, bytearray)) else str(info[12])
    except Exception:
        child_name = None

    # If prismatic and limits meaningful, use span
    if joint_type == con.JOINT_PRISMATIC:
        try:
            lower = float(info[8])
            upper = float(info[9])
            # span = upper - lower is the correct travel
            if not np.isclose(upper, lower):
                span = upper - lower
                if span > 0.0:
                    return float(span)
        except Exception:
            pass

    # If cylinder length was specified in URDF visuals, prefer that
    if child_name and child_name in cylinder_lengths:
        try:
            return float(cylinder_lengths[child_name])
        except Exception:
            pass

    # fallback
    return float(fallback)


def get_logical_joints(robot) -> List:
    """
    Return a list of logical joints in order of 'controllable' actuators.

    Each returned element is either:
      - int: a single controllable joint index (revolute/prismatic)
      - list/tuple of three ints: spherical joint represented as a triplet of revolute joints

    Expects LoadRobot to provide:
      - robot.controllable_joint_idx : iterable of controllable joint indices (in order)
      - robot.spherical_joint_idx : list of triplets (optional)
      - robot.spherical_groups : mapping first_triplet_index -> following_fixed_joint_index (optional)
    """
    seen = set()
    logical = []

    spherical_triplets = getattr(robot, 'spherical_joint_idx', []) or []
    sph_first_map = {t[0]: t for t in spherical_triplets}

    for j in getattr(robot, 'controllable_joint_idx', []):
        if j in seen:
            continue
        if j in sph_first_map:
            trip = list(sph_first_map[j])
            logical.append(trip)
            seen.update(trip)
        else:
            logical.append(j)
            seen.add(j)
    return logical


# --------------------------
# Encoder: URDF -> decision vector
# --------------------------

def _name_to_opt_code(name: str) -> int:
    """
    Map URDF joint type name -> optimizer code (0/1/2).
    Falls back to 'revolute' for unknown names.
    """
    if not isinstance(name, str):
        return NAME_TO_OPT_CODE['revolute']
    n = name.lower()
    if 'revol' in n:
        return NAME_TO_OPT_CODE['revolute']
    if 'pris' in n:
        return NAME_TO_OPT_CODE['prismatic']
    if 'sph' in n:
        return NAME_TO_OPT_CODE['spherical']
    return NAME_TO_OPT_CODE['revolute']


def encode_seed(dec_vec: Tuple[int, List[str], List[Tuple[float, float, float]], List[float]],
                min_joints: int = 4,
                max_joints: int = 7,
                link_length_bounds: Tuple[float, float] = (0.05, 0.75)) -> np.ndarray:
    """
    Turn (n_joints, types, axes, lengths) into the optimizer decision vector:
      [n_joints, type0, axis0, len0, type1, axis1, len1, ...] padded to 1 + 3 * max_joints

    dec_vec:
      - n_joints: int
      - types: list of strings ("revolute","prismatic","spherical")
      - axes: list of tuples like (1,0,0) or (0,0,0) for spherical placeholders
      - lengths: list of floats
    """
    n_joints, types, axes, lengths = dec_vec
    lo_len, hi_len = link_length_bounds

    # Clip declared n_joints to allowed range
    n_joints = int(np.clip(int(n_joints), min_joints, max_joints))

    vec = []
    vec.append(float(n_joints))

    for t, a, L in zip(types, axes, lengths):
        t_code = _name_to_opt_code(t)

        # spherical placeholder: axis is ignored by decoder; use 0
        if t == 'spherical':
            a_code = 0
        else:
            # axis may be a tuple of floats. Reduce to integer axis codes.
            try:
                a_tuple = tuple(int(round(float(v))) for v in a)
                a_code = STR_TO_AXIS_CODE.get(f"{a_tuple[0]} {a_tuple[1]} {a_tuple[2]}", 0)
            except Exception:
                a_code = 0

        L_clipped = float(np.clip(float(L), lo_len, hi_len))
        vec.extend([float(t_code), float(a_code), float(L_clipped)])

    # pad with safe defaults (prismatic, x-axis, minimal length)
    pad_type = NAME_TO_OPT_CODE['prismatic']
    pad_axis = 0
    pad_len = float(lo_len)
    while len(vec) < 1 + 3 * max_joints:
        vec.extend([float(pad_type), float(pad_axis), float(pad_len)])

    return np.asarray(vec, dtype=float)


# --------------------------
# URDF -> decision vector main export
# --------------------------

def urdf_to_decision_vector(urdf_path: str, fallback_length: float = 0.25) -> Tuple[int, List[str], List[Tuple[float, float, float]], List[float]]:
    """
    Load URDF into pybullet via LoadRobot, detect controllable joints (including spherical triplets),
    and return (n_controllable, types, axes, lengths).

    types are strings ('revolute','prismatic','spherical')
    axes are tuples (x,y,z) or (0.0,0.0,0.0) for spherical
    lengths are floats estimated as described in get_link_length
    """
    pyb = PybUtils(renders=False)
    con = pyb.con

    # Load robot through LoadRobot. This will let it populate spherical groups and controllable joints.
    robot = LoadRobot(
        con,
        urdf_path,
        [0, 0, 0],
        con.getQuaternionFromEuler([0, 0, 0]),
        home_config=None,
        collision_objects=[],
        ee_link_name='end_effector',
    )

    # parse cylinder lengths once
    cylinder_lengths = parse_cylinder_lengths(urdf_path)

    logical_joints = get_logical_joints(robot)

    types = []
    axes = []
    lengths = []

    spherical_groups = getattr(robot, 'spherical_groups', {}) or {}

    def ji(idx):
        return _safe_get_joint_info(con, robot.robotId, idx)

    for item in logical_joints:
        if isinstance(item, (list, tuple)):
            # spherical logical joint (triplet)
            types.append('spherical')
            axes.append((0.0, 0.0, 0.0))

            # try to locate the fixed joint after the triplet per URDFGen convention
            fixed_idx = None
            try:
                fixed_idx = spherical_groups.get(item[0], None)
            except Exception:
                fixed_idx = None

            if fixed_idx is None:
                # fallback heuristic: look for the next fixed joint index in controllable_joint_idx
                max_trip = max(item)
                for j in getattr(robot, 'controllable_joint_idx', []):
                    if j <= max_trip:
                        continue
                    info = ji(j)
                    if info is None:
                        continue
                    if info[2] == con.JOINT_FIXED:
                        fixed_idx = j
                        break

            if fixed_idx is None:
                fixed_idx = item[0]

            lengths.append(get_link_length(robot, fixed_idx, cylinder_lengths, fallback=fallback_length))

        else:
            # single controllable joint
            info = ji(item)
            if info is None:
                types.append('revolute')
                axes.append((0.0, 0.0, 1.0))
                lengths.append(fallback_length)
                continue

            jtype_code = info[2]
            if jtype_code == con.JOINT_REVOLUTE:
                jtype = 'revolute'
            elif jtype_code == con.JOINT_PRISMATIC:
                jtype = 'prismatic'
            elif jtype_code == con.JOINT_FIXED:
                # A fixed controllable joint is odd. Treat as 'fixed' and skip it
                # Here we choose to skip fixed joints entirely, consistent with optimizer.
                # But we must not modify `logical_joints` here; instead perform a skip by continuing.
                # Note: real URDFs produced by URDFGen should not put fixed joints in controllable_joint_idx.
                continue
            else:
                jtype = 'revolute'

            # axis is in info[13]
            raw_axis = info[13]
            try:
                axis_tuple = tuple(float(v) for v in raw_axis)
                # normalize to integer-like axes when appropriate
                axis_tuple = tuple(0.0 if abs(v) < 1e-6 else (1.0 if abs(v - 1.0) < 1e-6 else float(v)) for v in axis_tuple)
            except Exception:
                axis_tuple = (0.0, 0.0, 1.0)

            types.append(jtype)
            axes.append(axis_tuple)
            lengths.append(get_link_length(robot, item, cylinder_lengths, fallback=fallback_length))

    # cleanup pybullet
    try:
        con.removeBody(robot.robotId)
    except Exception:
        pass
    pyb.disconnect()

    return len(types), types, axes, lengths


def load_seeds(urdf_dir: str, max_joints: int = 7, min_joints: int = 4) -> List[np.ndarray]:
    """
    Iterate *.urdf files in urdf_dir and return a list of decision vectors (numpy arrays)
    encoded with encode_seed.
    """
    urdfs = sorted([os.path.join(urdf_dir, f) for f in os.listdir(urdf_dir) if f.endswith('.urdf')])
    seeds = []
    for u in urdfs:
        try:
            dec = urdf_to_decision_vector(u)
            encoded = encode_seed(dec, min_joints=min_joints, max_joints=max_joints)
            seeds.append(encoded)
        except Exception:
            # intentionally skip problematic URDFs
            continue
    return seeds


# --------------------------
# Quick smoke test
# --------------------------
if __name__ == "__main__":
    test_urdf = "manipulator_codesign/urdf/robots/nsga2_seeds/gen_seed_9.urdf"
    if os.path.exists(test_urdf):
        dec = urdf_to_decision_vector(test_urdf)
        print("decoded:", dec)
        encoded = encode_seed(dec, max_joints=7)
        print("encoded vector length:", encoded.shape)
        print("encoded head:", encoded[:12])
    else:
        print("No sample URDF at", test_urdf)
