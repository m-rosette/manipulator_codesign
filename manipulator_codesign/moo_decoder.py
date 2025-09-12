# manipulator_codesign/moo_decoder.py
import numpy as np
from math import pi
from typing import List, Tuple, Optional

# Local mapping used by the optimizer (stable, independent of URDFGen internals)
OPT_CODE_TO_NAME = {
    0: "revolute",
    1: "prismatic",
    2: "spherical"
}
NAME_TO_OPT_CODE = {v: k for k, v in OPT_CODE_TO_NAME.items()}

# Axis mapping for optimizer integer -> URDFGen axis string
AXIS_CODE_TO_STR = {
    0: "1 0 0",
    1: "0 1 0",
    2: "0 0 1"
}


def _default_joint_limits(joint_type: str, axis: Optional[str], length: float) -> Optional[Tuple[float, float]]:
    if joint_type == "revolute":
        return (-pi, pi)
    if joint_type == "prismatic":
        if axis == '0 0 1':
            # z-slider: [0, length]
            return (0.0, max(1e-6, float(length)))
        else:
            half = float(length) / 2.0
            return (-half, half)
    if joint_type == "spherical":
        # URDFGen expands spherical into 3 revolutes; provide revolute-style defaults
        return (-pi, pi)
    return None


def decode_decision_vector(
    x: np.ndarray,
    min_joints: int,
    max_joints: int,
    *,
    link_length_bounds: Tuple[float, float] = (0.05, 0.75),
    include_joint_limits: bool = False,
    pad_strategy: str = "pad"   # "pad" or "error" when too few controllable joints
):
    """
    Decode a single decision vector `x` produced/consumed by NSGA2.

    Returns either (n, types, axes, lengths) or (n, types, axes, lengths, joint_limits)
    where:
      - n is the number of controllable joints (after skipping any defensive 'fixed' codes)
      - types: list of strings compatible with URDFGen e.g. 'revolute','prismatic','spherical'
      - axes: list of axis strings '1 0 0' etc. For spherical entries axis will be '0 0 0' placeholder
      - lengths: list of floats (clipped to link_length_bounds)
    """
    x = np.asarray(x).ravel()
    expected_len = 1 + 3 * max_joints
    if x.size < expected_len:
        raise ValueError(f"decision vector too short: got {x.size}, expected >= {expected_len}")
    if not np.all(np.isfinite(x)):
        raise ValueError("decision vector contains non-finite values")

    # declared number of controllable joint slots (what the optimizer set)
    declared_n = int(np.rint(x[0]))
    declared_n = int(np.clip(declared_n, min_joints, max_joints))

    lo_len, hi_len = link_length_bounds

    joint_types: List[str] = []
    joint_axes: List[str] = []
    link_lengths: List[float] = []
    joint_limits: List[Optional[Tuple[float, float]]] = []

    for i in range(declared_n):
        base = 1 + 3 * i
        jt_code = int(np.rint(x[base + 0]))
        ja_code = int(np.rint(x[base + 1]))
        length_raw = float(x[base + 2])

        # map optimizer code -> canonical joint name
        jt_code = int(np.clip(jt_code, 0, max(OPT_CODE_TO_NAME.keys())))
        jtype = OPT_CODE_TO_NAME[jt_code]

        # defensive: if a fixed-type code were to appear, skip it
        if jtype == "fixed":
            # optimizer shouldn't produce this code, but skip defensively
            continue

        # axis mapping
        if jtype == "spherical":
            # spherical is expanded by URDFGen; the optimizer axis is ignored
            axis_str = "0 0 0"
        else:
            ja_code = int(np.clip(ja_code, 0, 2))
            axis_str = AXIS_CODE_TO_STR[ja_code]

        # length clipping
        length = float(np.clip(length_raw, lo_len, hi_len))

        joint_types.append(jtype)
        joint_axes.append(axis_str)
        link_lengths.append(length)

        if include_joint_limits:
            joint_limits.append(_default_joint_limits(jtype, axis_str, length))

    # If we ended up with fewer than min_joints controllable joints (unlikely with correct x),
    # either pad with safe defaults or raise depending on pad_strategy.
    if len(joint_types) < min_joints:
        if pad_strategy == "error":
            raise RuntimeError(f"Decoded only {len(joint_types)} controllable joints, fewer than min_joints={min_joints}")
        # pad with safe prismatic x-axis defaults
        while len(joint_types) < min_joints:
            joint_types.append("prismatic")
            joint_axes.append("1 0 0")
            link_lengths.append(lo_len)
            if include_joint_limits:
                joint_limits.append(_default_joint_limits("prismatic", "1 0 0", lo_len))

    if include_joint_limits:
        return len(joint_types), joint_types, joint_axes, link_lengths, joint_limits
    return len(joint_types), joint_types, joint_axes, link_lengths


def decode_population(
    X: np.ndarray,
    min_joints: int,
    max_joints: int,
    *,
    link_length_bounds: Tuple[float, float] = (0.05, 0.75),
    include_joint_limits: bool = False,
    pad_strategy: str = "pad"
):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    out = []
    for i in range(X.shape[0]):
        parsed = decode_decision_vector(
            X[i],
            min_joints,
            max_joints,
            link_length_bounds=link_length_bounds,
            include_joint_limits=include_joint_limits,
            pad_strategy=pad_strategy
        )
        out.append(parsed)
    return out
