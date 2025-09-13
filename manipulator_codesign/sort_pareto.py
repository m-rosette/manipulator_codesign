import pickle
import pandas as pd
import numpy as np
import pybullet as p
import argparse
from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.urdf_gen import URDFGen

def main():
    parser = argparse.ArgumentParser(description="Sort and analyze Pareto front results.")
    parser.add_argument('--data', type=str, required=True, help='Path to the results .pkl file')
    parser.add_argument('--decision_vec_idx', type=int, default=None, help='Index of the decision vector to save as URDF')
    parser.add_argument('--print_full', action='store_true', help='Print the full DataFrame instead of truncating')
    args = parser.parse_args()

    # 1) Load results
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    X = data['X']         # shape (n_solutions, n_vars)
    F = data['F']         # shape (n_solutions, n_objs)

    # 2) Build DataFrame of objectives
    obj_names = ['pose_error','torque','joint_count',
                 'conditioning_index','rrmc_score','rrt_path_cost']
    df = pd.DataFrame(F, columns=obj_names)

    # 3) Decode decision vectors and apply mappings for readability
    decoded_info = [decode_decision_vector(vec, 5, 7) for vec in X]
    # Each entry is (num_joints, joint_types, joint_axes, link_lengths)
    df['num_joints'] = [d[0] for d in decoded_info]

    # Map joint types (ints -> readable strings)
    df['joint_types'] = [
        [URDFGen.map_joint_type_inverse(jt) for jt in d[1]] for d in decoded_info
    ]

    # Map axes (ints -> "x y z" strings)
    df['joint_axes'] = [
        [URDFGen.map_axis_inverse(ax) for ax in d[2]] for d in decoded_info
    ]

    # Link lengths can stay numeric - round to 3 decimals
    df['link_lengths'] = [
        [round(l, 3) for l in d[3]] for d in decoded_info
    ]

    # 4) Sort by objectives (example: pose_error ascending)
    df_sorted = df.sort_values(
        by=['pose_error'],
        ascending=[True]
    )

    # 5) Print results
    if args.print_full:
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 200,
                               'display.colheader_justify', 'left'):
            print(df_sorted)
    else:
        with pd.option_context(
            'display.colheader_justify', 'left'
        ):
            print(df_sorted)

    # 8) Save an indexed decision vector as a URDF if requested
    if args.decision_vec_idx is not None:
        num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(X[args.decision_vec_idx], 5, 7)
        chain = KinematicChainPyBullet(
            p, [0, 0, 0], num_joints, joint_types, joint_axes, link_lengths, collision_objects=[]
        )
        chain.build_robot()
        chain.save_urdf(f'robot_{args.decision_vec_idx}')


if __name__ == "__main__":
    main()