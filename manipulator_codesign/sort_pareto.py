import pickle
import pandas as pd
import numpy as np
import pybullet as p
from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet

# 1) Load your results
with open('/home/marcus/IMML/data/nsga2_results/results_20250705_180220.pkl', 'rb') as f:
    data = pickle.load(f)
X = data['X']         # shape (n_solutions, n_vars)
F = data['F']         # shape (n_solutions, n_objs)

# 2) Build DataFrame
obj_names = ['pose_error','torque','joint_count',
             'conditioning_index','rrmc_score','rrt_path_cost']
df = pd.DataFrame(F, columns=obj_names)

# 3) Sort by the three you care about
df_sorted = df.sort_values(
    by=['joint_count', 'conditioning_index'],
    ascending=[False, True]   # descending on joint_count
)

# 4) Pick top
sample_num = 25
top = df_sorted.head(sample_num)
top_X = X[top.index, :]

print("Top (min pose_error, min cond_idx, max joint_count):")
print(top)

# # 5) Save an indexed decision vector as a urdf
# decision_vec_idx = 0
# num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(top_X[decision_vec_idx], 4, 7)
# chain = KinematicChainPyBullet(p, [0, 0, 0], num_joints, joint_types, joint_axes, link_lengths, collision_objects=[])
# chain.build_robot()
# chain.save_urdf('robot')