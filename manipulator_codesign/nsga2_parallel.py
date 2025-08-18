import os
import pickle
from datetime import datetime
import argparse
import numpy as np
import ray
import wandb

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.core.callback import Callback 

from manipulator_codesign.nsga2_operators import SeededSampling, MixedSampling, MixedCrossover, MixedMutation
from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.urdf_to_decision_vector import load_seeds
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
import manipulator_codesign.orchard_workspace as orchard_ws
from pybullet_robokit.load_objects import LoadObjects


# -------- Ray Actor for Persistent Evaluation --------
@ray.remote
class Evaluator:
    def __init__(self,
                 mesh_path: str,
                 robot_urdf: str,
                 flags: int):
        import pybullet as p, pybullet_data
        self.p = p
        self.p.connect(p.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)

        self.robot_urdf = robot_urdf
        self.flags = flags

        # single‐time mesh load
        self.tree_collision_shape = self.p.createCollisionShape(
            shapeType=self.p.GEOM_MESH,
            fileName=mesh_path,
            flags=self.p.GEOM_FORCE_CONCAVE_TRIMESH
        )

    def evaluate(self, x, 
                 targets_list, targets_offset_list,
                 robot_translations_list, mobile_base_translations_list,
                 min_j, max_j,
                 alpha, beta, delta, gamma):
        p = self.p    
        p.setGravity(0, 0, -9.81)   
        self.object_loader = LoadObjects(p)

        n_positions = len(robot_translations_list)
        assert n_positions == len(mobile_base_translations_list) == len(targets_list) == len(targets_offset_list), \
            "All lists of per-base inputs must have the same length"
        
        # load robot at a default position
        # (this will be overridden in the loop below)
        self.amiga_id = self.object_loader.load_urdf(
            self.robot_urdf,
            start_pos=[0, 0, 0],
            start_orientation=[0,0,0],
            fix_base=True,
            flags=self.flags
        )

        # load tree
        self.tree_id = p.createMultiBody(
            baseCollisionShapeIndex=self.tree_collision_shape,
            baseVisualShapeIndex=-1,
            basePosition=[0,0,0]
        )
        self.object_loader.collision_objects.extend([self.amiga_id, self.tree_id])

        # decode kinematic chain
        n, types, axes, lengths = decode_decision_vector(x, min_j, max_j)

        ch = KinematicChainPyBullet(
            p, [0, 0, 0],
            n, types, axes, lengths,
            collision_objects=self.object_loader.collision_objects
        )

        if not ch.is_built:
            ch.build_robot()
        ch.load_robot()

        for i in range(n_positions):
            self.p.resetBasePositionAndOrientation(
                self.amiga_id,
                mobile_base_translations_list[i],
                self.p.getQuaternionFromEuler([0, 0, 0])
            )
            self.p.resetBasePositionAndOrientation(
                ch.robot.robotId,
                robot_translations_list[i],
                self.p.getQuaternionFromEuler([0, 0, 0])
            )
            self.p.resetBasePositionAndOrientation(
                self.tree_id,
                [0, 0, 0],
                self.p.getQuaternionFromEuler([0, 0, 0])
            )
            
            targets = ch.sample_collision_free_poses(targets_list[i])
            targets_offset = ch.sample_collision_free_poses(targets_offset_list[i])

            ch.compute_chain_metrics(targets, targets_offset)

        ch.compute_chain_metric_stats()

        # cleanup
        p.resetSimulation()

        return {
            'pose_error':             ch.mean_pose_error,
            'rrt_path_cost':          ch.mean_rrt_path_cost,
            'torque':                 ch.mean_torque,
            'joint_count':            ch.num_joints,
            'conditioning_index':     ch.global_conditioning_index,
            'delta_joint_score_rrmc': ch.mean_delta_joint_score_rrmc,
            'pos_error_rrmc':         ch.mean_pos_error_rrmc
        }


# -------- Problem Definition --------
class KinematicChainProblem(Problem):
    def __init__(self, targets, targets_offset, robot_translation, mobile_base_translation,
                 seeds, dec_vec_bounds, min_joints=2, max_joints=7,
                 alpha=1, beta=1, delta=1, gamma=1,
                 cal_samples=15, num_actors=4, num_objectives=6):
        print("[KinematicChainProblem] Initializing and calibrating...")
        self.targets = targets
        self.targets_offset = targets_offset
        self.robot_translation = np.asarray(robot_translation, dtype=float)
        self.mobile_base_translation = np.asarray(mobile_base_translation, dtype=float)
        self.min_joints, self.max_joints = min_joints, max_joints
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma

        xl = dec_vec_bounds[0]
        xu = dec_vec_bounds[1]

        super().__init__(n_var=len(xl), n_obj=num_objectives, xl=np.array(xl), xu=np.array(xu))

        self._x_cal = self._make_calibration_batch(seeds, cal_samples)

        # create a pool of Evaluator actors
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mesh_path = os.path.join(script_dir, "meshes", "before_mesh_transformed.obj")
        robot_urdf = os.path.join(script_dir, "urdf", "robots", "amiga.urdf")
        flags = 0
        self.actors = [
            Evaluator.options(max_concurrency=1).remote(
                mesh_path,
                robot_urdf,
                flags
            )
            for _ in range(num_actors)
        ]

        self._parallel_calibration()

    def _make_calibration_batch(self, seeds, cal_samples):
        """
        Take up to cal_samples from provided seeds, 
        then fill the rest with uniform random draws.
        """
        seeds = [np.asarray(s, float) for s in seeds]
        n_var = len(self.xl)
        # sanity check
        for s in seeds:
            assert s.shape == (n_var,), "seed vector has wrong length"

        n_seed = min(len(seeds), cal_samples)
        X_seeded = np.stack(seeds[:n_seed], axis=0)

        if cal_samples > n_seed:
            n_rand = cal_samples - n_seed
            X_rand = np.random.uniform(self.xl, self.xu, (n_rand, n_var))
            return np.vstack([X_seeded, X_rand])
        else:
            return X_seeded

    def _parallel_calibration(self):
        print("[Calibration] Running parallel calibration samples...")
        # Calibrate the problem with the first location and targets in the lists
        location_idx = 0
        robot_translation = [self.robot_translation[location_idx]]
        mobile_base_translation = [self.mobile_base_translation[location_idx]]
        targets = [self.targets[location_idx]]
        targets_offset = [self.targets_offset[location_idx]]
        
        # round-robin assignment
        futures = []
        for i, x in enumerate(self._x_cal):
            actor = self.actors[i % len(self.actors)]
            futures.append(actor.evaluate.remote(
                x, targets, targets_offset,
                robot_translation, mobile_base_translation,
                self.min_joints, self.max_joints,
                self.alpha, self.beta, self.delta, self.gamma
            ))
        res = ray.get(futures)

        def bounds(arr):
            lo, hi = min(arr), max(arr)
            return (lo, hi if hi>lo else lo+1e-6)

        self.pose_bounds   = bounds([r['pose_error'] for r in res])
        self.rrt_bounds    = bounds([r['rrt_path_cost'] for r in res])
        self.torque_bounds = bounds([r['torque'] for r in res])
        self.delta_bounds  = bounds([r['delta_joint_score_rrmc'] for r in res])
        self.pos_bounds    = bounds([r['pos_error_rrmc'] for r in res])
        self.jcount_bounds = bounds([r['joint_count'] for r in res])

    def _evaluate(self, X, out, *args, **kwargs):
        futures = []
        for i in range(X.shape[0]):
            actor = self.actors[i % len(self.actors)]
            futures.append(actor.evaluate.remote(
                X[i], self.targets, self.targets_offset,
                self.robot_translation, self.mobile_base_translation,
                self.min_joints, self.max_joints,
                self.alpha, self.beta, self.delta, self.gamma
            ))
        res = ray.get(futures)

        # assemble F just as before…
        F = np.zeros((X.shape[0], self.n_obj))
        def lin(v, lo, hi):
            return np.clip((v - lo) / max(1e-8, hi - lo), 0, 1)

        for i, r in enumerate(res):
            p_lo, p_hi   = self.pose_bounds
            rr_lo, rr_hi = self.rrt_bounds
            t_lo, t_hi   = self.torque_bounds
            d_lo, d_hi   = self.delta_bounds
            pr_lo, pr_hi = self.pos_bounds
            jc_lo, jc_hi = self.jcount_bounds

            F[i, 0] = self.alpha * lin(r['pose_error'], p_lo, p_hi)
            F[i, 1] = self.beta  * lin(r['torque'], t_lo, t_hi)
            F[i, 2] = self.delta * lin(r['joint_count'], jc_lo, jc_hi)
            F[i, 3] = self.gamma * abs(r['conditioning_index'] - 1)

            # F[i, 4] = lin(r['delta_joint_score_rrmc'], d_lo, d_hi)
            # F[i, 5] = lin(r['pos_error_rrmc'], pr_lo, pr_hi)
            w_delta_rrmc, w_pos_rrmc = 0.5, 0.5   # or tune to your preferences

            # in _evaluate, replace the two objectives at indices 4,5 with one:
            F[i, 4] = (w_delta_rrmc * lin(r['delta_joint_score_rrmc'], d_lo, d_hi)
                       + w_pos_rrmc   * lin(r['pos_error_rrmc'],      pr_lo, pr_hi))
            F[i, 5] = lin(r['rrt_path_cost'], rr_lo, rr_hi)

        out["F"] = F


# -------- W&B Callback --------
class WandbLogger(Callback):
    def __init__(self):
        super().__init__()
        self.gen = 0
    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        mean_obj = F.mean(axis=0)
        
        # objective names
        obj_names = ['pose_error', 'torque', 'joint_count',
                     'conditioning_index', 'rrmc_score',
                     'rrt_path_cost']

        # log per-generation aggregates
        log_dict = {"generation": self.gen}
        log_dict.update({f'{obj_names[i]}_mean': mean_obj[i] for i in range(F.shape[1])})
        wandb.log(log_dict, step=self.gen)

        self.gen += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NSGA2 with optional W&B and mixed-mode logging")
    parser.add_argument('--max_joints',  type=int, default=7, help='Maximum joints in chain')
    parser.add_argument('--min_joints',  type=int, default=5, help='Minimum joints in chain')
    parser.add_argument('--population',   type=int, default=16, help='Population size')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--calibration_samples', type=int, default=6, help='Number of calibration samples')
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    parser.add_argument("--mixed", dest="mixed", action="store_true", help="Use mixed custom operators")
    parser.add_argument("--no-mixed", dest="mixed", action="store_false", help="Use standard Pymoo operators")
    parser.set_defaults(mixed=True)
    args = parser.parse_args()

    ################################################################
    ######################### STAGE INPUTS #########################
    use_wandb = args.wandb
    use_mixed = args.mixed
    
    # Setup kinematic chain parameters
    min_joints = args.min_joints
    max_joints = args.max_joints
    joint_type_bounds = (0, 2)  # 0: revolute, 1: prismatic, 2: fixed
    joint_axis_bounds = (0, 2)  # 0: x-axis, 1: y-axis, 2: z-axis
    link_length_bounds = (0.05, 0.75)  # min and max link length

    # Setup problem parameters
    num_generations = args.generations
    num_population = args.population
    calibration_samples = args.calibration_samples
    num_objectives = 6  # pose_error, torque, joint_count, conditioning_index, rrmc_score, rrt_path_cost
    num_actors = os.cpu_count() // 2  # tune this to control memory vs. throughput
    ray.init(num_cpus=num_actors)

    # Set the robot starting position and translation
    # CURRENT MESH X-BOUND: [-6.2, 7.7]
    robot_to_amiga_translation = [0, 0, 1.025]
    amiga_to_robot_translation = [0, -0.3, 0]
    robot_system_locations = [[-4.0, -2.25, 0.0], [-1.5, -2.25, 0.0], [1.0, -2.25, 0.0], [3.5, -2.25, 0.0], [6.0, -2.25, 0.0]]
    
    # Specify the window position and size for the loaded prune points (only uses prune points within this window)
    window_x_positions = [loc[0] for loc in robot_system_locations]
    window_size = 2.5
    ################################################################
    ################################################################

    # decision vector bounds
    xl = [min_joints] + [joint_type_bounds[0], joint_axis_bounds[0], link_length_bounds[0]] * max_joints
    xu = [max_joints] + [joint_type_bounds[1], joint_axis_bounds[1], link_length_bounds[1]] * max_joints
    decision_vector_bounds = (xl, xu)
    var_types = ['int'] + ['int','int','real'] * max_joints

    # Find the urdf seeds
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_dir = os.path.join(script_dir, 'urdf', 'robots', 'nsga2_seeds')
    seeds = load_seeds(urdf_dir, max_joints=max_joints)

    # Load the robot system translation
    robot_translations = []
    amiga_translations = []
    target_poses_list = []
    target_offset_poses_list = []
    for robot_system_translation in robot_system_locations:
        # Calculate the robot and amiga translations based on the system translation
        # This assumes the robot is always at the origin of the mobile base
        # and the amiga is offset by a fixed translation.
        robot_translations.append(np.add(robot_to_amiga_translation, robot_system_translation))
        amiga_translations.append(np.add(amiga_to_robot_translation, robot_system_translation))

        # Load the prune poses from the YAML file
        pose_data_results = orchard_ws.get_prune_poses_from_yaml(
            yaml_path='manipulator_codesign/prune_data/all_branches_info.yaml',
            robot_base=robot_translations[-1],
            window_size=window_size,
            min_y=None,
            max_y=0.0,
            )
        target_poses, target_offset_poses = orchard_ws.package_poses(pose_data_results)
        target_poses_list.append(target_poses)
        target_offset_poses_list.append(target_offset_poses)

    # Stage the problem
    problem = KinematicChainProblem(
        target_poses_list,
        target_offset_poses_list,
        robot_translation=robot_translations,
        mobile_base_translation=amiga_translations,
        seeds=seeds,
        dec_vec_bounds=decision_vector_bounds,
        min_joints=min_joints,
        max_joints=max_joints,
        cal_samples=calibration_samples,
        num_actors=num_actors,
        num_objectives=num_objectives,
    )

    callback = None
    if use_wandb:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key is None:
            raise RuntimeError("Please set WANDB_API_KEY in your environment to use W&B")
        wandb.login(key=api_key)
        wandb.init(
            project="manipulator_codesign",
            entity="rosettem-oregon-state-university",
            name=f"nsga2_run_{datetime.now():%Y%m%d_%H%M%S}",
            config={
                "pop_size": num_population,
                "n_gen": num_generations,
                "min_joints": problem.min_joints,
                "max_joints": problem.max_joints,
                "alpha": problem.alpha,
                "beta": problem.beta,
                "delta": problem.delta,
                "gamma": problem.gamma,
                "sampling": "Seeded+Mixed",
                "crossover": "MixedCrossover",
                "mutation": "MixedMutation"
            }
        )
        callback = WandbLogger()

    # algorithm selection
    if use_mixed:
        sampling  = SeededSampling(var_types, seeds, MixedSampling(var_types))
        crossover = MixedCrossover(var_types)
        mutation  = MixedMutation(var_types,
                                  prob_real=1.0/(1+3*max_joints),
                                  prob_int=0.1)
        algo = NSGA2(
            pop_size=num_population,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )
    else:
        algo = NSGA2(
            pop_size=num_population,
            sampling=LatinHypercubeSampling(),
            crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
            mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20),
            eliminate_duplicates=True
        )

    # dynamic minimize call: include callback only if set
    minimize_kwargs = {
        'problem': problem,
        'algorithm': algo,
        'termination': ('n_gen', num_generations),
        'seed': 1,
        'verbose': True
    }
    if callback is not None:
        minimize_kwargs['callback'] = callback

    res = minimize(**minimize_kwargs)

    # save results locally
    data_dir = 'data/nsga2_results'
    os.makedirs(data_dir, exist_ok=True)
    fn = os.path.join(data_dir, f"results_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    with open(fn, 'wb') as f:
        pickle.dump({'X': res.X, 'F': res.F}, f)
    print(f"Saved results to {fn}")

    if use_wandb:
        artifact = wandb.Artifact('nsga2-results', type='dataset')
        artifact.add_file(fn)
        wandb.log_artifact(artifact)
        wandb.finish()
