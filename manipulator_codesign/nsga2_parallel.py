#!/usr/bin/env python3
"""
nsga2_ray_batched.py

This is your original NSGA2 entry script updated to:
 - create Ray Evaluator actors that expose evaluate_batch(...)
 - send chunks of decision vectors to each actor to amortize RPC / setup overhead

Behavior is intentionally conservative: each actor uses a single pybullet DIRECT client
and reuses the same Python helpers you already have (KinematicChainPyBullet, LoadObjects, etc).
If you want true multi-client-per-process packing (multiple pybullet clients inside one actor),
you can extend the Evaluator to create multiple clients and forward physicsClientId to your helper classes.
"""

import os
import pickle
from datetime import datetime
import argparse
import math
import time
import numpy as np
import ray
import wandb

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

from manipulator_codesign.nsga2_operators import SeededSampling, MixedSampling, MixedCrossover, MixedMutation
from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.urdf_to_decision_vector import load_seeds
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
import manipulator_codesign.orchard_workspace as orchard_ws
from pybullet_robokit.load_objects import LoadObjects
from nsga2_callbacks import CombinedCallback, WandbLogger, CheckpointCallback


# -------- Ray Actor for Persistent Evaluation (batched) --------
@ray.remote
class Evaluator:
    """
    Ray actor that holds one persistent pybullet DIRECT client and exposes evaluate_batch(...)
    items: list of tuples (x, targets, targets_offset, robot_translations, mobile_base_translations,
                          min_j, max_j, alpha, beta, delta, gamma)
    """
    def __init__(self, mesh_path: str, robot_urdf: str, flags: int, sim_dt: float = 0.02):
        # local import so actor process has pybullet available even if driver env differs
        import pybullet as p, pybullet_data
        self.p = p
        self.pybullet_data = pybullet_data
        self.robot_urdf = robot_urdf
        self.flags = flags
        self.sim_dt = sim_dt
        # create one DIRECT client for this actor process
        self._cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.sim_dt)

        # NOTE: Original code created a collision shape once in __init__.
        # many pybullet users find that resetSimulation invalidates handles, so
        # for safety we will (re)create collision shapes inside each evaluation.
        # But we keep other persistent items (like URDF path) here.
        self._mesh_path = mesh_path

    def _evaluate_single(self, x,
                         targets_list, targets_offset_list,
                         robot_translations_list, mobile_base_translations_list,
                         min_j, max_j,
                         alpha, beta, delta, gamma):
        """
        Almost identical to your original Evaluator.evaluate(...) body but
        adapted to run inside the actor's single persistent pybullet client.
        This function uses self.p (pybullet module) and the single client self._cid.
        """
        p = self.p
        cid = self._cid

        # reset simulation state at the start of each evaluation to keep
        # per-eval worlds independent and deterministic
        p.resetSimulation()

        # restore common global state for this client
        p.setGravity(0, 0, -9.81)

        # create or load any per-eval objects needed
        # create tree collision shape fresh (safe wrt resetSimulation)
        tree_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=self._mesh_path,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )

        # create object loader tied to this client (your LoadObjects expects a pybullet module)
        object_loader = LoadObjects(p)

        # load robot base (this will be repositioned per evaluation)
        amiga_id = object_loader.load_urdf(
            self.robot_urdf,
            start_pos=[0, 0, 0],
            start_orientation=[0, 0, 0],
            fix_base=True,
            flags=self.flags
        )

        # load the tree body (use the freshly created collision shape)
        tree_id = p.createMultiBody(
            baseCollisionShapeIndex=tree_collision_shape,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0]
        )
        object_loader.collision_objects.extend([amiga_id, tree_id])

        # decode candidate kinematic chain (uses your decode_decision_vector)
        n, types, axes, lengths = decode_decision_vector(x, min_j, max_j)

        ch = KinematicChainPyBullet(
            p, [0, 0, 0],
            n, types, axes, lengths,
            collision_objects=object_loader.collision_objects
        )

        if not ch.is_built:
            ch.build_robot()
        ch.load_robot()

        n_positions = len(robot_translations_list)
        assert n_positions == len(mobile_base_translations_list) == len(targets_list) == len(targets_offset_list), \
            "All lists of per-base inputs must have the same length"

        for i in range(n_positions):
            p.resetBasePositionAndOrientation(
                amiga_id,
                mobile_base_translations_list[i],
                p.getQuaternionFromEuler([0, 0, 0])
            )
            p.resetBasePositionAndOrientation(
                ch.robot.robotId,
                robot_translations_list[i],
                p.getQuaternionFromEuler([0, 0, 0])
            )
            p.resetBasePositionAndOrientation(
                tree_id,
                [0, 0, 0],
                p.getQuaternionFromEuler([0, 0, 0])
            )

            targets = ch.sample_collision_free_poses(targets_list[i])
            targets_offset = ch.sample_collision_free_poses(targets_offset_list[i])
            ch.compute_chain_metrics(targets, targets_offset)

        ch.compute_chain_metric_stats()

        # cleanup by resetting simulation again so no state leaks between evals
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

    def evaluate_batch(self, items):
        """
        items: list of tuples matching the args to _evaluate_single:
            (x, targets_list, targets_offset_list, robot_translations_list,
             mobile_base_translations_list, min_j, max_j, alpha, beta, delta, gamma)

        Returns: list of result dicts in the same order.
        """
        results = []
        for itm in items:
            # itm is expected to exactly match the arguments we pass from _evaluate
            res = self._evaluate_single(*itm)
            results.append(res)
        return results

    def close(self):
        try:
            self.p.disconnect(self._cid)
        except Exception:
            pass


# -------- Problem Definition (same structure but batch-aware) --------
class KinematicChainProblem(Problem):
    def __init__(self, targets, targets_offset, robot_translation, mobile_base_translation,
                 seeds, dec_vec_bounds, min_joints=2, max_joints=7,
                 alpha=1, beta=1, delta=1, gamma=1,
                 cal_samples=15, num_actors=4, num_objectives=6,
                 cal_target_samples=5, cal_downsample_seed=None):
        print("[KinematicChainProblem] Initializing and calibrating...")
        self.targets = targets
        self.targets_offset = targets_offset
        self.robot_translation = np.asarray(robot_translation, dtype=float)
        self.mobile_base_translation = np.asarray(mobile_base_translation, dtype=float)
        self.min_joints, self.max_joints = min_joints, max_joints
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma

        xl = dec_vec_bounds[0]
        xu = dec_vec_bounds[1]

        # number of poses to use during calibration for targets/targets_offset (default 5)
        self.cal_target_samples = int(cal_target_samples)
        # optional seed for reproducible random downsampling; set to None for non-deterministic
        self.cal_downsample_seed = cal_downsample_seed

        super().__init__(n_var=len(xl), n_obj=num_objectives, xl=np.array(xl), xu=np.array(xu))

        self._x_cal = self._make_calibration_batch(seeds, cal_samples)

        # create a pool of Evaluator actors
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mesh_path = os.path.join(script_dir, "meshes", "before_mesh_transformed.obj")
        robot_urdf = os.path.join(script_dir, "urdf", "robots", "amiga.urdf")
        flags = 0

        # spawn ray actors (each actor holds one local DIRECT pybullet client)
        self.actors = [
            Evaluator.options(max_concurrency=1).remote(
                mesh_path,
                robot_urdf,
                flags,
                sim_dt=0.02
            )
            for _ in range(num_actors)
        ]

        self._parallel_calibration()

    def _make_calibration_batch(self, seeds, cal_samples):
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
        
    def _maybe_random_downsample(self, poses, max_samples):
        """
        If poses contains more than max_samples, randomly downsample to exactly max_samples.
        Uses numpy's PCG64 via default_rng. If self.cal_downsample_seed is set, sampling is reproducible.
        """
        if poses is None:
            return poses
        n = len(poses)
        if n <= max_samples:
            return poses
        rng = np.random.default_rng(self.cal_downsample_seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        idx.sort()
        return [poses[i] for i in idx]

    def _parallel_calibration(self):
        print("[Calibration] Running parallel calibration samples...")
        # Calibrate the problem with the first location and targets in the lists
        location_idx = 0
        robot_translation = [self.robot_translation[location_idx]]
        mobile_base_translation = [self.mobile_base_translation[location_idx]]

        # original full lists for the location
        targets_full = self.targets[location_idx]
        targets_offset_full = self.targets_offset[location_idx]

        # if > cal_target_samples, randomly downsample to cal_target_samples
        targets_ds = self._maybe_random_downsample(targets_full, self.cal_target_samples)
        targets_offset_ds = self._maybe_random_downsample(targets_offset_full, self.cal_target_samples)

        targets = [targets_ds]
        targets_offset = [targets_offset_ds]

        # Prepare items for evaluate_batch
        items = []
        for x in self._x_cal:
            items.append((
                x, targets, targets_offset,
                robot_translation, mobile_base_translation,
                self.min_joints, self.max_joints,
                self.alpha, self.beta, self.delta, self.gamma
            ))

        # round-robin assign chunks across actors
        n_actors = len(self.actors)
        chunk_size = int(math.ceil(len(items) / max(1, n_actors)))
        futures = []
        for i, actor in enumerate(self.actors):
            start = i * chunk_size
            stop = min((i + 1) * chunk_size, len(items))
            if start >= stop:
                continue
            chunk = items[start:stop]
            futures.append(actor.evaluate_batch.remote(chunk))
        chunks = ray.get(futures)
        res = [r for chunk in chunks for r in chunk]

        def bounds(arr):
            lo, hi = min(arr), max(arr)
            return (lo, hi if hi > lo else lo + 1e-6)

        self.pose_bounds   = bounds([r['pose_error'] for r in res])
        self.rrt_bounds    = bounds([r['rrt_path_cost'] for r in res])
        self.torque_bounds = bounds([r['torque'] for r in res])
        self.delta_bounds  = bounds([r['delta_joint_score_rrmc'] for r in res])
        self.pos_bounds    = bounds([r['pos_error_rrmc'] for r in res])
        self.jcount_bounds = bounds([r['joint_count'] for r in res])

        print("[Calibration] Completed. Starting real evaluations now...")

    def _evaluate(self, X, out, *args, **kwargs):
        # Prepare batched items for each decision vector in X
        items = []
        for i in range(X.shape[0]):
            items.append((
                X[i], self.targets, self.targets_offset,
                self.robot_translation, self.mobile_base_translation,
                self.min_joints, self.max_joints,
                self.alpha, self.beta, self.delta, self.gamma
            ))

        # Distribute chunks across actors (contiguous chunks)
        n_actors = len(self.actors)
        chunk_size = int(math.ceil(len(items) / max(1, n_actors)))
        futures = []
        actor_chunks_info = []  # for reconstructing order
        for i, actor in enumerate(self.actors):
            start = i * chunk_size
            stop = min((i + 1) * chunk_size, len(items))
            if start >= stop:
                continue
            chunk = items[start:stop]
            actor_chunks_info.append((actor, start, stop))
            futures.append(actor.evaluate_batch.remote(chunk))

        # collect results
        chunks = ray.get(futures)
        # flatten preserving original order
        res = [None] * len(items)
        for (actor, start, stop), chunk_res in zip(actor_chunks_info, chunks):
            for local_idx, r in enumerate(chunk_res):
                res[start + local_idx] = r

        # assemble F
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

            w_delta_rrmc, w_pos_rrmc = 0.5, 0.5
            F[i, 4] = (w_delta_rrmc * lin(r['delta_joint_score_rrmc'], d_lo, d_hi)
                       + w_pos_rrmc   * lin(r['pos_error_rrmc'],      pr_lo, pr_hi))
            F[i, 5] = lin(r['rrt_path_cost'], rr_lo, rr_hi)

        out["F"] = F


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

    min_joints = args.min_joints
    max_joints = args.max_joints

    joint_type_bounds = (0, 2)  # 0: revolute, 1: prismatic, 2: fixed
    joint_axis_bounds = (0, 2)
    link_length_bounds = (0.05, 0.75)

    num_generations = args.generations
    num_population = args.population
    calibration_samples = args.calibration_samples
    num_objectives = 6

    # choose number of actors based on available CPUs
    available_cpus = os.cpu_count() or 1
    num_actors = max(1, available_cpus)  # you can tune this: set smaller if you want fewer processes
    print(f"[main] available_cpus={available_cpus}, num_actors={num_actors}")

    # initialize Ray with a CPU budget matching actor count
    ray.init(num_cpus=num_actors)

    robot_to_amiga_translation = [0, 0, 1.025]
    amiga_to_robot_translation = [0, -0.3, 0]
    robot_system_locations = [[-4.0, -2.25, 0.0], [-1.5, -2.25, 0.0], [1.0, -2.25, 0.0], [3.5, -2.25, 0.0], [6.0, -2.25, 0.0]]

    window_x_positions = [loc[0] for loc in robot_system_locations]
    window_size = 2.5

    data_dir = 'data/nsga2_results'
    os.makedirs(data_dir, exist_ok=True)
    ################################################################
    ################################################################

    xl = [min_joints] + [joint_type_bounds[0], joint_axis_bounds[0], link_length_bounds[0]] * max_joints
    xu = [max_joints] + [joint_type_bounds[1], joint_axis_bounds[1], link_length_bounds[1]] * max_joints
    decision_vector_bounds = (xl, xu)
    var_types = ['int'] + ['int','int','real'] * max_joints

    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_dir = os.path.join(script_dir, 'urdf', 'robots', 'nsga2_seeds')
    seeds = load_seeds(urdf_dir, max_joints=max_joints)

    # Load the robot system translation and prune poses
    robot_translations = []
    amiga_translations = []
    target_poses_list = []
    target_offset_poses_list = []
    for robot_system_translation in robot_system_locations:
        robot_translations.append(np.add(robot_to_amiga_translation, robot_system_translation))
        amiga_translations.append(np.add(amiga_to_robot_translation, robot_system_translation))

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

    callback = CheckpointCallback(out_dir=data_dir, every=1, keep_last=5)

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
        callback = CombinedCallback(callback, WandbLogger())

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

    fn = os.path.join(data_dir, f"results_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    with open(fn, 'wb') as f:
        pickle.dump({'X': res.X, 'F': res.F}, f)
    print(f"Saved results to {fn}")

    if use_wandb:
        artifact = wandb.Artifact('nsga2-results', type='dataset')
        artifact.add_file(fn)
        wandb.log_artifact(artifact)
        wandb.finish()

    # teardown Ray actors by shutting down Ray
    ray.shutdown()
