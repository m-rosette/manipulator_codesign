import os
import pickle
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import ray
import wandb
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.core.callback import Callback 

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.urdf_to_decision_vector import encode_seed, urdf_to_decision_vector, load_seeds
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
import manipulator_codesign.orchard_workspace as orchard_ws
from pybullet_robokit.load_objects import LoadObjects


# -------- Seeded & Mixed Operators --------
class SeededSampling(Sampling):
    def __init__(self, var_types, seeds, fallback_sampler):
        super().__init__()
        self.var_types = var_types
        self.seeds = [np.asarray(x, dtype=float) for x in seeds]
        self.fallback = fallback_sampler
        n_var = len(var_types)
        for x in self.seeds:
            assert x.shape == (n_var,)
    def _do(self, problem, n_samples, **kwargs):
        print("[SeededSampling] Generating initial population with seeds...")
        n_seeds = min(len(self.seeds), n_samples)
        X_seeded = np.stack(self.seeds[:n_seeds], axis=0)
        n_rem = n_samples - n_seeds
        if n_rem > 0:
            X_rest = self.fallback._do(problem, n_rem, **kwargs)
            return np.vstack([X_seeded, X_rest])
        return X_seeded


class MixedSampling(Sampling):
    def __init__(self, var_types):
        super().__init__()
        self.var_types = var_types
        self.lhs = LatinHypercubeSampling()

    def _do(self, problem, n_samples, **kwargs):
        # 1) first generate the “real” vars
        X = self.lhs._do(problem, n_samples, **kwargs)

        # 2) now overwrite all integer slots with true randint [xl, xu] inclusive
        for i, t in enumerate(self.var_types):
            if t != 'real':
                lo, hi = int(problem.xl[i]), int(problem.xu[i])
                # +1 on hi to make it inclusive
                X[:, i] = np.random.randint(lo, hi + 1, size=n_samples)

        return X


class MixedCrossover(Crossover):
    def __init__(self, var_types, eta_sbx=10, prob_real=0.9, prob_int=0.5):
        super().__init__(2, 2)
        self.var_types = var_types
        self.sbx = SimulatedBinaryCrossover(prob=prob_real, eta=eta_sbx)
        self.uni = UniformCrossover(prob=prob_int)
    def _do(self, problem, X, **kwargs):
        real_idx = [i for i, t in enumerate(self.var_types) if t == 'real']
        int_idx = [i for i, t in enumerate(self.var_types) if t != 'real']
        Y = np.empty_like(X)
        if real_idx:
            sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                               xl=problem.xl[real_idx], xu=problem.xu[real_idx])
            Yr = self.sbx._do(sub_prob, X[:, :, real_idx], **kwargs)
            for idx, col in enumerate(real_idx):
                Y[:, :, col] = Yr[:, :, idx]
        if int_idx:
            Yi = self.uni._do(problem, X[:, :, int_idx], **kwargs)
            for idx, col in enumerate(int_idx):
                Y[:, :, col] = Yi[:, :, idx].round().astype(int)
        return Y


class MixedMutation(Mutation):
    def __init__(self, var_types, eta_pm=20, prob_real=None, prob_int=0.1):
        super().__init__(1, 1)
        self.var_types = var_types
        self.eta_pm = eta_pm
        self.prob_real = prob_real
        self.prob_int = prob_int
    def _do(self, problem, X, **kwargs):
        real_idx = [i for i, t in enumerate(self.var_types) if t == 'real']
        int_idx = [i for i, t in enumerate(self.var_types) if t != 'real']
        Y = X.copy()
        if X.ndim == 2:
            if real_idx:
                sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                                   xl=problem.xl[real_idx], xu=problem.xu[real_idx])
                Yr = PolynomialMutation(eta=self.eta_pm, prob=self.prob_real)._do(
                    sub_prob, X[:, real_idx], **kwargs)
                Y[:, real_idx] = Yr
            for col in int_idx:
                mask = np.random.rand(Y.shape[0]) < self.prob_int
                if mask.any():
                    lo, hi = int(problem.xl[col]), int(problem.xu[col])
                    Y[mask, col] = np.random.randint(lo, hi + 1, mask.sum())
        else:
            # 3D fallback
            n = X.shape[0]
            if real_idx:
                sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                                   xl=problem.xl[real_idx], xu=problem.xu[real_idx])
                Yr = PolynomialMutation(eta=self.eta_pm, prob=self.prob_real)._do(
                    sub_prob, X[:, 0, real_idx], **kwargs)
                Y[:, 0, real_idx] = Yr
            for col in int_idx:
                mask = np.random.rand(n) < self.prob_int
                if mask.any():
                    lo, hi = int(problem.xl[col]), int(problem.xu[col])
                    Y[mask, 0, col] = np.random.randint(lo, hi + 1, mask.sum())
        return Y


# -------- Ray Actor for Persistent Evaluation --------
@ray.remote
class Evaluator:
    def __init__(self,
                 mesh_path: str,
                 robot_urdf: str,
                 mobile_base_translation,
                 flags: int):
        import pybullet as p, pybullet_data
        self.p = p
        self.p.connect(p.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)

        # single‐time mesh load
        self.tree_collision_shape = self.p.createCollisionShape(
            shapeType=self.p.GEOM_MESH,
            fileName=mesh_path,
            flags=self.p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        self.robot_urdf = robot_urdf
        self.mobile_base_translation = mobile_base_translation
        self.flags = flags

    def evaluate(self, x, targets, targets_offset,
                 robot_translation, min_j, max_j,
                 alpha, beta, delta, gamma):
        p = self.p       

        # load robot
        object_loader = LoadObjects(p)
        amiga_id = object_loader.load_urdf(
            self.robot_urdf,
            start_pos=self.mobile_base_translation,
            start_orientation=[0,0,0],
            fix_base=True,
            flags=self.flags
        )

        # load tree once per evaluation
        tree_id = p.createMultiBody(
            baseCollisionShapeIndex=self.tree_collision_shape,
            baseVisualShapeIndex=-1,
            basePosition=[0,0,0]
        )
        object_loader.collision_objects.extend([amiga_id, tree_id])

        # decode and build kinematic chain
        n, types, axes, lengths = decode_decision_vector(x, min_j, max_j)
        ch = KinematicChainPyBullet(
            p, robot_translation,
            n, types, axes, lengths,
            collision_objects=object_loader.collision_objects
        )
        if not ch.is_built:
            ch.build_robot()
        ch.load_robot()
        ch.compute_chain_metrics(targets, targets_offset)

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
                 seeds, min_joints=2, max_joints=7,
                 alpha=1, beta=1, delta=1, gamma=1,
                 cal_samples=15, num_actors=4):
        print("[KinematicChainProblem] Initializing and calibrating...")
        self.targets = targets
        self.targets_offset = targets_offset
        self.robot_translation = np.asarray(robot_translation, dtype=float)
        self.mobile_base_translation = np.asarray(mobile_base_translation, dtype=float)
        self.min_joints, self.max_joints = min_joints, max_joints
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma

        xl = [min_joints] + [0,0,0.05] * max_joints
        xu = [max_joints] + [2,2,0.75] * max_joints
        super().__init__(n_var=len(xl), n_obj=6, xl=np.array(xl), xu=np.array(xu))

        self._x_cal = self._make_calibration_batch(seeds, cal_samples)

        # create a pool of Evaluator actors
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # mesh_path = "manipulator_codesign/manipulator_codesign/meshes/before_mesh_transformed.obj"
        # robot_urdf = "manipulator_codesign/manipulator_codesign/urdf/robots/amiga.urdf"
        mesh_path = os.path.join(script_dir, "meshes", "before_mesh_transformed.obj")
        robot_urdf = os.path.join(script_dir, "urdf", "robots", "amiga.urdf")
        flags = 0
        self.actors = [
            Evaluator.options(max_concurrency=1).remote(
                mesh_path,
                robot_urdf,
                self.mobile_base_translation,
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
        # round-robin assignment
        futures = []
        for i, x in enumerate(self._x_cal):
            actor = self.actors[i % len(self.actors)]
            futures.append(actor.evaluate.remote(
                x, self.targets, self.targets_offset,
                self.robot_translation,
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
                self.robot_translation,
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
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--mixed", dest="mixed", action="store_true",
        help="Use mixed custom operators"
    )
    parser.add_argument(
        "--no-mixed", dest="mixed", action="store_false",
        help="Use standard Pymoo operators"
    )
    parser.set_defaults(mixed=True)
    args = parser.parse_args()
    use_wandb = args.wandb
    use_mixed = args.mixed

    ray.init(num_cpus=os.cpu_count())

    # set up operators
    max_joints = 7
    num_generations = 150
    num_population = 250
    var_types = ['int'] + ['int','int','real'] * max_joints

    # Find the urdf seeds
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_dir = os.path.join(script_dir, 'urdf', 'robots', 'nsga2_seeds')
    seeds = load_seeds(urdf_dir, max_joints=max_joints)

    # Set the robot starting position and translation
    # CURRENT MESH X-BOUND: [-6.2, 7.7]
    robot_to_amiga_translation = [0, 0, 1.025]
    amiga_to_robot_translation = [0, -0.3, 0]
    robot_system_translation = [-5.0, -2.5, 0.0]

    robot_translation = np.add(robot_to_amiga_translation, robot_system_translation)
    amiga_translation = np.add(amiga_to_robot_translation, robot_system_translation)
    
    # Load target prune points 
    window_x_position = robot_system_translation[0]
    window_size = 2

    target_poses, target_offset_poses = orchard_ws.get_prune_poses_from_yaml(
        yaml_path='manipulator_codesign/prune_data/all_branches_info.yaml',
        robot_base=robot_translation,
        window_size=window_size,
        min_y=None,
        max_y=0.0,
        )
    
    print(len(target_poses))

    problem = KinematicChainProblem(
        target_poses,
        target_offset_poses,
        robot_translation=robot_translation,
        mobile_base_translation=amiga_translation,
        seeds=seeds,
        cal_samples=3,
        num_actors=os.cpu_count() // 2       # tune this to control memory vs. throughput
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
