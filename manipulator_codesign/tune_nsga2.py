import os
import argparse
import pickle
import numpy as np
from datetime import datetime
import ray
import wandb

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.hyperparameters import HyperparameterProblem, MultiRun
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume

# ------------------------------------------------------------------
# Import your existing problem and seed utilities
from manipulator_codesign.urdf_to_decision_vector import load_seeds
from manipulator_codesign import orchard_workspace as orchard_ws
from manipulator_codesign.nsga2_parallel import KinematicChainProblem


# -------- W&B Callback for Hyperparameter Tuning --------
class WandbHPLogger(Callback):
    def __init__(self):
        super().__init__()
        self.eval = 0

    def notify(self, algorithm):
        # algorithm.opt is Optuna, algorithm.opt.evals stores trial results
        # log recent trial hypervolume
        if hasattr(algorithm, 'opt') and hasattr(algorithm.opt, 'trials'):
            trial = algorithm.opt.trials[-1]
            # trial.value is the performance (mean hv)
            wandb.log({'trial': self.eval, 'mean_hypervolume': trial.value}, step=self.eval)
            self.eval += 1

# ---------------------------------------------------------------------
# CLI & runtime defaults (matched with main GA script)
# ---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for NSGA-II (updated)")
    parser.add_argument('--trials',      type=int, default=10, help='Number of tuning evaluations (n_eval)')
    parser.add_argument('--tune_seeds',  type=int, default=3,  help='Different RNG seeds per eval (MultiRun)')
    parser.add_argument('--tune_gen',    type=int, default=3,  help='Generations per tuning evaluation')
    parser.add_argument('--wandb',       action='store_true', default=False, help='Enable W&B logging')
    parser.add_argument('--mixed',       dest='mixed', action='store_true', help='Use mixed custom operators')
    parser.add_argument('--no-mixed',    dest='mixed', action='store_false', help='Use standard Pymoo operators')
    parser.set_defaults(mixed=True)
    parser.add_argument('--max_joints',  type=int, default=7, help='Maximum joints in chain (for loading seeds)')
    parser.add_argument('--min_joints',  type=int, default=5, help='Minimum joints in chain')
    parser.add_argument('--pop_guess',   type=int, default=12, help='Reference population size to tune around')
    parser.add_argument('--data_yaml',   type=str, default='manipulator_codesign/prune_data/all_branches_info.yaml',
                        help='Path to prune pose YAML used by the GA')
    parser.add_argument('--window_size', type=float, default=2.0, help='Window size for selecting prune points')
    args = parser.parse_args()

    # W&B init (optional)
    if args.wandb:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key is None:
            raise RuntimeError("Please set WANDB_API_KEY in your environment to use W&B")
        wandb.login(key=api_key)
        wandb.init(
            project="manipulator_codesign_hyperparam",
            entity="rosettem-oregon-state-university",
            name=f"hp_opt_run_{datetime.now():%Y%m%d_%H%M%S}",
            config={
                "trials": args.trials,
                "tune_seeds": args.tune_seeds,
                "tune_gen": args.tune_gen,
                "min_joints": args.min_joints,
                "max_joints": args.max_joints,
                "pop_guess": args.pop_guess
            }
        )

    # -----------------------------------------------------------------
    # Mirror main GA parameter staging
    # -----------------------------------------------------------------
    # kinematic chain params (same defaults as your main script)
    min_joints = args.min_joints
    max_joints = args.max_joints
    joint_type_bounds = (0, 2)
    joint_axis_bounds = (0, 2)
    link_length_bounds = (0.05, 0.75)

    # problem-level defaults
    num_generations = args.tune_gen
    num_population  = args.pop_guess
    calibration_samples = 15  # tune-time default (pilot uses larger cal set if you want)
    num_objectives = 6

    # parallelism (match main)
    num_actors = max(1, (os.cpu_count() or 1) // 2)
    ray.init(num_cpus=num_actors)

    # translations used for selecting prune poses (copied from main)
    robot_to_amiga_translation = [0, 0, 1.025]
    amiga_to_robot_translation = [0, -0.3, 0]
    robot_system_translation = [-5.0, -2.5, 0.0]
    robot_translation = np.add(robot_to_amiga_translation, robot_system_translation)
    amiga_translation = np.add(amiga_to_robot_translation, robot_system_translation)

    # Load seeds and prune poses (same functions used by main)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_dir = os.path.join(script_dir, 'urdf', 'robots', 'nsga2_seeds')
    print("[Step 1] Loading URDF seeds from:", urdf_dir)
    seeds = load_seeds(urdf_dir, max_joints=max_joints)

    print("[Step 2] Loading prune poses from yaml")
    pose_data_results = orchard_ws.get_prune_poses_from_yaml(
        yaml_path=args.data_yaml,
        robot_base=robot_translation,
        window_size=args.window_size,
        min_y=None,
        max_y=0.0,
    )
    target_poses, target_offset_poses = orchard_ws.package_poses(pose_data_results)

    # Stage the problem (same constructor signature as your GA)
    print("[Step 3] Instantiating KinematicChainProblem for tuning runs")
    base_problem = KinematicChainProblem(
        target_poses,
        target_offset_poses,
        robot_translation=robot_translation,
        mobile_base_translation=amiga_translation,
        seeds=seeds,
        dec_vec_bounds=(
            [min_joints] + [joint_type_bounds[0], joint_axis_bounds[0], link_length_bounds[0]] * max_joints,
            [max_joints] + [joint_type_bounds[1], joint_axis_bounds[1], link_length_bounds[1]] * max_joints
        ),
        min_joints=min_joints,
        max_joints=max_joints,
        cal_samples=calibration_samples,
        num_actors=num_actors,
        num_objectives=num_objectives,
    )

    # -----------------------------------------------------------------
    # Pilot run to estimate hypervolume reference (small, fast)
    # -----------------------------------------------------------------
    print("[Step 4] Pilot NSGA-II run for hv_ref estimation")
    pilot_algo = NSGA2(pop_size=6, eliminate_duplicates=True)
    pilot = minimize(
        problem=base_problem,
        algorithm=pilot_algo,
        termination=('n_gen', max(1, num_generations // 2)),
        seed=1,
        verbose=False
    )
    F_pilot = pilot.F
    # compute reference as slightly larger than observed maxima
    f_max = np.max(F_pilot, axis=0)
    hv_ref = (f_max * 1.05).tolist()
    print(f"[Pilot] hv_ref = {hv_ref}")
    if args.wandb:
        wandb.log({'hv_ref': hv_ref})

    # -----------------------------------------------------------------
    # MultiRun wrapper: average hypervolume across multiple seeds
    # -----------------------------------------------------------------
    print("[Step 5] Creating MultiRun wrapper (measure mean hypervolume)")
    def func_stats(Fs):
        # Fs is a list of result objects; compute mean hypervolume
        return {"hv": np.mean([Hypervolume(ref_point=hv_ref).do(F.F) for F in Fs])}

    multi = MultiRun(
        problem=base_problem,
        seeds=list(range(1, args.tune_seeds + 1)),
        func_stats=func_stats,
        termination=('n_gen', num_generations)
    )

    # -----------------------------------------------------------------
    # Algorithm template: NSGA2 with tunable fields
    # -----------------------------------------------------------------
    print("[Step 6] Preparing NSGA2 template for tuning")
    # Keep a template with defaults. The HyperparameterProblem + Optuna wrapper
    # will propose values for fields that are present on this object.
    algo_template = NSGA2(
        pop_size=num_population,
        eliminate_duplicates=True
    )

    # If you want the tuner to explore mixed vs standard operators,
    # handle that inside the HyperparameterProblem or by making two
    # separate HyperparameterProblems. For now we leave sampling/crossover/mutation
    # unspecified so the tuner can try changing known algorithm attributes.

    # -----------------------------------------------------------------
    # Wrap into HyperparameterProblem and run Optuna
    # -----------------------------------------------------------------
    print("[Step 7] Wrapping into HyperparameterProblem and running Optuna")
    hp = HyperparameterProblem(algo_template, multi)

    hp_callback = WandbHPLogger() if args.wandb else None

    res = minimize(
        problem=hp,
        algorithm=Optuna(),               # Optuna wrapper from pymoo
        termination=('n_eval', args.trials),
        seed=42,
        callback=hp_callback,
        verbose=True
    )

    # -----------------------------------------------------------------
    # Extract best hyperparameters and save
    # -----------------------------------------------------------------
    best_params = res.X
    print("\n=== Best Hyperparameters ===")
    if isinstance(best_params, dict):
        for k, v in best_params.items():
            print(f"  {k} = {v}")
    else:
        # Some wrappers return structures; show raw for debugging
        print(best_params)

    if args.wandb:
        # log best params as key-values
        if isinstance(best_params, dict):
            wandb.log({f"best_{k}": (v.tolist() if hasattr(v, 'tolist') else v)
                       for k, v in best_params.items()})

    # Save results locally
    data_dir = 'data/hyperparam_opt'
    os.makedirs(data_dir, exist_ok=True)
    fn = os.path.join(data_dir, f"results_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    with open(fn, 'wb') as f:
        pickle.dump({'X': res.X, 'F': res.F}, f)
    print(f"Saved results to {fn}")

    if args.wandb:
        artifact = wandb.Artifact('hyperparam-opt-results', type='dataset')
        artifact.add_file(fn)
        wandb.log_artifact(artifact)
        wandb.finish()