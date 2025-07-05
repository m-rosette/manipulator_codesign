import os
import argparse
import pickle
import numpy as np
from datetime import datetime
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

# ------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for NSGA-II with W&B")
    parser.add_argument('--trials',    type=int, default=3, help='Number of tuning evaluations')
    parser.add_argument('--tune_seeds',type=int, default=3, help='Different RNG seeds per eval')
    parser.add_argument('--tune_gen',  type=int, default=2,help='Generations per tuning evaluation')
    parser.add_argument('--n_pts',     type=int, default=5,help='Number of target points')
    parser.add_argument('--max_joints',type=int, default=7,help='Maximum joints in chain')
    parser.add_argument('--wandb',     action='store_true',default=False, help='Enable W&B logging')
    args = parser.parse_args()

    # Initialize W&B if requested
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
                "n_pts": args.n_pts,
                "max_joints": args.max_joints
            }
        )

    # 1) Load seeds and define target configurations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_dir = os.path.join(script_dir, 'urdf', 'robots', 'nsga2_seeds')
    print("[Step 1] Loading URDF seeds from:", urdf_dir)
    seeds   = load_seeds(urdf_dir, max_joints=args.max_joints)
    targets = np.random.uniform([-2,0,0], [2,2,2], (args.n_pts,3)).tolist()

    # 2) Instantiate the base multi-objective problem
    print("[Step 2] Initializing KinematicChainProblem...")
    base_problem = KinematicChainProblem(targets, seeds=seeds, cal_samples=15)

    # 3) Automatic hypervolume reference estimation via pilot NSGA-II run
    print("[Step 3] Starting pilot NSGA-II run for hv_ref estimation...")
    pilot_algo = NSGA2(pop_size=5, eliminate_duplicates=True)
    pilot = minimize(
        problem=base_problem,
        algorithm=pilot_algo,
        termination=('n_gen', args.tune_gen),
        seed=1,
        verbose=False
    )
    F_pilot = pilot.F
    f_max = F_pilot.max(axis=0)
    hv_ref = f_max * 1.05
    print(f"[Pilot] hv_ref = {hv_ref}")
    if args.wandb:
        wandb.log({'hv_ref': hv_ref.tolist()})

    # 4) Define a MultiRun performance wrapper that measures mean hypervolume
    print("[Step 4] Creating MultiRun performance wrapper...")
    multi = MultiRun(
        problem=base_problem,
        seeds=list(range(1, args.tune_seeds+1)),
        func_stats=lambda Fs: {
            "hv": np.mean([Hypervolume(ref_point=hv_ref).do(F.F) for F in Fs])
        },
        termination=('n_gen', args.tune_gen)
    )

    # 5) Create a template NSGA2 (its parameters will be tuned)
    print("[Step 5] Preparing NSGA2 template for tuning...")
    algo_template = NSGA2()

    # 6) Wrap into a HyperparameterProblem
    print("[Step 6] Wrapping algorithm & performance into HyperparameterProblem...")
    hp = HyperparameterProblem(algo_template, multi)

    # 7) Run the hyperparameter optimization with Optuna
    print("[Step 7] Starting hyperparameter tuning with Optuna...")
    hp_callback = WandbHPLogger() if args.wandb else None
    res = minimize(
        problem=hp,
        algorithm=Optuna(),
        termination=('n_eval', args.trials),
        seed=42,
        callback=hp_callback,
        verbose=True
    )

    # 8) Extract and display best hyperparameters
    best_params = res.X
    print("\n=== Best Hyperparameters ===")
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    if args.wandb:
        wandb.log({f"best_{k}": v for k, v in best_params.items()})

    # 9) Save results locally
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
