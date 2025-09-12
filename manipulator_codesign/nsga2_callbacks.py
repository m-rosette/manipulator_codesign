import os
import pickle
import tempfile
import wandb
import numpy as np
from typing import Optional, Any
from pymoo.core.callback import Callback


# -------- Combined Callback --------
class CombinedCallback(Callback):
    def __init__(self, *callbacks):
        super().__init__()
        self.callbacks = callbacks

    def notify(self, algorithm):
        for c in self.callbacks:
            # prefer notify method if present, otherwise call if callable
            if hasattr(c, "notify"):
                try:
                    c.notify(algorithm)
                    continue
                except Exception:
                    pass
            if callable(c):
                try:
                    c(algorithm)
                except Exception:
                    pass


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

# -------- Checkpoint Callback --------
class CheckpointCallback(Callback):
    def __init__(self, out_dir='data/nsga2_checkpoints', every=1, keep_last=0):
        super().__init__()
        self.out_dir = out_dir
        self.every = every
        self.keep_last = keep_last  # keep_last>0 will remove older checkpoints
        os.makedirs(self.out_dir, exist_ok=True)
        self.gen = 0

    def _atomic_dump(self, obj, fn):
        # write to tmp and atomically replace
        dirn = os.path.dirname(fn)
        with tempfile.NamedTemporaryFile(dir=dirn, delete=False) as tf:
            pickle.dump(obj, tf)
            tmpname = tf.name
        os.replace(tmpname, fn)

    def notify(self, algorithm):
        if self.gen % self.every == 0:
            pop = algorithm.pop
            try:
                X = pop.get("X")
                F = pop.get("F")
            except Exception:
                # fallback: iterate if pop.get not available
                X = None; F = None
                try:
                    X = np.vstack([ind.X for ind in pop])
                    F = np.vstack([ind.F for ind in pop])
                except Exception:
                    pass

            d = {
                "generation": self.gen,
                "X": X,
                "F": F,
            }
            fn = os.path.join(self.out_dir, f"checkpoint_gen_{self.gen:04d}.pkl")
            self._atomic_dump(d, fn)

            # remove older checkpoints if requested
            if self.keep_last > 0:
                files = sorted([f for f in os.listdir(self.out_dir) if f.startswith("checkpoint_gen_")])
                while len(files) > self.keep_last:
                    os.remove(os.path.join(self.out_dir, files.pop(0)))

        self.gen += 1

class FidelityCallback(Callback):
    """
    pymoo-style Callback that increments problem.current_generation once per generation.
    Accepts the problem instance but does not import its type to avoid circular imports.
    """
    def __init__(self, problem: Any):
        super().__init__()
        self.problem = problem
        # ensure attribute exists
        if not hasattr(self.problem, "current_generation"):
            try:
                setattr(self.problem, "current_generation", 0)
            except Exception:
                pass

    def notify(self, algorithm):
        try:
            # increment for next generation
            current = getattr(self.problem, "current_generation", 0)
            setattr(self.problem, "current_generation", current + 1)
        except Exception as e:
            print("FidelityCallback.notify error:", e)
            pass