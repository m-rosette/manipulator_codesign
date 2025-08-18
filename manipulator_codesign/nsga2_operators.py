import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation


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