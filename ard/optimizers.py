from __future__ import annotations

import hashlib
import os
import sys
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, TimeSeriesSplit

_IMPORT_ERRORS: Dict[str, str] = {}

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception as e:
    LGBMRegressor = None
    _HAS_LGBM = False
    _IMPORT_ERRORS["lightgbm"] = str(e)

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception as e:
    XGBRegressor = None
    _HAS_XGB = False
    _IMPORT_ERRORS["xgboost"] = str(e)

try:
    import optuna  # type: ignore
    from optuna.samplers import TPESampler  # type: ignore
    _HAS_OPTUNA = True
except Exception as e:
    optuna = None
    TPESampler = None
    _HAS_OPTUNA = False
    _IMPORT_ERRORS["optuna"] = str(e)

from .metrics import _metrics_core, eval_cv
from .preprocess import make_pipeline

from .constants import (
    DEFAULT_SEARCH_SPACE_CONFIG,
    DETERMINISTIC_ENV_VARS,
    HISTGB_BASE_PARAMS,
    HISTGB_SEED_PARAMS,
    LIGHTGBM_BASE_PARAMS,
    LIGHTGBM_DETERMINISTIC_PARAMS,
    LIGHTGBM_SEED_PARAMS,
    XGBOOST_BASE_PARAMS,
    XGBOOST_DETERMINISTIC_PARAMS,
    XGBOOST_SEED_PARAMS,
)

@dataclass
class ParamSpec:
    kind: str  # "int", "float", "logfloat", "cat"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None

def INT(low: int, high: int) -> ParamSpec:
    return ParamSpec(kind="int", low=float(low), high=float(high))

def FLOAT(low: float, high: float) -> ParamSpec:
    return ParamSpec(kind="float", low=float(low), high=float(high))

def LOGFLOAT(low: float, high: float) -> ParamSpec:
    return ParamSpec(kind="logfloat", low=float(low), high=float(high))

def CAT(choices: List[Any]) -> ParamSpec:
    return ParamSpec(kind="cat", choices=list(choices))

def sample_param(spec: ParamSpec, rng: np.random.Generator):
    if spec.kind == "int":
        return int(rng.integers(int(spec.low), int(spec.high) + 1))
    if spec.kind == "float":
        return float(rng.uniform(spec.low, spec.high))
    if spec.kind == "logfloat":
        lo, hi = math.log(spec.low), math.log(spec.high)
        return float(math.exp(rng.uniform(lo, hi)))
    if spec.kind == "cat":
        return spec.choices[int(rng.integers(0, len(spec.choices)))]
    raise ValueError(f"Unknown spec kind: {spec.kind}")

def clamp_param(spec: ParamSpec, val):
    if spec.kind == "int":
        return int(max(int(spec.low), min(int(spec.high), int(round(val)))))
    if spec.kind in ("float", "logfloat"):
        return float(max(spec.low, min(spec.high, float(val))))
    if spec.kind == "cat":
        if isinstance(val, (int, np.integer)) and 0 <= int(val) < len(spec.choices):
            return spec.choices[int(val)]
        return val if val in spec.choices else spec.choices[0]
    return val

def encode(space: Dict[str, ParamSpec], params: Dict[str, Any]) -> np.ndarray:
    """Encode a parameter dict into a flat numeric vector.

    logfloat parameters are encoded in log-space so that PSO/GA velocity
    arithmetic operates on a uniform scale (e.g. learning_rate: 0.01-0.2
    maps to log-space -4.6..-1.6 rather than linear 0.01..0.2).
    """
    vec = []
    for k, s in space.items():
        v = params[k]
        if s.kind == "cat":
            if v not in (s.choices or []):
                raise ValueError(f"Categorical value {v!r} is not in allowed choices for parameter {k!r}: {s.choices!r}")
            vec.append(float(s.choices.index(v)))
        elif s.kind == "logfloat":
            vec.append(math.log(max(float(v), 1e-300)))
        else:
            vec.append(float(v))
    return np.array(vec, dtype=float)

def decode(space: Dict[str, ParamSpec], vec: np.ndarray) -> Dict[str, Any]:
    """Decode a flat numeric vector back to a parameter dict.

    logfloat parameters are decoded from log-space (inverse of encode).
    Clamping is applied in the appropriate space before returning.
    """
    out: Dict[str, Any] = {}
    i = 0
    for k, s in space.items():
        v = vec[i]
        if s.kind == "cat":
            idx = int(round(v))
            idx = max(0, min(len(s.choices) - 1, idx))
            out[k] = s.choices[idx]
        elif s.kind == "logfloat":
            log_low = math.log(s.low)
            log_high = math.log(s.high)
            v_clamped = max(log_low, min(log_high, float(v)))
            out[k] = float(math.exp(v_clamped))
        else:
            out[k] = clamp_param(s, v)
        i += 1
    return out

# -----------------------------
# Preprocess / estimator helpers

def _aggregate_params_median_mode(params_list: List[Dict[str, Any]], space: Dict[str, "ParamSpec"]) -> Dict[str, Any]:
    """Aggregate per-fold best params — numeric: median, categorical: mode.

    Safe for nested CV; does not peek at outer test performance.
    """
    if not params_list:
        return {}

    out: Dict[str, Any] = {}
    for pname, spec in space.items():
        vals = [p[pname] for p in params_list if pname in p]
        if not vals:
            continue
        kind = getattr(spec, "kind", None)
        if kind in ("int", "float", "logfloat"):
            arr = np.asarray(vals, dtype=float)
            med = float(np.median(arr))
            if spec.low is not None:
                med = max(float(spec.low), med)
            if spec.high is not None:
                med = min(float(spec.high), med)
            out[pname] = int(round(med)) if kind == "int" else float(med)
        elif kind == "cat":
            counts: Dict[Any, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            best = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
            if spec.choices is not None and best not in spec.choices and len(spec.choices) > 0:
                best = spec.choices[0]
            out[pname] = best
        else:
            out[pname] = vals[0]
    return out

class Objective:
    """Inner-loop objective: minimize RMSE on a CV iterator."""
    def __init__(self, base_pipeline: Pipeline, space: Dict[str, ParamSpec], X: pd.DataFrame, y: np.ndarray, cv):
        self.base_pipeline = base_pipeline
        self.space = space
        self.X = X
        self.y = y
        self.cv = cv
        self.cache: Dict[str, Tuple[float, Dict[str, float]]] = {}

    @staticmethod
    def _make_cache_key(params: Dict[str, Any]) -> str:
        """Deterministic, collision-resistant cache key for a param dict.

        Uses MD5 over canonical JSON so that int/np.int64 for the same value
        map to the same key and the key is stable across Python processes
        (no PYTHONHASHSEED dependence unlike the built-in hash()).
        """
        def _canonical(v: Any) -> Any:
            if isinstance(v, (np.floating, float)):
                return ("f", format(float(v), ".17g"))
            if isinstance(v, (np.integer, int)):
                return ("i", int(v))
            if isinstance(v, np.bool_):
                return ("b", bool(v))
            if isinstance(v, (list, tuple)):
                return ("l", [_canonical(x) for x in v])
            if isinstance(v, dict):
                return ("d", sorted((str(kk), _canonical(vv)) for kk, vv in v.items()))
            return ("r", repr(v))
        canonical = sorted((str(k), _canonical(v)) for k, v in params.items())
        raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def evaluate(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        key = self._make_cache_key(params)
        if key in self.cache:
            return self.cache[key]
        pipe = clone(self.base_pipeline)
        pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
        _, mean = eval_cv(pipe, self.X, self.y, self.cv)
        score = mean["rmse"]
        self.cache[key] = (score, mean)
        return score, mean

# -----------------------------
# Metaheuristics
# -----------------------------
@dataclass
class OptResult:
    best_params: Dict[str, Any]
    best_score: float
    history: List[float]
    runtime_s: float

class BaseOptimizer:
    def __init__(self, space: Dict[str, ParamSpec], seed: int = 42):
        self.space = space
        self.rng = np.random.default_rng(seed)

    def random_params(self) -> Dict[str, Any]:
        return {k: sample_param(s, self.rng) for k, s in self.space.items()}

class PSOOptimizer(BaseOptimizer):
    def run(self, obj: Objective, iters: int = 40, n_particles: int = 18,
            w: float = 0.6, c1: float = 1.3, c2: float = 1.3, **kwargs) -> OptResult:
        t0 = time.time()
        dim = len(self.space)

        # Compute v_max per dimension in encoded space to prevent velocity explosion.
        # logfloat params are encoded in log-space, so their range is log(high)-log(low).
        v_max = np.empty(dim, dtype=float)
        for idx, (k, s) in enumerate(self.space.items()):
            if s.kind == "int":
                v_max[idx] = 0.5 * (s.high - s.low)
            elif s.kind == "float":
                v_max[idx] = 0.5 * (s.high - s.low)
            elif s.kind == "logfloat":
                v_max[idx] = 0.5 * (math.log(s.high) - math.log(s.low))
            elif s.kind == "cat":
                v_max[idx] = max(1.0, 0.5 * (len(s.choices) - 1))
            else:
                v_max[idx] = 1.0
        v_max = np.maximum(v_max, 1e-6)

        parts, vels, pbest, pbest_s = [], [], [], []
        for _ in range(n_particles):
            p = self.random_params()
            x = encode(self.space, p)
            # Initialize velocities within [-v_max, v_max]
            v = self.rng.uniform(-v_max * 0.1, v_max * 0.1, size=dim)
            s, _ = obj.evaluate(p)
            parts.append(x); vels.append(v)
            pbest.append(x.copy()); pbest_s.append(s)

        gbest_idx = int(np.argmin(pbest_s))
        gbest = pbest[gbest_idx].copy()
        gbest_s = float(pbest_s[gbest_idx])
        hist = [gbest_s]

        for _it in range(iters):
            for i in range(n_particles):
                r1 = self.rng.random(dim); r2 = self.rng.random(dim)
                vels[i] = w * vels[i] + c1 * r1 * (pbest[i] - parts[i]) + c2 * r2 * (gbest - parts[i])
                # Clamp velocity to prevent divergence (standard PSO best practice)
                vels[i] = np.clip(vels[i], -v_max, v_max)
                parts[i] = parts[i] + vels[i]
                p = decode(self.space, parts[i])
                parts[i] = encode(self.space, p)
                s, _ = obj.evaluate(p)
                if s < pbest_s[i]:
                    pbest[i] = parts[i].copy()
                    pbest_s[i] = s
                    if s < gbest_s:
                        gbest_s = s
                        gbest = parts[i].copy()
            hist.append(gbest_s)

        best_params = decode(self.space, gbest)
        return OptResult(best_params=best_params, best_score=float(gbest_s), history=hist, runtime_s=time.time() - t0)

class GAOptimizer(BaseOptimizer):
    def run(self, obj: Objective, iters: int = 40, pop: int = 24,
            mut_rate: float = 0.2, elite: int = 2, **kwargs) -> OptResult:
        t0 = time.time()
        P = [self.random_params() for _ in range(pop)]
        scores = [obj.evaluate(p)[0] for p in P]
        best_idx = int(np.argmin(scores))
        best = P[best_idx].copy()
        best_s = float(scores[best_idx])
        hist = [best_s]
        keys = list(self.space.keys())

        def tournament():
            a, b = int(self.rng.integers(0, pop)), int(self.rng.integers(0, pop))
            return P[a] if scores[a] < scores[b] else P[b]

        for _it in range(iters):
            order = np.argsort(scores)
            newP = [P[i] for i in order[:elite]]
            while len(newP) < pop:
                p1, p2 = tournament(), tournament()
                child: Dict[str, Any] = {}
                for k in keys:
                    spec = self.space[k]
                    if self.rng.random() < mut_rate:
                        # Mutation: resample from full space
                        child[k] = sample_param(spec, self.rng)
                    elif spec.kind in ("int", "float", "logfloat"):
                        # BLX-α blend crossover for numeric params:
                        # explore the interval [min-α·d, max+α·d] where d = |p1-p2|
                        ALPHA = 0.5
                        v1 = float(p1[k])
                        v2 = float(p2[k])
                        if spec.kind == "logfloat":
                            v1 = math.log(max(v1, 1e-300))
                            v2 = math.log(max(v2, 1e-300))
                            lo_b, hi_b = math.log(spec.low), math.log(spec.high)
                        else:
                            lo_b, hi_b = float(spec.low), float(spec.high)
                        d = abs(v1 - v2)
                        lo_blend = max(lo_b, min(v1, v2) - ALPHA * d)
                        hi_blend = min(hi_b, max(v1, v2) + ALPHA * d)
                        if hi_blend <= lo_blend:
                            val = self.rng.uniform(lo_b, hi_b)
                        else:
                            val = self.rng.uniform(lo_blend, hi_blend)
                        if spec.kind == "logfloat":
                            child[k] = float(math.exp(val))
                        elif spec.kind == "int":
                            child[k] = int(round(val))
                        else:
                            child[k] = float(val)
                        child[k] = clamp_param(spec, child[k])
                    else:
                        # Categorical: inherit from one parent
                        child[k] = p1[k] if self.rng.random() < 0.5 else p2[k]
                newP.append(child)
            P = newP
            scores = [obj.evaluate(p)[0] for p in P]
            best_idx = int(np.argmin(scores))
            if scores[best_idx] < best_s:
                best = P[best_idx].copy()
                best_s = float(scores[best_idx])
            hist.append(best_s)

        return OptResult(best_params=best, best_score=float(best_s), history=hist, runtime_s=time.time() - t0)

class ABCOptimizer(BaseOptimizer):
    def run(self, obj: Objective, iters: int = 40, pop: int = 20,
            limit: int = 10, **kwargs) -> OptResult:
        t0 = time.time()
        keys = list(self.space.keys())
        foods = [self.random_params() for _ in range(pop)]
        scores = [obj.evaluate(p)[0] for p in foods]
        trials = [0] * pop
        best_idx = int(np.argmin(scores))
        best = foods[best_idx].copy()
        best_s = float(scores[best_idx])
        hist = [best_s]

        def neighbor(i: int) -> Dict[str, Any]:
            j = int(self.rng.integers(0, pop))
            while j == i:
                j = int(self.rng.integers(0, pop))
            x_i = encode(self.space, foods[i])
            x_k = encode(self.space, foods[j])
            v = np.array(x_i, copy=True)
            dim = int(self.rng.integers(0, len(keys)))
            phi = float(self.rng.uniform(-1.0, 1.0))
            v[dim] = x_i[dim] + phi * (x_i[dim] - x_k[dim])
            return decode(self.space, v)

        for _it in range(iters):
            # Employed bee phase
            for i in range(pop):
                cand = neighbor(i)
                s, _ = obj.evaluate(cand)
                if s < scores[i]:
                    foods[i] = cand; scores[i] = s; trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bee phase
            fit = np.array([1.0 / (1e-9 + s) for s in scores])
            prob = fit / fit.sum()
            for _ in range(pop):
                i = int(self.rng.choice(np.arange(pop), p=prob))
                cand = neighbor(i)
                s, _ = obj.evaluate(cand)
                if s < scores[i]:
                    foods[i] = cand; scores[i] = s; trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bee phase
            for i in range(pop):
                if trials[i] >= limit:
                    foods[i] = self.random_params()
                    scores[i] = obj.evaluate(foods[i])[0]
                    trials[i] = 0

            best_idx = int(np.argmin(scores))
            if scores[best_idx] < best_s:
                best = foods[best_idx].copy()
                best_s = float(scores[best_idx])
            hist.append(best_s)

        return OptResult(best_params=best, best_score=float(best_s), history=hist, runtime_s=time.time() - t0)

class RandomSearchOptimizer(BaseOptimizer):
    """Baseline random search over the hyperparameter space."""
    def run(self, obj: Objective, iters: int = 40, **kwargs) -> OptResult:
        t0 = time.time()
        best_params: Dict[str, Any] = {}
        best_s = float('inf')
        hist: List[float] = []
        n = max(1, int(iters))
        for _ in range(n):
            p = self.random_params() if len(self.space) else {}
            s, _ = obj.evaluate(p)
            if s < best_s:
                best_s = float(s)
                best_params = dict(p)
            hist.append(best_s)
        return OptResult(best_params=best_params, best_score=float(best_s), history=hist, runtime_s=time.time() - t0)

class TPEOptimizer(BaseOptimizer):
    """Bayesian Optimization using Optuna's TPE sampler."""
    def run(self, obj: Objective, iters: int = 40, **kwargs) -> OptResult:
        if not _HAS_OPTUNA:
            raise RuntimeError(
                "Optuna is not installed in the active interpreter.\n"
                f"Python executable: {sys.executable}\n"
                "Install with:\n"
                f"  {sys.executable} -m pip install optuna\n"
                "Or remove 'TPE' from optimizers.\n"
                "Tip: run Streamlit via `python -m streamlit run app_streamlit.py` to avoid interpreter mismatch."
            )

        t0 = time.time()
        seed = int(self.rng.integers(0, 2**31 - 1))
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        hist: List[float] = []

        def _suggest(trial):
            params: Dict[str, Any] = {}
            for k, spec in self.space.items():
                if spec.kind == 'int':
                    params[k] = trial.suggest_int(k, int(spec.low), int(spec.high))
                elif spec.kind == 'float':
                    params[k] = trial.suggest_float(k, float(spec.low), float(spec.high))
                elif spec.kind == 'logfloat':
                    params[k] = trial.suggest_float(k, float(spec.low), float(spec.high), log=True)
                elif spec.kind == 'cat':
                    params[k] = trial.suggest_categorical(k, list(spec.choices or []))
                else:
                    raise ValueError(f"Unknown ParamSpec kind: {spec.kind}")
            return params

        def objective_fn(trial):
            p = _suggest(trial) if len(self.space) else {}
            s, _ = obj.evaluate(p)
            prev_best = float(hist[-1]) if hist else float('inf')
            hist.append(float(min(prev_best, s)))
            return float(s)

        study.optimize(objective_fn, n_trials=max(1, int(iters)), n_jobs=1, show_progress_bar=False)

        best_params = dict(study.best_params) if study.best_trial is not None else {}
        best_score = float(study.best_value) if study.best_trial is not None else float('inf')
        if not hist:
            hist = [best_score]
        return OptResult(best_params=best_params, best_score=best_score, history=hist, runtime_s=time.time() - t0)

# -----------------------------
# Time features (optional)
# -----------------------------

def _apply_seed_params(params: Dict[str, Any], names: Tuple[str, ...], seed: int) -> Dict[str, Any]:
    """Set every supported seed parameter to the same integer seed."""
    out = dict(params)
    for name in names:
        out[name] = int(seed)
    return out


def default_models(seed: int, deterministic: bool = False) -> Dict[str, Any]:
    """Return model instances with paper-grade best-effort reproducibility controls.

    Deterministic mode deliberately favors repeatability over speed: common BLAS
    and OpenMP backends are restricted to one thread, and model-specific seed
    aliases are all populated. Exact bitwise identity can still depend on OS,
    compiler, and third-party library versions; use the provided Docker/conda
    files for stronger environment-level reproducibility.
    """
    models: Dict[str, Any] = {}
    n_jobs = 1 if deterministic else -1
    if deterministic:
        for key, value in DETERMINISTIC_ENV_VARS.items():
            os.environ.setdefault(key, value)

    if _HAS_LGBM:
        lgbm_kwargs: Dict[str, Any] = dict(LIGHTGBM_BASE_PARAMS)
        lgbm_kwargs.update(n_jobs=n_jobs)
        lgbm_kwargs = _apply_seed_params(lgbm_kwargs, LIGHTGBM_SEED_PARAMS, seed)
        if deterministic:
            lgbm_kwargs.update(LIGHTGBM_DETERMINISTIC_PARAMS)
        models["LightGBM"] = LGBMRegressor(**lgbm_kwargs)

    if _HAS_XGB:
        xgb_kwargs: Dict[str, Any] = dict(XGBOOST_BASE_PARAMS)
        xgb_kwargs.update(n_jobs=n_jobs)
        xgb_kwargs = _apply_seed_params(xgb_kwargs, XGBOOST_SEED_PARAMS, seed)
        if deterministic:
            xgb_kwargs.update(XGBOOST_DETERMINISTIC_PARAMS)
        models["XGBoost"] = XGBRegressor(**xgb_kwargs)

    hist_kwargs: Dict[str, Any] = dict(HISTGB_BASE_PARAMS)
    hist_kwargs = _apply_seed_params(hist_kwargs, HISTGB_SEED_PARAMS, seed)
    models["HistGB"] = HistGradientBoostingRegressor(**hist_kwargs)
    return models


def _spec_from_config(config: Tuple[str, Optional[float], Optional[float], Optional[List[Any]]]) -> ParamSpec:
    kind, low, high, choices = config
    return ParamSpec(kind=str(kind), low=None if low is None else float(low), high=None if high is None else float(high), choices=None if choices is None else list(choices))


def default_spaces(model_name: str) -> Dict[str, ParamSpec]:
    """Build a model search space from constants.DEFAULT_SEARCH_SPACE_CONFIG."""
    model_config = DEFAULT_SEARCH_SPACE_CONFIG.get(model_name, {})
    return {name: _spec_from_config(spec) for name, spec in model_config.items()}

# -----------------------------
# Paper utilities / CLI API
# -----------------------------

