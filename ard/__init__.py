"""Advanced Research Dashboard package."""

from __future__ import annotations

import os

# Prevent OpenMP/BLAS oversubscription stalls before heavy sklearn imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .engine_core import (
    _HAS_LGBM,
    _HAS_OPTUNA,
    _HAS_SHAP,
    _HAS_XGB,
    _IMPORT_ERRORS,
    _build_global_weighted_metrics,
    _prepare_dataset,
    _run_shap_pipeline,
    run_experiment,
    run_experiment_from_path,
    run_experiment_multi_zone,
    run_experiment_multi_seed,
    parse_seed_list,
    run_experiment_safe,
)
from .metrics import _metrics, _metrics_core, eval_cv, eval_cv_with_oof
from .optimizers import (
    ABCOptimizer,
    BaseOptimizer,
    CAT,
    FLOAT,
    GAOptimizer,
    INT,
    LOGFLOAT,
    Objective,
    OptResult,
    PSOOptimizer,
    ParamSpec,
    RandomSearchOptimizer,
    TPEOptimizer,
    clamp_param,
    decode,
    default_models,
    default_spaces,
    encode,
    sample_param,
)
from .preprocess import (
    _read_csv_with_fallbacks,
    _set_best_effort_determinism,
    detect_zone_column,
    ensure_cyclical_time_features,
    make_pipeline,
    make_preprocess,
    read_uploaded_csv_with_fallbacks,
)
from .shap_utils import compute_shap, create_shap_comparison_artifacts
from .xai_artifacts import create_pfi_artifacts, create_lime_artifacts, create_xai_agreement_artifacts
from .advanced_visuals import create_prediction_diagnostic_figures, create_multi_seed_figures

__all__ = [
    "_HAS_LGBM", "_HAS_OPTUNA", "_HAS_SHAP", "_HAS_XGB", "_IMPORT_ERRORS",
    "_build_global_weighted_metrics", "_prepare_dataset", "_run_shap_pipeline",
    "run_experiment", "run_experiment_from_path", "run_experiment_multi_zone", "run_experiment_multi_seed", "parse_seed_list", "run_experiment_safe",
    "_metrics", "_metrics_core", "eval_cv", "eval_cv_with_oof",
    "ABCOptimizer", "BaseOptimizer", "CAT", "FLOAT", "GAOptimizer", "INT", "LOGFLOAT",
    "Objective", "OptResult", "PSOOptimizer", "ParamSpec", "RandomSearchOptimizer", "TPEOptimizer",
    "clamp_param", "decode", "default_models", "default_spaces", "encode", "sample_param",
    "_read_csv_with_fallbacks", "_set_best_effort_determinism", "detect_zone_column",
    "ensure_cyclical_time_features", "make_pipeline", "make_preprocess", "read_uploaded_csv_with_fallbacks",
    "compute_shap", "create_shap_comparison_artifacts",
    "create_pfi_artifacts", "create_lime_artifacts", "create_xai_agreement_artifacts",
    "create_prediction_diagnostic_figures", "create_multi_seed_figures",
]
