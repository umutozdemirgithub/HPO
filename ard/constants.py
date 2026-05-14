"""Shared constants for the Advanced Research Dashboard.

Keeping UI labels, artifact groups, and dataset heuristics in one module avoids
silent drift between Streamlit, the engine, and reporting code.
"""
from __future__ import annotations

DEFAULT_FOLDS = 3
DEFAULT_ITERS = 40
DEFAULT_SEED = 42
DEFAULT_SHAP_ROWS = 800
FAST_SHAP_ROWS = 300
MAX_UPLOAD_MB = 100

METRIC_RENAME_MAP = {
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "r2": "R2",
    "medae": "MedianAE",
    "explained_variance": "ExplainedVar",
    "mape": "MAPE",
    "max_error": "MaxError",
    "bias": "Bias",
    "nrmse": "NRMSE",
}

FIGURE_GROUPS = {
    "Prediction fidelity": [
        "prediction_vs_time_*.png", "actual_vs_predicted_*.png",
        "absolute_error_*.png", "top_methods_error_bar.png",
    ],
    "Residual diagnostics": ["residual_*.png", "absolute_error_*.png"],
    "Optimization & runtime": [
        "convergence_*.png", "optimizer_convergence_overlay.png",
        "pareto_frontier.png", "rmse_runtime.png", "ablation_iters_*.png",
    ],
    "Metric heatmaps": ["metric_performance_heatmap.png", "fold_rmse_heatmap.png"],
    "Data profiling": ["timeseries_power.png", "ssrd_scatter.png", "correlation_heatmap.png"],
    "Fold stability": ["fold_stability_rmse_box.png", "fold_rmse_heatmap.png"],
    "Wilcoxon": ["wilcoxon_*.png"],
    "Multi-seed stability": ["multi_seed_*.png"],
}

ZONE_CANDIDATES = [
    "zone", "ZONE", "Zone", "ZONEID", "ZoneID", "zoneid", "zone_id", "Zone_Id",
    "solar_zone", "SolarZone", "region", "REGION", "Region", "site", "SITE", "Site",
    "building", "BUILDING", "Building", "plant", "PLANT", "Plant", "location", "Location",
]

ALLOWED_UPLOAD_EXTENSIONS = {"csv", "txt", "xlsx", "mat"}
ALLOWED_UPLOAD_MIME_HINTS = {
    "text/csv",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",  # browsers often use this for .mat files
    "application/x-matlab-data",
}


# Deterministic execution controls. These are best-effort controls; exact bitwise
# reproducibility can still vary by OS, compiler, BLAS/OpenMP, and library version.
DETERMINISTIC_ENV_VARS = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

# Model defaults are kept outside the model-construction function so CLI,
# Streamlit, tests, and future configs can refer to the same source of truth.
LIGHTGBM_BASE_PARAMS = {
    "verbose": -1,
}
LIGHTGBM_DETERMINISTIC_PARAMS = {
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
}
LIGHTGBM_SEED_PARAMS = (
    "random_state",
    "bagging_seed",
    "feature_fraction_seed",
    "data_random_seed",
    "drop_seed",
)

XGBOOST_BASE_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "verbosity": 0,
}
XGBOOST_DETERMINISTIC_PARAMS = {
    "nthread": 1,
}
XGBOOST_SEED_PARAMS = (
    "random_state",
    "seed",
)

HISTGB_BASE_PARAMS = {}
HISTGB_SEED_PARAMS = ("random_state",)

# Search-space specs use a serializable representation to avoid a circular
# dependency between constants.py and optimizers.ParamSpec.
# Tuple format: (kind, low, high, choices)
DEFAULT_SEARCH_SPACE_CONFIG = {
    "LightGBM": {
        "n_estimators": ("int", 200, 1200, None),
        "learning_rate": ("logfloat", 0.01, 0.2, None),
        "num_leaves": ("int", 31, 255, None),
        "subsample": ("float", 0.7, 1.0, None),
        "colsample_bytree": ("float", 0.7, 1.0, None),
        "reg_lambda": ("float", 0.0, 5.0, None),
        "min_child_samples": ("int", 5, 80, None),
    },
    "XGBoost": {
        "n_estimators": ("int", 200, 1200, None),
        "learning_rate": ("logfloat", 0.01, 0.2, None),
        "max_depth": ("int", 3, 12, None),
        "subsample": ("float", 0.7, 1.0, None),
        "colsample_bytree": ("float", 0.7, 1.0, None),
        "reg_lambda": ("float", 0.0, 5.0, None),
        "min_child_weight": ("logfloat", 0.5, 10.0, None),
    },
    "HistGB": {
        "learning_rate": ("logfloat", 0.01, 0.2, None),
        "max_depth": ("int", 3, 12, None),
        "max_leaf_nodes": ("int", 15, 63, None),
        "min_samples_leaf": ("int", 5, 80, None),
    },
}

OPTIMIZER_RUNTIME_DEFAULTS = {
    "PSO": {"min_particles": 12, "max_particles": 30, "particles_per_dim": 2, "particle_offset": 12},
    "GA": {"min_pop": 16, "max_pop": 40, "pop_per_dim": 3, "pop_offset": 12},
    "ABC": {"min_pop": 14, "max_pop": 30, "pop_per_dim": 2, "pop_offset": 10, "limit_fraction": 4, "min_limit": 6},
}
