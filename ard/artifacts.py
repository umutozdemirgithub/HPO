from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .optimizers import _IMPORT_ERRORS as _OPTIMIZER_IMPORT_ERRORS, default_spaces
from .shap_utils import _IMPORT_ERRORS as _SHAP_IMPORT_ERRORS
from .wilcoxon import _IMPORT_ERRORS as _WILCOXON_IMPORT_ERRORS

_IMPORT_ERRORS: Dict[str, str] = {}
_IMPORT_ERRORS.update(_OPTIMIZER_IMPORT_ERRORS)
_IMPORT_ERRORS.update(_SHAP_IMPORT_ERRORS)
_IMPORT_ERRORS.update(_WILCOXON_IMPORT_ERRORS)

def export_latex_table(metrics_df: pd.DataFrame, out_path: str) -> str:
    cols = ["model", "optimizer", "MAE", "RMSE", "R2", "runtime_s"]
    df = metrics_df.copy()
    alias_map = {
        "model": ["model", "Model"],
        "optimizer": ["optimizer", "Optimizer"],
        "MAE": ["MAE", "mae"],
        "RMSE": ["RMSE", "rmse"],
        "R2": ["R2", "r2"],
        "runtime_s": ["runtime_s", "Runtime_s", "runtime", "time_s"],
    }
    for canonical, aliases in alias_map.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df[canonical] = df[alias]
                break
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"export_latex_table requires columns {cols}; missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df[cols].copy()
    df["runtime_s"] = df["runtime_s"].map(lambda x: f"{float(x):.2f}")
    for c in ["MAE", "RMSE", "R2"]:
        df[c] = df[c].map(lambda x: f"{float(x):.4f}")
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Model and optimizer comparison. Lower is better for MAE/RMSE.}",
        r"\begin{tabular}{l l r r r r}",
        r"\hline",
        r"Model & Optimizer & MAE & RMSE & $R^2$ & Time (s) \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        tex.append(f"{r['model']} & {r['optimizer']} & {r['MAE']} & {r['RMSE']} & {r['R2']} & {r['runtime_s']} \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    Path(out_path).write_text("\n".join(tex) + "\n", encoding="utf-8")
    return out_path


def export_supplementary_table(out_path: str) -> str:
    """Export hyperparameter search space as a supplementary CSV table (S1).

    Produces a table suitable for a journal 'Supplementary Material' section
    listing model, parameter name, type, and search range for every tunable
    hyperparameter used in the study.
    """
    rows = []
    for model_name in ["LightGBM", "XGBoost", "HistGB"]:
        space = default_spaces(model_name)
        for param, spec in space.items():
            row: Dict[str, Any] = {
                "Model": model_name,
                "Parameter": param,
                "Type": spec.kind,
                "Low": spec.low if spec.kind != "cat" else "",
                "High": spec.high if spec.kind != "cat" else "",
                "Choices": ", ".join(str(c) for c in spec.choices) if spec.choices else "",
                "Notes": (
                    "log-uniform sampling"
                    if spec.kind == "logfloat"
                    else ("integer" if spec.kind == "int" else "")
                ),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path

def make_all_standard_figures(res: Dict[str, Any]) -> List[str]:
    fig_dir = res.get("fig_dir")
    if not fig_dir or not os.path.isdir(fig_dir):
        return []
    return sorted([os.path.join(fig_dir, f) for f in os.listdir(fig_dir) if f.lower().endswith(".png")])

def make_shap_figures(res: Dict[str, Any]) -> List[str]:
    return make_all_standard_figures(res)

# -----------------------------
# Figure generation helper (extracted for clarity/testability)
# -----------------------------
from .constants import ZONE_CANDIDATES


def _compute_fold_stability(fold_tables: Dict[str, pd.DataFrame], out_dir: str, log) -> None:
    """Write fold_stability_summary.csv — called exactly once."""
    try:
        keys = list(fold_tables.keys())
        if not keys:
            return
        stab_rows = []
        for k in keys:
            d = fold_tables[k]
            if d is None or len(d) == 0:
                continue
            stab_rows.append({
                "method": k,
                "rmse_mean": float(d["rmse"].mean()),
                "rmse_std": float(d["rmse"].std(ddof=1) if len(d) > 1 else 0.0),
                "mae_mean": float(d["mae"].mean()),
                "r2_mean": float(d["r2"].mean()),
            })
        if stab_rows:
            pd.DataFrame(stab_rows).sort_values("rmse_mean").to_csv(os.path.join(out_dir, "fold_stability_summary.csv"), index=False)
    except Exception as _e:
        log(f"Fold stability summary skipped due to error: {_e}")



def _prepare_run_artifacts(base_out_dir: str, fixed_out_dir: Optional[str], generate_figures: bool, compute_diagnostics: bool) -> Tuple[str, str, str]:
    run_id = time.strftime("run_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    if fixed_out_dir is not None:
        out_dir = fixed_out_dir
        run_id = os.path.basename(fixed_out_dir)
    else:
        out_dir = os.path.join(base_out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    if generate_figures or compute_diagnostics:
        os.makedirs(fig_dir, exist_ok=True)
    return run_id, out_dir, fig_dir


def _write_run_metadata_file(
    out_dir: str,
    *,
    run_id: str,
    seed: int,
    folds: int,
    iters: int,
    time_series_cv: bool,
    nested_cv: bool,
    strict_time_order: bool,
    generate_figures: bool,
    compute_diagnostics: bool,
    with_shap: bool,
    shap_rows: int,
    date_column: Optional[str],
    datetime_format: Optional[str],
    feature_cols: List[str],
    target_cols: List[str],
    n_rows: int,
    models_selected: List[str],
    optimizers_selected: List[str],
    ui_metadata: Optional[Dict[str, Any]],
) -> None:
    import platform as _platform
    from importlib import metadata as _im
    pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "xgboost", "lightgbm", "shap", "optuna", "streamlit"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = _im.version(p)
        except Exception:
            pass
    meta = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": _platform.platform(),
        "seed": int(seed),
        "folds": int(folds),
        "iters": int(iters),
        "time_series_cv": bool(time_series_cv),
        "nested_cv": bool(nested_cv),
        "strict_time_order": bool(strict_time_order),
        "generate_figures": bool(generate_figures),
        "compute_diagnostics": bool(compute_diagnostics),
        "with_shap": bool(with_shap),
        "shap_rows": int(shap_rows),
        "shap_model_scope": "full_data_retrain",
        "shap_model_scope_note": (
            "SHAP values are computed on a model retrained on the full dataset. "
            "This is intentional for global explainability; generalisation is "
            "reported separately via nested CV metrics."
        ),
        "date_column": date_column,
        "datetime_format": datetime_format,
        "feature_cols": list(feature_cols),
        "target_cols": list(target_cols),
        "n_rows": int(n_rows),
        "n_features": int(len(feature_cols)),
        "package_versions": versions,
        "import_errors": dict(_IMPORT_ERRORS),
        "models_selected": list(models_selected),
        "optimizers_selected": list(optimizers_selected),
        "sidebar_selections": dict(ui_metadata or {}),
        "aggregation_mode_primary": (ui_metadata or {}).get("aggregation_mode_primary"),
        "aggregation_mode_secondary": (ui_metadata or {}).get("aggregation_mode_secondary"),
        "zone_detection_candidate": (ui_metadata or {}).get("zone_detection_candidate"),
        "zone_confirmation_state": (ui_metadata or {}).get("zone_confirmation_state"),
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)



