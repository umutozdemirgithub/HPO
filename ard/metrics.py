from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

def _metrics_core(y_true: np.ndarray, y_pred: np.ndarray, *, drop_non_finite: bool = False, include_n: bool = False) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if drop_non_finite:
        mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        y_true_arr = y_true_arr[mask]
        y_pred_arr = y_pred_arr[mask]
        if y_true_arr.size == 0:
            out = {
                "mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "medae": float("nan"), "explained_variance": float("nan"), "mape": float("nan"),
                "max_error": float("nan"), "bias": float("nan"), "nrmse": float("nan"),
            }
            if include_n:
                out["n"] = 0
            return out

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    medae = float(median_absolute_error(y_true_arr, y_pred_arr))
    explained_variance = float(explained_variance_score(y_true_arr, y_pred_arr))
    max_err = float(max_error(y_true_arr, y_pred_arr))
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_true = np.where(np.abs(y_true_arr) < 1e-12, np.nan, y_true_arr)
        ape = np.abs((y_true_arr - y_pred_arr) / safe_true)
    mape = float(np.nanmean(ape) * 100.0) if np.any(~np.isnan(ape)) else float("nan")
    bias = float(np.mean(y_pred_arr - y_true_arr))
    denom = float(np.nanmax(y_true_arr) - np.nanmin(y_true_arr))
    nrmse = float(rmse / denom) if math.isfinite(denom) and abs(denom) > 1e-12 else float("nan")
    out = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "medae": medae,
        "explained_variance": explained_variance,
        "mape": mape,
        "max_error": max_err,
        "bias": bias,
        "nrmse": nrmse,
    }
    if include_n:
        out["n"] = int(y_true_arr.size)
    return out

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return _metrics_core(y_true, y_pred, drop_non_finite=False, include_n=False)

def eval_cv(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, cv) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X)):
        est = clone(pipeline)
        est.fit(X.iloc[tr], y[tr])
        pred = est.predict(X.iloc[te])
        rows.append({"fold": fold, **_metrics(y[te], pred)})
    df = pd.DataFrame(rows)
    mean = {col: float(df[col].mean()) for col in df.columns if col != "fold"}
    return df, mean

def eval_cv_with_oof(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv,
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    """Evaluate via CV and return (fold_metrics_df, mean_metrics, oof_predictions)."""
    rows: List[Dict[str, Any]] = []
    oof = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)

    for fold, (tr, te) in enumerate(cv.split(X)):
        est = clone(pipeline)
        est.fit(X.iloc[tr], y[tr])
        pred = est.predict(X.iloc[te])
        oof[te] = pred
        rows.append({"fold": fold, **_metrics(y[te], pred)})

    df = pd.DataFrame(rows)
    mean = {col: float(df[col].mean()) for col in df.columns if col != "fold"}
    return df, mean, oof

# -----------------------------
# Objective
# -----------------------------

