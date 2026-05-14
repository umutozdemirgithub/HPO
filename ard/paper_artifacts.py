from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "item"


def _method_key(model: Any, optimizer: Any) -> str:
    return f"{model}__{optimizer}"


def _split_report_key(key: str) -> Dict[str, str]:
    zone = ""
    method_key = key
    if "::" in key:
        zone, method_key = key.split("::", 1)
    if "__" in method_key:
        model, optimizer = method_key.split("__", 1)
    else:
        model, optimizer = method_key, ""
    return {"zone": zone, "model": model, "optimizer": optimizer, "method_key": method_key}


def write_best_hyperparameters_table(best_params_report: Dict[str, Dict[str, Any]], out_dir: str) -> Optional[str]:
    rows: List[Dict[str, Any]] = []
    for key, rep in (best_params_report or {}).items():
        meta = _split_report_key(key)
        params = rep.get("selected_params", {}) if isinstance(rep, dict) else {}
        row: Dict[str, Any] = {
            "zone": meta["zone"],
            "model": meta["model"],
            "optimizer": meta["optimizer"],
            "method_key": meta["method_key"],
            "nested_cv": bool(rep.get("nested_cv", False)) if isinstance(rep, dict) else False,
            "selection_rule": rep.get("selection_rule", "") if isinstance(rep, dict) else "",
        }
        if isinstance(params, dict):
            for p, v in params.items():
                row[str(p)] = v
        rows.append(row)
    if not rows:
        return None
    path = os.path.join(out_dir, "best_hyperparameters.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_rank_table(metrics_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    if metrics_df is None or metrics_df.empty:
        return None
    df = metrics_df.copy()
    required = {"model", "optimizer", "rmse", "mae", "r2"}
    if not required.issubset(df.columns):
        return None
    df["method"] = df["model"].astype(str) + "__" + df["optimizer"].astype(str)
    df["rmse_rank"] = pd.to_numeric(df["rmse"], errors="coerce").rank(method="min", ascending=True)
    df["mae_rank"] = pd.to_numeric(df["mae"], errors="coerce").rank(method="min", ascending=True)
    df["r2_rank"] = pd.to_numeric(df["r2"], errors="coerce").rank(method="min", ascending=False)
    if "runtime_s" in df.columns:
        df["runtime_rank"] = pd.to_numeric(df["runtime_s"], errors="coerce").rank(method="min", ascending=True)
        rank_cols = ["rmse_rank", "mae_rank", "r2_rank", "runtime_rank"]
    else:
        rank_cols = ["rmse_rank", "mae_rank", "r2_rank"]
    df["overall_rank_score"] = df[rank_cols].mean(axis=1)
    df["overall_rank"] = df["overall_rank_score"].rank(method="min", ascending=True).astype(int)
    out_cols = ["overall_rank", "model", "optimizer", "method", "rmse", "mae", "r2"]
    if "runtime_s" in df.columns:
        out_cols.append("runtime_s")
    out_cols += rank_cols + ["overall_rank_score"]
    rank_df = df[out_cols].sort_values(["overall_rank", "rmse"]).reset_index(drop=True)
    path = os.path.join(out_dir, "rank_table.csv")
    rank_df.to_csv(path, index=False)
    rank_df.to_csv(os.path.join(out_dir, "critical_difference_rank_table.csv"), index=False)
    return path


def write_runtime_accuracy_improvement(metrics_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    if metrics_df is None or metrics_df.empty or not {"model", "optimizer", "rmse", "runtime_s"}.issubset(metrics_df.columns):
        return None
    rows: List[Dict[str, Any]] = []
    df = metrics_df.copy()
    for model, g in df.groupby("model", dropna=False):
        base = g[g["optimizer"].astype(str) == "None"]
        if base.empty:
            continue
        b = base.iloc[0]
        b_rmse = float(b.get("rmse", np.nan))
        b_mae = float(b.get("mae", np.nan)) if "mae" in b else np.nan
        b_r2 = float(b.get("r2", np.nan)) if "r2" in b else np.nan
        b_rt = float(b.get("runtime_s", np.nan))
        for _, r in g.iterrows():
            rmse = float(r.get("rmse", np.nan))
            mae = float(r.get("mae", np.nan)) if "mae" in r else np.nan
            r2 = float(r.get("r2", np.nan)) if "r2" in r else np.nan
            rt = float(r.get("runtime_s", np.nan))
            rows.append({
                "model": model,
                "optimizer": r.get("optimizer", ""),
                "baseline_optimizer": "None",
                "rmse": rmse,
                "baseline_rmse": b_rmse,
                "delta_rmse": rmse - b_rmse,
                "rmse_improvement": b_rmse - rmse,
                "rmse_improvement_pct": ((b_rmse - rmse) / b_rmse * 100.0) if np.isfinite(b_rmse) and b_rmse != 0 else np.nan,
                "mae": mae,
                "baseline_mae": b_mae,
                "delta_mae": mae - b_mae,
                "r2": r2,
                "baseline_r2": b_r2,
                "delta_r2": r2 - b_r2,
                "runtime_s": rt,
                "baseline_runtime_s": b_rt,
                "runtime_increase_s": rt - b_rt,
                "runtime_multiplier": (rt / b_rt) if np.isfinite(b_rt) and b_rt > 0 else np.nan,
            })
    if not rows:
        return None
    path = os.path.join(out_dir, "runtime_accuracy_improvement.csv")
    pd.DataFrame(rows).sort_values(["model", "rmse"]).to_csv(path, index=False)
    return path


def write_ablation_table(metrics_df: pd.DataFrame, out_dir: str, *, nested_cv: bool, time_series_cv: bool, with_shap: bool) -> Optional[str]:
    if metrics_df is None or metrics_df.empty or not {"model", "optimizer", "rmse"}.issubset(metrics_df.columns):
        return None
    rows: List[Dict[str, Any]] = []
    df = metrics_df.copy()
    for model, g in df.groupby("model", dropna=False):
        base = g[g["optimizer"].astype(str) == "None"]
        if not base.empty:
            b = base.sort_values("rmse").iloc[0]
            rows.append({
                "scope": "per_model",
                "setting": f"{model} default baseline",
                "model": model,
                "optimizer": "None",
                "time_series_cv": bool(time_series_cv),
                "nested_cv": bool(nested_cv),
                "hpo": False,
                "shap_generated": bool(with_shap),
                "rmse": b.get("rmse", np.nan),
                "mae": b.get("mae", np.nan),
                "r2": b.get("r2", np.nan),
                "runtime_s": b.get("runtime_s", np.nan),
                "baseline_rmse": b.get("rmse", np.nan),
                "rmse_improvement_vs_default": 0.0,
            })
        hpo = g[g["optimizer"].astype(str) != "None"]
        if not hpo.empty:
            best = hpo.sort_values("rmse").iloc[0]
            baseline_rmse = float(base.sort_values("rmse").iloc[0].get("rmse", np.nan)) if not base.empty else np.nan
            rows.append({
                "scope": "per_model",
                "setting": f"{model} best HPO",
                "model": model,
                "optimizer": best.get("optimizer", ""),
                "time_series_cv": bool(time_series_cv),
                "nested_cv": bool(nested_cv),
                "hpo": True,
                "shap_generated": bool(with_shap),
                "rmse": best.get("rmse", np.nan),
                "mae": best.get("mae", np.nan),
                "r2": best.get("r2", np.nan),
                "runtime_s": best.get("runtime_s", np.nan),
                "baseline_rmse": baseline_rmse,
                "rmse_improvement_vs_default": baseline_rmse - float(best.get("rmse", np.nan)) if np.isfinite(baseline_rmse) else np.nan,
            })
    best_all = df.sort_values("rmse").head(1)
    if not best_all.empty:
        r = best_all.iloc[0]
        rows.append({
            "scope": "overall",
            "setting": "overall best evaluated configuration",
            "model": r.get("model", ""),
            "optimizer": r.get("optimizer", ""),
            "time_series_cv": bool(time_series_cv),
            "nested_cv": bool(nested_cv),
            "hpo": str(r.get("optimizer", "")) != "None",
            "shap_generated": bool(with_shap),
            "rmse": r.get("rmse", np.nan),
            "mae": r.get("mae", np.nan),
            "r2": r.get("r2", np.nan),
            "runtime_s": r.get("runtime_s", np.nan),
            "baseline_rmse": np.nan,
            "rmse_improvement_vs_default": np.nan,
        })
    if not rows:
        return None
    path = os.path.join(out_dir, "ablation_table.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_actual_vs_predicted_plot(y_true: List[float] | np.ndarray, predictions: Dict[str, List[float]], metrics_df: pd.DataFrame, fig_dir: str) -> Optional[str]:
    if metrics_df is None or metrics_df.empty or not predictions:
        return None
    best = metrics_df.sort_values("rmse").iloc[0]
    key = _method_key(best["model"], best["optimizer"])
    pred = predictions.get(key)
    if pred is None:
        return None
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(pred, dtype=float)
    n = min(len(y), len(yp))
    if n < 10:
        return None
    y = y[:n]; yp = yp[:n]
    mask = np.isfinite(y) & np.isfinite(yp)
    if mask.sum() < 10:
        return None
    y_m = y[mask]; yp_m = yp[mask]
    lo = float(np.nanmin([np.nanmin(y_m), np.nanmin(yp_m)]))
    hi = float(np.nanmax([np.nanmax(y_m), np.nanmax(yp_m)]))
    plt.figure(figsize=(6.2, 6.0))
    plt.scatter(y_m, yp_m, s=8, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted — {key}")
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    specific = os.path.join(fig_dir, f"actual_vs_predicted_{_safe_name(key)}.png")
    generic = os.path.join(fig_dir, "actual_vs_predicted_best.png")
    plt.savefig(specific, dpi=220)
    plt.savefig(generic, dpi=220)
    plt.close()
    return specific


def write_fold_zone_heatmap(fold_tables: Dict[str, pd.DataFrame], metrics_df: pd.DataFrame, fig_dir: str, out_dir: str) -> Optional[str]:
    if metrics_df is None or metrics_df.empty or not fold_tables:
        return None
    best = metrics_df.sort_values("rmse").iloc[0]
    key = _method_key(best["model"], best["optimizer"])
    ft = fold_tables.get(key)
    if ft is None or ft.empty or not {"zone", "fold", "rmse"}.issubset(ft.columns):
        return None
    df = ft.copy()
    df["zone"] = df["zone"].astype(str)
    df["fold"] = df["fold"].astype(str)
    pivot = df.pivot_table(index="zone", columns="fold", values="rmse", aggfunc="mean")
    if pivot.empty:
        return None
    csv_path = os.path.join(out_dir, f"fold_zone_rmse_heatmap_{_safe_name(key)}.csv")
    pivot.to_csv(csv_path)
    plt.figure(figsize=(max(5, 1.1 * len(pivot.columns) + 2), max(4, 0.6 * len(pivot.index) + 2)))
    arr = pivot.to_numpy(dtype=float)
    im = plt.imshow(arr, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [f"Fold {c}" for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Outer fold")
    plt.ylabel("Zone")
    plt.title(f"Fold × Zone RMSE Heatmap — {key}")
    plt.colorbar(im, label="RMSE")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"fold_zone_rmse_heatmap_{_safe_name(key)}.png")
    plt.savefig(fig_path, dpi=220)
    plt.savefig(os.path.join(fig_dir, "fold_zone_rmse_heatmap_best.png"), dpi=220)
    plt.close()
    return fig_path


def write_paper_tables(metrics_df: pd.DataFrame, best_params_report: Dict[str, Dict[str, Any]], out_dir: str, *, nested_cv: bool, time_series_cv: bool, with_shap: bool) -> Dict[str, Optional[str]]:
    return {
        "best_hyperparameters": write_best_hyperparameters_table(best_params_report, out_dir),
        "rank_table": write_rank_table(metrics_df, out_dir),
        "runtime_accuracy_improvement": write_runtime_accuracy_improvement(metrics_df, out_dir),
        "ablation_table": write_ablation_table(metrics_df, out_dir, nested_cv=nested_cv, time_series_cv=time_series_cv, with_shap=with_shap),
    }


def write_paper_figures(y_true: List[float] | np.ndarray, predictions: Dict[str, List[float]], metrics_df: pd.DataFrame, fold_tables: Dict[str, pd.DataFrame], fig_dir: str, out_dir: str) -> Dict[str, Optional[str]]:
    return {
        "actual_vs_predicted": write_actual_vs_predicted_plot(y_true, predictions, metrics_df, fig_dir),
        "fold_zone_heatmap": write_fold_zone_heatmap(fold_tables, metrics_df, fig_dir, out_dir),
    }
