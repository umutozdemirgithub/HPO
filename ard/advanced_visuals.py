from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_") or "method"


def _save_note_figure(path: str, title: str, message: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(9, 3.8))
    plt.axis("off")
    plt.title(title, loc="left", fontsize=13, fontweight="bold")
    plt.text(0.02, 0.62, message, va="top", ha="left", wrap=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path




def _as_float_array(values: Any) -> np.ndarray:
    """Return a 1-D float ndarray for matplotlib, never a pandas Series."""
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _as_str_list(values: Any) -> List[str]:
    """Return plain Python strings for categorical matplotlib axes."""
    return [str(v) for v in list(values)]

def create_prediction_diagnostic_figures(
    *,
    fig_dir: str,
    out_dir: str,
    y: np.ndarray,
    predictions: Dict[str, List[float]],
    metrics_df: pd.DataFrame,
    fold_tables: Dict[str, pd.DataFrame],
    histories: Dict[str, List[float]],
    date_values: Optional[np.ndarray] = None,
    log: Optional[Any] = None,
) -> Dict[str, List[str]]:
    """Generate paper-ready prediction, residual, optimizer, and fold-stability figures.

    This function intentionally uses matplotlib only, so PNG export does not require
    Plotly/Kaleido. It writes every figure to ``fig_dir`` and returns categorized paths.
    """
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    files: Dict[str, List[str]] = {"prediction": [], "residuals": [], "optimization": [], "folds": [], "metrics": []}

    def _log(msg: str) -> None:
        if log:
            log(msg)

    try:
        y_arr = np.asarray(y, dtype=float).reshape(-1)

        if metrics_df is not None and not metrics_df.empty:
            metric_candidates = [c for c in ["rmse", "mae", "r2", "mape", "nrmse", "runtime_s"] if c in metrics_df.columns]
            if metric_candidates and {"model", "optimizer"}.issubset(metrics_df.columns):
                view = metrics_df.copy().reset_index(drop=True)
                labels = [f"{r['model']}\n{r['optimizer']}" for _, r in view.iterrows()]
                mat_cols = []
                for col in metric_candidates:
                    vals = pd.to_numeric(view[col], errors="coerce").to_numpy(dtype=float)
                    finite = np.isfinite(vals)
                    if not finite.any():
                        continue
                    mn, mx = float(np.nanmin(vals)), float(np.nanmax(vals))
                    scaled = np.zeros_like(vals, dtype=float) if mx == mn else (vals - mn) / (mx - mn)
                    if col.lower() == "r2":
                        scaled = 1.0 - scaled
                    mat_cols.append((col, scaled))
                if mat_cols:
                    mat = np.vstack([v for _, v in mat_cols]).T
                    plt.figure(figsize=(max(7.5, 0.65 * len(metric_candidates) + 5), max(5.0, 0.45 * len(labels) + 1.8)))
                    im = plt.imshow(mat, aspect="auto", cmap="viridis_r")
                    plt.colorbar(im, label="Normalized score (lower is better)")
                    plt.xticks(np.arange(len(mat_cols)), [c for c, _ in mat_cols], rotation=35, ha="right")
                    plt.yticks(np.arange(len(labels)), labels)
                    plt.title("Metric performance heatmap")
                    plt.tight_layout()
                    p = os.path.join(fig_dir, "metric_performance_heatmap.png")
                    plt.savefig(p, dpi=260, bbox_inches="tight")
                    plt.close()
                    files["metrics"].append(p)

            if {"model", "optimizer"}.issubset(metrics_df.columns) and "rmse" in metrics_df.columns:
                top = metrics_df.sort_values("rmse").head(min(15, len(metrics_df))).copy()
                top["method"] = [f"{r['model']}__{r['optimizer']}" for _, r in top.iterrows()]
                top = top.iloc[::-1]
                plt.figure(figsize=(10, max(5.2, 0.38 * len(top) + 1.8)))
                plt.barh(_as_str_list(top["method"]), _as_float_array(top["rmse"]), alpha=0.9, label="RMSE")
                if "mae" in top.columns:
                    plt.barh(_as_str_list(top["method"]), _as_float_array(top["mae"]), alpha=0.55, label="MAE")
                    plt.legend()
                plt.xlabel("Error")
                plt.title("Top methods error bar")
                plt.tight_layout()
                p = os.path.join(fig_dir, "top_methods_error_bar.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["metrics"].append(p)

        if predictions:
            # Generate diagnostics for all methods (capped for readability if huge).
            method_order = list(predictions.keys())
            if metrics_df is not None and not metrics_df.empty and {"model", "optimizer", "rmse"}.issubset(metrics_df.columns):
                ranked = [f"{r['model']}__{r['optimizer']}" for _, r in metrics_df.sort_values("rmse").iterrows()]
                method_order = [m for m in ranked if m in predictions] + [m for m in method_order if m not in ranked]
            for method_key in method_order[: max(1, min(20, len(method_order)))]:
                pred_arr = np.asarray(predictions[method_key], dtype=float).reshape(-1)
                n = min(len(y_arr), len(pred_arr))
                if n < 5:
                    continue
                yt = y_arr[:n]
                yp = pred_arr[:n]
                mask = np.isfinite(yt) & np.isfinite(yp)
                if mask.sum() < 5:
                    continue
                yt_m, yp_m = yt[mask], yp[mask]
                resid = yt_m - yp_m
                abs_err = np.abs(resid)
                safe = _safe_name(method_key)
                lo = float(np.nanmin([yt_m.min(), yp_m.min()]))
                hi = float(np.nanmax([yt_m.max(), yp_m.max()]))

                plt.figure(figsize=(6.4, 6.2))
                plt.scatter(yt_m, yp_m, s=14, alpha=0.42)
                plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="Ideal")
                try:
                    coef = np.polyfit(yt_m, yp_m, 1)
                    xline = np.linspace(lo, hi, 100)
                    plt.plot(xline, coef[0] * xline + coef[1], linewidth=1.1, label="Linear fit")
                    plt.legend(fontsize=8)
                except Exception:
                    pass
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"Actual vs predicted — {method_key}")
                plt.tight_layout()
                p = os.path.join(fig_dir, f"actual_vs_predicted_scatter_{safe}.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["prediction"].append(p)

                pd.DataFrame({"y_true": yt_m, "y_pred": yp_m, "residual": resid, "abs_error": abs_err}).to_csv(
                    os.path.join(out_dir, f"prediction_residual_diagnostics_{safe}.csv"), index=False
                )

                q = np.linspace(0, 1, len(abs_err))
                plt.figure(figsize=(7.2, 4.5))
                plt.plot(q, np.sort(abs_err), linewidth=1.4)
                plt.xlabel("Quantile")
                plt.ylabel("Absolute error")
                plt.title(f"Absolute error quantile curve — {method_key}")
                plt.tight_layout()
                p = os.path.join(fig_dir, f"absolute_error_quantile_{safe}.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["residuals"].append(p)

                q_theory = np.linspace(0.01, 0.99, len(resid))
                try:
                    from scipy.stats import norm
                    theoretical = norm.ppf(q_theory)
                except Exception:
                    theoretical = np.linspace(-2.33, 2.33, len(resid))
                sample = np.sort((resid - np.mean(resid)) / (np.std(resid) if np.std(resid) > 0 else 1.0))
                plt.figure(figsize=(5.8, 5.6))
                plt.scatter(theoretical, sample, s=12, alpha=0.48)
                loq, hiq = float(min(theoretical.min(), sample.min())), float(max(theoretical.max(), sample.max()))
                plt.plot([loq, hiq], [loq, hiq], linestyle="--", linewidth=1.0)
                plt.xlabel("Theoretical normal quantile")
                plt.ylabel("Standardized residual quantile")
                plt.title(f"Residual QQ plot — {method_key}")
                plt.tight_layout()
                p = os.path.join(fig_dir, f"residual_qq_{safe}.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["residuals"].append(p)

                x = np.arange(len(abs_err))
                xlabel = "OOF sample order"
                if date_values is not None and len(date_values) >= n:
                    try:
                        x = np.asarray(date_values[:n])[mask]
                        xlabel = "Time"
                    except Exception:
                        x = np.arange(len(abs_err))
                plt.figure(figsize=(10, 3.8))
                plt.plot(x, abs_err, linewidth=0.9)
                plt.xlabel(xlabel)
                plt.ylabel("Absolute error")
                plt.title(f"Absolute error over time/order — {method_key}")
                plt.tight_layout()
                p = os.path.join(fig_dir, f"absolute_error_over_time_{safe}.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["residuals"].append(p)

        if histories:
            valid = {k: np.asarray(v, dtype=float) for k, v in histories.items() if v is not None and len(v) >= 2}
            if valid:
                plt.figure(figsize=(9.5, 5.4))
                for k, vals in valid.items():
                    plt.plot(np.arange(1, len(vals) + 1), vals, linewidth=1.2, label=k)
                plt.xlabel("Iteration")
                plt.ylabel("Best RMSE so far")
                plt.title("Optimizer convergence overlay")
                plt.legend(fontsize=7, ncol=2)
                plt.tight_layout()
                p = os.path.join(fig_dir, "optimizer_convergence_overlay.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["optimization"].append(p)
                rows = []
                for k, vals in valid.items():
                    rows.append({"method": k, "first": float(vals[0]), "last": float(vals[-1]), "best": float(np.nanmin(vals)), "auc": float(np.trapz(vals))})
                pd.DataFrame(rows).sort_values("best").to_csv(os.path.join(out_dir, "optimizer_convergence_summary.csv"), index=False)
            else:
                files["optimization"].append(_save_note_figure(
                    os.path.join(fig_dir, "optimizer_convergence_overlay_not_available.png"),
                    "Optimizer convergence overlay not available",
                    "No iteration-level optimizer history was produced. This is expected when only the 'None' optimizer is selected or when the selected optimizer does not expose per-iteration history.",
                ))

        if fold_tables:
            rows = []
            for key, ft in fold_tables.items():
                if isinstance(ft, pd.DataFrame) and {"fold", "rmse"}.issubset(ft.columns):
                    for _, r in ft.iterrows():
                        try:
                            rows.append({"method": key, "fold": int(r["fold"]), "rmse": float(r["rmse"])})
                        except Exception:
                            pass
            if rows:
                fdf = pd.DataFrame(rows)
                heat = fdf.pivot_table(index="method", columns="fold", values="rmse", aggfunc="mean")
                plt.figure(figsize=(max(7.5, 1.0 * heat.shape[1] + 4), max(5.0, 0.35 * heat.shape[0] + 1.8)))
                im = plt.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="magma_r")
                plt.colorbar(im, label="RMSE")
                plt.xticks(np.arange(heat.shape[1]), [str(c) for c in heat.columns])
                plt.yticks(np.arange(heat.shape[0]), heat.index.tolist())
                plt.xlabel("Fold")
                plt.ylabel("Method")
                plt.title("Fold RMSE heatmap")
                plt.tight_layout()
                p = os.path.join(fig_dir, "fold_rmse_heatmap.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["folds"].append(p)
    except Exception as exc:
        _log(f"Advanced diagnostic figures skipped: {type(exc).__name__}: {exc}")
    return files


def create_multi_seed_figures(*, per_seed_metrics: pd.DataFrame, summary: pd.DataFrame, fig_dir: str, out_dir: str) -> Dict[str, List[str]]:
    """Generate figures describing seed-to-seed stability."""
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    files: Dict[str, List[str]] = {"multi_seed": []}
    if per_seed_metrics is None or per_seed_metrics.empty:
        files["multi_seed"].append(_save_note_figure(
            os.path.join(fig_dir, "multi_seed_not_available.png"),
            "Multi-seed stability not available",
            "No per-seed metrics were available. Enable multi-seed evaluation and provide at least two seeds to generate seed stability figures.",
        ))
        return files
    try:
        df = per_seed_metrics.copy()
        if {"model", "optimizer"}.issubset(df.columns):
            df["method"] = df["model"].astype(str) + "__" + df["optimizer"].astype(str)
        elif "method" not in df.columns:
            return files

        n_seeds = int(df["seed"].nunique()) if "seed" in df.columns else 0
        if n_seeds < 2:
            files["multi_seed"].append(_save_note_figure(
                os.path.join(fig_dir, "multi_seed_not_available.png"),
                "Multi-seed stability not available for this run",
                "This run contains only one seed. Enable multi-seed evaluation and use a seed list such as 42, 101, 202 to generate RMSE stability, mean ± std ranking, and seed sensitivity figures.",
            ))
            return files

        if {"seed", "rmse"}.issubset(df.columns):
            top_methods = df.groupby("method")["rmse"].mean().sort_values().head(12).index.tolist()
            view = df[df["method"].isin(top_methods)].copy()
            plt.figure(figsize=(10, max(5.2, 0.34 * len(top_methods) + 2)))
            for method, grp in view.groupby("method"):
                grp = grp.sort_values("seed")
                plt.plot(_as_float_array(grp["seed"]), _as_float_array(grp["rmse"]), marker="o", linewidth=1.1, label=method)
            plt.xlabel("Seed")
            plt.ylabel("RMSE")
            plt.title("Multi-seed RMSE stability")
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            p = os.path.join(fig_dir, "multi_seed_rmse_lines.png")
            plt.savefig(p, dpi=260, bbox_inches="tight")
            plt.close()
            files["multi_seed"].append(p)

            box_data = [view.loc[view["method"] == m, "rmse"].to_numpy(dtype=float) for m in top_methods]
            plt.figure(figsize=(max(8.5, 0.55 * len(top_methods) + 3), 5.0))
            plt.boxplot(box_data, tick_labels=top_methods, showmeans=True)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("RMSE")
            plt.title("Multi-seed RMSE distribution")
            plt.tight_layout()
            p = os.path.join(fig_dir, "multi_seed_rmse_boxplot.png")
            plt.savefig(p, dpi=260, bbox_inches="tight")
            plt.close()
            files["multi_seed"].append(p)

        if summary is not None and not summary.empty and {"model", "optimizer"}.issubset(summary.columns):
            s = summary.copy()
            s["method"] = s["model"].astype(str) + "__" + s["optimizer"].astype(str)
            mean_col = "rmse_mean" if "rmse_mean" in s.columns else ("RMSE" if "RMSE" in s.columns else None)
            std_col = "rmse_std" if "rmse_std" in s.columns else ("RMSE_std" if "RMSE_std" in s.columns else None)
            if mean_col:
                top = s.sort_values(mean_col).head(min(15, len(s))).iloc[::-1]
                xerr = pd.to_numeric(top[std_col], errors="coerce").to_numpy(dtype=float) if std_col in top.columns else None
                plt.figure(figsize=(10, max(5.0, 0.38 * len(top) + 1.8)))
                plt.barh(top["method"].astype(str).to_list(), pd.to_numeric(top[mean_col], errors="coerce").to_numpy(dtype=float), xerr=xerr, alpha=0.9)
                plt.xlabel("RMSE mean ± std" if xerr is not None else "RMSE mean")
                plt.title("Multi-seed mean ± std ranking")
                plt.tight_layout()
                p = os.path.join(fig_dir, "multi_seed_mean_std_ranking.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["multi_seed"].append(p)

            if {"rmse_mean", "rmse_std"}.issubset(s.columns):
                s["rmse_cv"] = s["rmse_std"] / s["rmse_mean"].replace(0, np.nan)
                s[["model", "optimizer", "method", "rmse_cv"]].to_csv(os.path.join(out_dir, "multi_seed_rmse_cv.csv"), index=False)
                top_cv = s.sort_values("rmse_cv", ascending=False).head(min(15, len(s))).iloc[::-1]
                plt.figure(figsize=(10, max(5.0, 0.38 * len(top_cv) + 1.8)))
                plt.barh(top_cv["method"].astype(str).to_list(), pd.to_numeric(top_cv["rmse_cv"], errors="coerce").to_numpy(dtype=float), alpha=0.9)
                plt.xlabel("Coefficient of variation (RMSE std / mean)")
                plt.title("Seed sensitivity / coefficient of variation")
                plt.tight_layout()
                p = os.path.join(fig_dir, "multi_seed_seed_sensitivity_cv.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                files["multi_seed"].append(p)
    except Exception:
        pass
    return files


__all__ = ["create_prediction_diagnostic_figures", "create_multi_seed_figures"]
