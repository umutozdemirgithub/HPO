from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .wilcoxon import (
    _HAS_SCIPY_STATS,
    _holm_bonferroni_adjust,
    _run_wilcoxon_from_differences,
)

def _pareto_front(rmse_vals: np.ndarray, runtime_vals: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points for two minimisation objectives."""
    rmse_arr = np.asarray(rmse_vals, dtype=float)
    runtime_arr = np.asarray(runtime_vals, dtype=float)
    n = len(rmse_arr)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                rmse_arr[j] <= rmse_arr[i]
                and runtime_arr[j] <= runtime_arr[i]
                and (rmse_arr[j] < rmse_arr[i] or runtime_arr[j] < runtime_arr[i])
            ):
                is_pareto[i] = False
                break
    return is_pareto


def _ordered_pareto_frontier(rmse_vals: np.ndarray, runtime_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frontier indices and coordinates ordered by runtime.

    The returned frontier contains only non-dominated points, sorted by runtime,
    which makes it suitable for plotting the actual Pareto frontier line.
    """
    rmse_arr = np.asarray(rmse_vals, dtype=float)
    runtime_arr = np.asarray(runtime_vals, dtype=float)
    mask = _pareto_front(rmse_arr, runtime_arr)
    frontier_idx = np.flatnonzero(mask)
    if frontier_idx.size == 0:
        return frontier_idx, runtime_arr[frontier_idx], rmse_arr[frontier_idx]
    order = np.argsort(runtime_arr[frontier_idx], kind="mergesort")
    frontier_idx = frontier_idx[order]
    return frontier_idx, runtime_arr[frontier_idx], rmse_arr[frontier_idx]


def _generate_standard_figures(
    fig_dir: str,
    out_dir: str,
    df2: pd.DataFrame,
    y: np.ndarray,
    target_cols: List[str],
    feature_cols: List[str],
    date_column: Optional[str],
    metrics_df: pd.DataFrame,
    fold_tables: Dict[str, pd.DataFrame],
    histories: Dict[str, List[float]],
    predictions: Dict[str, List[float]],
    log,
) -> None:

    # Convergence plots
    for key, hist in histories.items():
        fig_c = plt.figure()
        try:
            plt.plot(hist)
            plt.xlabel("iteration"); plt.ylabel("best RMSE")
            plt.title(f"Convergence: {key}")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"convergence_{key}.png"), dpi=200)
        finally:
            plt.close(fig_c)

    try:
        x_ts = df2[date_column] if (date_column and date_column in df2.columns) else np.arange(len(y))

        fig_ts = plt.figure()
        try:
            plt.plot(x_ts, y, linewidth=0.8)
            plt.xlabel("time" if (date_column and date_column in df2.columns) else "index")
            plt.ylabel(target_cols[0])
            plt.title(f"{target_cols[0]} time series")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "timeseries_power.png"), dpi=200)
        finally:
            plt.close(fig_ts)

        if "SSRD" in feature_cols and "SSRD" in df2.columns:
            plt.figure()
            plt.scatter(df2["SSRD"].to_numpy(), y, s=4, alpha=0.3)
            plt.xlabel("SSRD"); plt.ylabel(target_cols[0])
            plt.title("SSRD vs target")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "ssrd_scatter.png"), dpi=200)
            plt.close()

        cols_for_corr = [c for c in feature_cols if c in df2.columns]
        if cols_for_corr:
            corr_df = df2[cols_for_corr + [target_cols[0]]].corr(numeric_only=True)
            plt.figure(figsize=(8, 6))
            plt.imshow(corr_df, aspect="auto")
            plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
            plt.yticks(range(len(corr_df.index)), corr_df.index)
            plt.colorbar(); plt.title("Correlation heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"), dpi=200)
            plt.close()

        if len(metrics_df) > 0:
            rt = metrics_df["runtime_s"].to_numpy(dtype=float)
            rm = metrics_df["rmse"].to_numpy(dtype=float)
            frontier_idx, frontier_rt, frontier_rm = _ordered_pareto_frontier(rm, rt)
            pareto_mask = np.zeros(len(metrics_df), dtype=bool)
            pareto_mask[frontier_idx] = True

            pareto_df = metrics_df.copy()
            pareto_df["is_pareto_optimal"] = pareto_mask
            pareto_df.to_csv(os.path.join(out_dir, "pareto_frontier_points.csv"), index=False)

            fig_p, ax_p = plt.subplots()
            try:
                ax_p.scatter(rt[~pareto_mask],rm[~pareto_mask], color="#AAAAAA", zorder=2, s=45, label="Dominated points",)
                ax_p.scatter(frontier_rt, frontier_rm,color="#1B5E20", zorder=3, s=70, marker="D",label="Pareto-optimal points",)
                if frontier_idx.size >= 2:
                    ax_p.step(frontier_rt,frontier_rm, where="post",color="#1B5E20",linewidth=1.5, zorder=2.5,label="Pareto frontier",)
                for _, r in metrics_df.iterrows():
                    ax_p.annotate(f"{r['model']}/{r['optimizer']}",(r["runtime_s"], r["rmse"]),fontsize=6.5, xytext=(4, 2), textcoords="offset points",)
                ax_p.set_xlabel("Runtime (s)")
                ax_p.set_ylabel("RMSE")
                ax_p.set_title("Runtime vs RMSE — Pareto frontier")
                ax_p.legend(fontsize=8)
                ax_p.set_xscale("symlog", linthresh=1.0)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "rmse_runtime.png"), dpi=200)
                plt.savefig(os.path.join(fig_dir, "pareto_frontier.png"), dpi=200)
            finally:
                plt.close(fig_p)

        if fold_tables:
            keys = list(fold_tables.keys())
            rmse_lists = [fold_tables[k]["rmse"].tolist() for k in keys]
            plt.figure(figsize=(max(8, len(keys) * 0.7), 4))
            plt.boxplot(rmse_lists, labels=keys, vert=True)
            plt.xticks(rotation=45, ha="right"); plt.ylabel("RMSE")
            plt.title("Fold stability (RMSE distribution)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "fold_stability_rmse_box.png"), dpi=200)
            plt.close()

        # Prediction vs time (OOF) — uses date column, not index
        try:
            x_time = df2[date_column] if (date_column and date_column in df2.columns) else np.arange(len(y))
            for method_key, oof in predictions.items():
                y_pred = np.asarray(oof, dtype=float).reshape(-1)
                n = min(len(y), len(y_pred))
                yt = np.asarray(y[:n], dtype=float)
                xp = np.asarray(x_time[:n])
                mask = (~np.isnan(yt)) & (~np.isnan(y_pred))
                if mask.sum() < 10:
                    continue
                plt.figure(figsize=(10, 3.5))
                plt.plot(xp[mask], yt[mask], label="Actual", linewidth=1.0)
                plt.plot(xp[mask], y_pred[mask], label="OOF prediction", linewidth=1.0)
                plt.xlabel("Time" if (date_column and date_column in df2.columns) else "Index")
                plt.ylabel(target_cols[0])
                plt.title(f"Prediction vs Time (OOF) — {method_key}")
                plt.legend(); plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"prediction_vs_time_{method_key}.png"), dpi=220)
                plt.close()
        except Exception as _e:
            log(f"Diagnostics (prediction-vs-time) skipped: {_e}")

        # Residual analysis for the best method
        try:
            if len(metrics_df) > 0:
                best_row = metrics_df.sort_values("rmse").iloc[0]
                best_key = f"{best_row['model']}__{best_row['optimizer']}"
                oof = predictions.get(best_key)
                if oof is not None:
                    y_pred = np.asarray(oof, dtype=float).reshape(-1)
                    n = min(len(y), len(y_pred))
                    yt = np.asarray(y[:n], dtype=float)
                    yp = np.asarray(y_pred[:n], dtype=float)
                    mask = (~np.isnan(yt)) & (~np.isnan(yp))
                    if mask.sum() >= 10:
                        resid = yt[mask] - yp[mask]
                        plt.figure(figsize=(6.5, 4.0))
                        plt.hist(resid, bins=40)
                        plt.xlabel("Residual (Actual − Predicted)"); plt.ylabel("Count")
                        plt.title(f"Residual Histogram — {best_key}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_dir, f"residual_hist_{best_key}.png"), dpi=220)
                        plt.close()

                        plt.figure(figsize=(6.5, 4.0))
                        plt.scatter(yp[mask], resid, s=10, alpha=0.3)
                        plt.axhline(0.0, linestyle="--", linewidth=1.0)
                        plt.xlabel("Predicted"); plt.ylabel("Residual (Actual − Predicted)")
                        plt.title(f"Residuals vs Predicted — {best_key}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_dir, f"residual_vs_pred_{best_key}.png"), dpi=220)
                        plt.close()
        except Exception as _e:
            log(f"Diagnostics (residuals) skipped: {_e}")

        # Wilcoxon signed-rank test
        try:
            if _HAS_SCIPY_STATS and fold_tables:
                w_rows = []
                detail_rows = []
                models_present = sorted({k.split("__", 1)[0] for k in fold_tables.keys() if "__" in k})
                for mn in models_present:
                    base_key = f"{mn}__None"
                    if base_key not in fold_tables:
                        continue
                    base_df = fold_tables[base_key].copy()
                    if "rmse" not in base_df.columns:
                        continue
                    base_rmse = pd.to_numeric(base_df["rmse"], errors="coerce").to_numpy(dtype=float)
                    model_rows = []
                    diffs_by_opt = []
                    labels = []
                    for mk, ft in fold_tables.items():
                        if not mk.startswith(mn + "__"):
                            continue
                        opt = mk.split("__", 1)[1]
                        if opt == "None" or "rmse" not in ft.columns:
                            continue
                        cur_rmse = pd.to_numeric(ft["rmse"], errors="coerce").to_numpy(dtype=float)
                        n_pairs = min(len(base_rmse), len(cur_rmse))
                        if n_pairs < 2:
                            continue
                        paired = pd.DataFrame({"fold": np.arange(1, n_pairs + 1, dtype=int),"baseline_rmse": base_rmse[:n_pairs],"optimizer_rmse": cur_rmse[:n_pairs],})
                        paired["improvement"] = paired["baseline_rmse"] - paired["optimizer_rmse"]
                        paired = paired[np.isfinite(paired["baseline_rmse"]) & np.isfinite(paired["optimizer_rmse"])].reset_index(drop=True)
                        if len(paired) < 2:
                            continue
                        stats_row = _run_wilcoxon_from_differences(paired["improvement"].to_numpy(dtype=float))
                        row = {"model": mn,"optimizer": opt,"pair_scope": "outer_fold",**stats_row,}
                        w_rows.append(row)
                        model_rows.append(row)
                        diffs_by_opt.append(paired["improvement"].to_numpy(dtype=float))
                        labels.append(opt)
                        for _, prow in paired.iterrows():
                            detail_rows.append({
                                "model": mn,
                                "optimizer": opt,
                                "pair_scope": "outer_fold",
                                "fold": int(prow["fold"]),
                                "pair_id": f"fold_{int(prow['fold'])}",
                                "baseline_rmse": float(prow["baseline_rmse"]),
                                "optimizer_rmse": float(prow["optimizer_rmse"]),
                                "improvement": float(prow["improvement"]),
                            })
                    if model_rows:
                        adj = _holm_bonferroni_adjust([r.get("p_value", np.nan) for r in model_rows])
                        for row, adj_p in zip(model_rows, adj):
                            row["p_value_holm"] = float(adj_p) if np.isfinite(adj_p) else float("nan")
                            row["significant_0_05"] = bool(np.isfinite(adj_p) and adj_p <= 0.05)

                    if diffs_by_opt:
                        plt.figure(figsize=(max(7.5, 1.7 * len(labels)), 4.8))
                        bp = plt.boxplot(diffs_by_opt, tick_labels=labels, patch_artist=True, showmeans=True)
                        for patch in bp.get('boxes', []):
                            patch.set(facecolor='#DCE8FF', edgecolor='#294399', linewidth=1.2)
                        for med in bp.get('medians', []):
                            med.set(color='#C62828', linewidth=1.5)
                        plt.axhline(0.0, linestyle='--', linewidth=1.0, color='#6B7280')
                        plt.ylabel('RMSE improvement vs None (positive is better)')
                        plt.title(f'Wilcoxon signed-rank distribution — {mn}')
                        plt.xticks(rotation=28, ha='right')
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_dir, f'wilcoxon_rmse_diffs_{mn}.png'), dpi=220)
                        plt.close()

                        model_df = pd.DataFrame(model_rows).sort_values('mean_improvement', ascending=False)
                        if not model_df.empty:
                            y_pos = np.arange(len(model_df))
                            plt.figure(figsize=(8.6, max(4.6, 0.7 * len(model_df) + 1.8)))
                            colors = ['#1B5E20' if v > 0 else '#B71C1C' for v in model_df['mean_improvement']]
                            mean_imp = pd.to_numeric(model_df['mean_improvement'], errors='coerce').to_numpy(dtype=float)
                            ci_low = pd.to_numeric(model_df['ci_low'], errors='coerce').to_numpy(dtype=float)
                            ci_high = pd.to_numeric(model_df['ci_high'], errors='coerce').to_numpy(dtype=float)
                            xerr = np.vstack([mean_imp - ci_low, ci_high - mean_imp])
                            plt.barh(y_pos, mean_imp, xerr=xerr, color=colors, alpha=0.9)
                            plt.yticks(y_pos, model_df['optimizer'])
                            plt.axvline(0.0, linestyle='--', linewidth=1.0, color='#6B7280')
                            plt.xlabel('Mean RMSE improvement vs None')
                            plt.title(f'Effect magnitude with 95% bootstrap CI — {mn}')
                            plt.tight_layout()
                            plt.savefig(os.path.join(fig_dir, f'wilcoxon_effect_sizes_{mn}.png'), dpi=220)
                            plt.close()

                            p_col = 'p_value_holm' if 'p_value_holm' in model_df.columns else 'p_value'
                            plt.figure(figsize=(8.6, max(4.6, 0.7 * len(model_df) + 1.8)))
                            sig_threshold = 0.05
                            colors = ['#0D47A1' if (np.isfinite(v) and v <= sig_threshold) else '#90A4AE' for v in model_df[p_col]]
                            plt.barh(y_pos, pd.to_numeric(model_df[p_col], errors='coerce').to_numpy(dtype=float), color=colors, alpha=0.95)
                            plt.yticks(y_pos, model_df['optimizer'])
                            plt.axvline(sig_threshold, linestyle='--', linewidth=1.0, color='#C62828')
                            plt.xlabel('Holm-adjusted Wilcoxon p-value' if p_col == 'p_value_holm' else 'Wilcoxon p-value')
                            plt.title(f'Statistical significance profile — {mn}')
                            plt.tight_layout()
                            plt.savefig(os.path.join(fig_dir, f'wilcoxon_pvalues_{mn}.png'), dpi=220)
                            plt.close()

                    model_detail_df = pd.DataFrame([r for r in detail_rows if r['model'] == mn])
                    if not model_detail_df.empty:
                        heat = model_detail_df.pivot(index='optimizer', columns='fold', values='improvement').sort_index()
                        plt.figure(figsize=(max(7.2, 1.0 * heat.shape[1] + 3.0), max(4.8, 0.7 * heat.shape[0] + 1.8)))
                        vmax = np.nanmax(np.abs(heat.to_numpy(dtype=float))) if heat.size else 1.0
                        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
                        im = plt.imshow(heat.to_numpy(dtype=float), aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
                        plt.colorbar(im, label='RMSE improvement vs None')
                        plt.xticks(np.arange(heat.shape[1]), [str(c) for c in heat.columns])
                        plt.yticks(np.arange(heat.shape[0]), heat.index.tolist())
                        plt.xlabel('Fold')
                        plt.ylabel('Optimizer')
                        plt.title(f'Fold-level improvement heatmap — {mn}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_dir, f'wilcoxon_fold_heatmap_{mn}.png'), dpi=220)
                        plt.close()
                if w_rows:
                    pd.DataFrame(w_rows).sort_values(["model", "p_value_holm", "p_value", "mean_improvement"], ascending=[True, True, True, False]).to_csv(
                        os.path.join(out_dir, "wilcoxon_results.csv"), index=False
                    )
                if detail_rows:
                    pd.DataFrame(detail_rows).to_csv(os.path.join(out_dir, "wilcoxon_fold_details.csv"), index=False)
        except Exception as _e:
            log(f"Diagnostics (wilcoxon) skipped: {_e}")

        # Ablation plots
        try:
            for method_key, hist in histories.items():
                if not hist or len(hist) < 3:
                    continue
                y_hist = np.asarray(hist, dtype=float)
                x_hist = np.arange(len(y_hist))
                plt.figure(figsize=(6.5, 4.0))
                plt.plot(x_hist, y_hist, marker="o", linewidth=1.0)
                plt.xlabel("Iteration"); plt.ylabel("Best RMSE so far")
                plt.title(f"Iteration-budget Ablation — {method_key}")
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"ablation_iters_{method_key}.png"), dpi=220)
                plt.close()
        except Exception as _e:
            log(f"Diagnostics (ablation) skipped: {_e}")

    except Exception as e:
        log(f"Visual analytics figure generation skipped: {e}")


