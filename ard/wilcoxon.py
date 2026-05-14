from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_IMPORT_ERRORS: Dict[str, str] = {}

try:
    from scipy.stats import wilcoxon  # type: ignore
    _HAS_SCIPY_STATS = True
except Exception as e:
    wilcoxon = None
    _HAS_SCIPY_STATS = False
    _IMPORT_ERRORS["scipy.stats"] = str(e)

def _bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        v = float(arr[0])
        return (v, v)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_means[i] = float(np.mean(sample))
    q_low = float(np.quantile(boot_means, alpha / 2.0))
    q_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return q_low, q_high

def _holm_bonferroni_adjust(p_values: List[float]) -> List[float]:
    vals = np.asarray([float(p) if p is not None else np.nan for p in p_values], dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(vals))[0]
    if finite_idx.size == 0:
        return out.tolist()
    order = finite_idx[np.argsort(vals[finite_idx])]
    m = len(order)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * vals[idx]
        adjusted = min(1.0, max(adjusted, running_max))
        out[idx] = adjusted
        running_max = adjusted
    return out.tolist()

def _run_wilcoxon_from_differences(diffs: np.ndarray) -> Dict[str, float]:
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n_pairs = int(d.size)
    n_nonzero = int(np.sum(~np.isclose(d, 0.0)))
    wins = int(np.sum(d > 0))
    losses = int(np.sum(d < 0))
    ties = int(np.sum(np.isclose(d, 0.0)))
    mean_imp = float(np.mean(d)) if n_pairs else float("nan")
    median_imp = float(np.median(d)) if n_pairs else float("nan")
    std_imp = float(np.std(d, ddof=1)) if n_pairs > 1 else float(0.0)
    stat = float("nan")
    p_value = float("nan")
    if _HAS_SCIPY_STATS and n_nonzero >= 2:
        try:
            w_res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            stat = float(getattr(w_res, "statistic", np.nan))
            p_value = float(getattr(w_res, "pvalue", np.nan))
        except Exception:
            stat = float("nan")
            p_value = float("nan")
    ci_low, ci_high = _bootstrap_mean_ci(d) if n_pairs else (float("nan"), float("nan"))
    effect_r = _wilcoxon_effect_size_from_stat(stat, n_pairs)
    return {
        "n_pairs": n_pairs,
        "n_nonzero": n_nonzero,
        "wilcoxon_stat": stat,
        "p_value": p_value,
        "mean_improvement": mean_imp,
        "median_improvement": median_imp,
        "std_improvement": std_imp,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": float(wins / n_pairs) if n_pairs else float("nan"),
        "effect_size_r": effect_r,
    }


def _sanitize_zone_value(zone_value: Any) -> str:
    raw = str(zone_value) if zone_value == zone_value else 'missing'
    cleaned = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in raw).strip('_')
    return cleaned or 'zone'

def _wilcoxon_effect_size_from_stat(stat: float, n: int) -> float:
    if n < 1 or not math.isfinite(float(stat)):
        return float("nan")
    denom = math.sqrt(float(n)) * ((float(n) + 1.0) / 2.0)
    if denom <= 0 or not math.isfinite(denom):
        return float("nan")
    return float(abs(float(stat)) / denom)

def _validate_selection_inputs(
    models_selected: List[str],
    optimizers_selected: List[str],
    available_models: Dict[str, Any],
    opt_map: Dict[str, Any],
) -> None:
    unknown_models = [m for m in models_selected if m not in available_models]
    if unknown_models:
        raise ValueError(f"Unknown model(s): {unknown_models}. Available models: {sorted(available_models.keys())}")
    unknown_optimizers = [o for o in optimizers_selected if o not in opt_map]
    if unknown_optimizers:
        raise ValueError(f"Unknown optimizer(s): {unknown_optimizers}. Available optimizers: {sorted(opt_map.keys())}")

def _aggregate_zone_wilcoxon(zone_results: List[Dict[str, Any]], out_dir: str, log) -> None:
    if not _HAS_SCIPY_STATS or not zone_results:
        return
    rows = []
    detail_rows = []
    by_zone_rows = []
    by_zone_detail_rows = []
    baseline_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    compare_map: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for zres in zone_results:
        zone_label = str(zres.get('zone_value'))
        for key, ft in (zres.get('fold_tables', {}) or {}).items():
            if '__' not in key or ft is None or len(ft) == 0 or 'rmse' not in ft.columns:
                continue
            model_name, optimizer_name = key.split('__', 1)
            ft2 = ft.copy()
            ft2 = ft2.reset_index(drop=True)
            ft2['fold'] = np.arange(1, len(ft2) + 1, dtype=int)
            ft2['rmse'] = pd.to_numeric(ft2['rmse'], errors='coerce')
            ft2 = ft2[np.isfinite(ft2['rmse'])][['fold', 'rmse']].reset_index(drop=True)
            if optimizer_name == 'None':
                baseline_map[(zone_label, model_name)] = ft2
            else:
                compare_map[(zone_label, model_name, optimizer_name)] = ft2

    aggregate_pairs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for (zone_label, model_name, optimizer_name), comp_df in compare_map.items():
        base_df = baseline_map.get((zone_label, model_name))
        if base_df is None or base_df.empty or comp_df.empty:
            continue
        n_pairs = min(len(base_df), len(comp_df))
        if n_pairs < 2:
            continue
        paired = pd.DataFrame({
            'fold': base_df['fold'].to_numpy(dtype=int)[:n_pairs],
            'baseline_rmse': base_df['rmse'].to_numpy(dtype=float)[:n_pairs],
            'optimizer_rmse': comp_df['rmse'].to_numpy(dtype=float)[:n_pairs],
        })
        paired['improvement'] = paired['baseline_rmse'] - paired['optimizer_rmse']
        paired = paired[np.isfinite(paired['baseline_rmse']) & np.isfinite(paired['optimizer_rmse'])].reset_index(drop=True)
        if len(paired) < 2:
            continue
        stats_row = _run_wilcoxon_from_differences(paired['improvement'].to_numpy(dtype=float))
        by_zone_rows.append({
            'zone': zone_label,
            'model': model_name,
            'optimizer': optimizer_name,
            'pair_scope': 'outer_fold',
            **stats_row,
        })
        for _, prow in paired.iterrows():
            by_zone_detail_rows.append({
                'zone': zone_label,
                'model': model_name,
                'optimizer': optimizer_name,
                'pair_scope': 'outer_fold',
                'fold': int(prow['fold']),
                'pair_id': f"{zone_label}__fold_{int(prow['fold'])}",
                'baseline_rmse': float(prow['baseline_rmse']),
                'optimizer_rmse': float(prow['optimizer_rmse']),
                'improvement': float(prow['improvement']),
            })
            aggregate_pairs.setdefault((model_name, optimizer_name), []).append({
                'zone': zone_label,
                'fold': int(prow['fold']),
                'pair_id': f"{zone_label}__fold_{int(prow['fold'])}",
                'baseline_rmse': float(prow['baseline_rmse']),
                'optimizer_rmse': float(prow['optimizer_rmse']),
                'improvement': float(prow['improvement']),
            })

    if by_zone_rows:
        by_zone_df = pd.DataFrame(by_zone_rows)
        for mn, g in by_zone_df.groupby('model', sort=False):
            adj = _holm_bonferroni_adjust(g['p_value'].tolist())
            by_zone_df.loc[g.index, 'p_value_holm'] = adj
            by_zone_df.loc[g.index, 'significant_0_05'] = pd.Series(adj, index=g.index).apply(lambda v: bool(np.isfinite(v) and v <= 0.05))
        by_zone_df.to_csv(os.path.join(out_dir, 'wilcoxon_by_zone.csv'), index=False)
    if by_zone_detail_rows:
        pd.DataFrame(by_zone_detail_rows).to_csv(os.path.join(out_dir, 'wilcoxon_fold_details_by_zone.csv'), index=False)

    for (model_name, optimizer_name), pair_rows in aggregate_pairs.items():
        diffs = np.asarray([r['improvement'] for r in pair_rows], dtype=float)
        if diffs.size < 2:
            continue
        stats_row = _run_wilcoxon_from_differences(diffs)
        rows.append({
            'model': model_name,
            'optimizer': optimizer_name,
            'pair_scope': 'zone_x_outer_fold',
            **stats_row,
        })
        for r in pair_rows:
            detail_rows.append({
                'model': model_name,
                'optimizer': optimizer_name,
                'pair_scope': 'zone_x_outer_fold',
                'zone': r['zone'],
                'fold': r['fold'],
                'pair_id': r['pair_id'],
                'baseline_rmse': r['baseline_rmse'],
                'optimizer_rmse': r['optimizer_rmse'],
                'improvement': r['improvement'],
            })

    if rows:
        rows_df = pd.DataFrame(rows)
        for mn, g in rows_df.groupby('model', sort=False):
            adj = _holm_bonferroni_adjust(g['p_value'].tolist())
            rows_df.loc[g.index, 'p_value_holm'] = adj
            rows_df.loc[g.index, 'significant_0_05'] = pd.Series(adj, index=g.index).apply(lambda v: bool(np.isfinite(v) and v <= 0.05))
        rows_df.to_csv(os.path.join(out_dir, 'wilcoxon_results.csv'), index=False)
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(os.path.join(out_dir, 'wilcoxon_fold_details.csv'), index=False)


