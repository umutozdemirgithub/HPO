from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
import uuid
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", message="Calling float on a single element Series is deprecated.*", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.model_selection import KFold, TimeSeriesSplit

# Prevent common OpenMP/BLAS oversubscription stalls in constrained runtimes.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from .preprocess import (
    _read_csv_with_fallbacks,
    _set_best_effort_determinism,
    detect_zone_column,
    ensure_cyclical_time_features,
    make_pipeline,
    make_preprocess,
    read_uploaded_csv_with_fallbacks,
)
from .metrics import _metrics, _metrics_core, eval_cv, eval_cv_with_oof
from .optimizers import (
    _HAS_LGBM,
    _HAS_OPTUNA,
    _HAS_XGB,
    _IMPORT_ERRORS as _OPTIMIZER_IMPORT_ERRORS,
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
    _aggregate_params_median_mode,
    clamp_param,
    decode,
    default_models,
    default_spaces,
    encode,
    sample_param,
)
from .shap_utils import (
    _HAS_SHAP,
    _IMPORT_ERRORS as _SHAP_IMPORT_ERRORS,
    compute_shap,
    create_shap_comparison_artifacts,
)
from .figures import _generate_standard_figures, _ordered_pareto_frontier, _pareto_front
from .wilcoxon import (
    _HAS_SCIPY_STATS,
    _IMPORT_ERRORS as _WILCOXON_IMPORT_ERRORS,
    _bootstrap_mean_ci,
    _holm_bonferroni_adjust,
    _run_wilcoxon_from_differences,
    _sanitize_zone_value,
    _validate_selection_inputs,
    _wilcoxon_effect_size_from_stat,
    _aggregate_zone_wilcoxon,
)
from .artifacts import (
    _compute_fold_stability,
    _prepare_run_artifacts,
    _write_run_metadata_file,
    export_latex_table,
    export_supplementary_table,
    make_all_standard_figures,
    make_shap_figures,
)
from .paper_artifacts import write_paper_tables, write_paper_figures
from .constants import OPTIMIZER_RUNTIME_DEFAULTS
from .errors import make_error_payload, warning_payload
from .xai_artifacts import (
    _HAS_LIME,
    _LIME_IMPORT_ERROR,
    create_lime_artifacts,
    create_pfi_artifacts,
    create_xai_agreement_artifacts,
)
from .advanced_visuals import create_prediction_diagnostic_figures, create_multi_seed_figures

_IMPORT_ERRORS: Dict[str, str] = {}
_IMPORT_ERRORS.update(_OPTIMIZER_IMPORT_ERRORS)
_IMPORT_ERRORS.update(_SHAP_IMPORT_ERRORS)
_IMPORT_ERRORS.update(_WILCOXON_IMPORT_ERRORS)
if not _HAS_LIME and _LIME_IMPORT_ERROR:
    _IMPORT_ERRORS.setdefault("lime", _LIME_IMPORT_ERROR)

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, Optional[str]], None]
DataFrameDict = Dict[str, pd.DataFrame]

def _build_global_weighted_metrics(aggregate_y_true: List[float], aggregate_predictions: Dict[str, List[float]], runtime_detail: Dict[str, float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y_true_arr = np.asarray(aggregate_y_true, dtype=float)
    for method_key, pred_vals in (aggregate_predictions or {}).items():
        preds = np.asarray(pred_vals, dtype=float)
        if preds.shape[0] != y_true_arr.shape[0]:
            continue
        core = _metrics_core(y_true_arr, preds, drop_non_finite=True, include_n=True)
        model_name, optimizer_name = method_key.split('__', 1) if '__' in method_key else (method_key, 'Unknown')
        rows.append({
            'model': model_name,
            'optimizer': optimizer_name,
            **{k: core.get(k) for k in ['mae','mse','rmse','r2','medae','explained_variance','mape','max_error','bias','nrmse']},
            'runtime_s': float(runtime_detail.get(method_key, float('nan'))),
            'aggregation_mode': 'global_weighted_oof',
            'n_samples': int(core.get('n', 0)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out['MAE']=out['mae']; out['MSE']=out['mse']; out['RMSE']=out['rmse']; out['R2']=out['r2']; out['MedianAE']=out['medae']; out['ExplainedVar']=out['explained_variance']; out['MAPE']=out['mape']; out['MaxError']=out['max_error']; out['Bias']=out['bias']; out['NRMSE']=out['nrmse']
        out = out.sort_values(['rmse','mae']).reset_index(drop=True)
    return out


def run_experiment_multi_zone(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    date_column: Optional[str],
    datetime_format: Optional[str],
    folds: int,
    seed: int,
    iters: int,
    models_selected: List[str],
    optimizers_selected: List[str],
    *,
    zone_column: Optional[str],
    include_cyclical: bool = False,
    time_series_cv: bool = True,
    nested_cv: bool = False,
    strict_time_order: bool = False,
    deterministic: bool = False,
    generate_figures: bool = True,
    compute_diagnostics: bool = True,
    write_metadata: bool = True,
    with_shap: bool = True,
    xai_methods: Optional[Any] = None,
    xai_build_comparison: bool = True,
    shap_rows: int = 800,
    progress_cb: Optional[ProgressCallback] = None,
    log_cb: Optional[LogCallback] = None,
    out_dir: str = 'ard_outputs',
    ui_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not zone_column or zone_column not in df.columns:
        return run_experiment(
            df=df, feature_cols=feature_cols, target_cols=target_cols, date_column=date_column, datetime_format=datetime_format,
            folds=folds, seed=seed, iters=iters, models_selected=models_selected, optimizers_selected=optimizers_selected,
            include_cyclical=include_cyclical, time_series_cv=time_series_cv, nested_cv=nested_cv, strict_time_order=strict_time_order,
            deterministic=deterministic, generate_figures=generate_figures, compute_diagnostics=compute_diagnostics,
            write_metadata=write_metadata, with_shap=with_shap, xai_methods=xai_methods, xai_build_comparison=xai_build_comparison, shap_rows=shap_rows, progress_cb=progress_cb, log_cb=log_cb, out_dir=out_dir, ui_metadata=ui_metadata
        )
    zone_values = [z for z in df[zone_column].dropna().unique().tolist()]
    if len(zone_values) < 2:
        return run_experiment(
            df=df.drop(columns=[zone_column], errors='ignore'), feature_cols=[c for c in feature_cols if c != zone_column], target_cols=target_cols, date_column=date_column, datetime_format=datetime_format,
            folds=folds, seed=seed, iters=iters, models_selected=models_selected, optimizers_selected=optimizers_selected,
            include_cyclical=include_cyclical, time_series_cv=time_series_cv, nested_cv=nested_cv, strict_time_order=strict_time_order,
            deterministic=deterministic, generate_figures=generate_figures, compute_diagnostics=compute_diagnostics,
            write_metadata=write_metadata, with_shap=with_shap, xai_methods=xai_methods, xai_build_comparison=xai_build_comparison, shap_rows=shap_rows, progress_cb=progress_cb, log_cb=log_cb, out_dir=out_dir, ui_metadata=ui_metadata
        )
    def log(msg: str):
        if log_cb:
            log_cb(msg)
    root_run_id = time.strftime('run_%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root_out_dir = os.path.join(out_dir, root_run_id)
    os.makedirs(root_out_dir, exist_ok=True)
    root_fig_dir = os.path.join(root_out_dir, 'figures')
    os.makedirs(root_fig_dir, exist_ok=True)
    os.makedirs(os.path.join(root_out_dir, 'zones'), exist_ok=True)
    log(f"🧭 Multi-zone mode confirmed on column '{zone_column}' with {len(zone_values)} zones.")
    zone_results=[]; zone_metrics_frames=[]; aggregate_predictions={}; aggregate_y_true=[]; aggregate_fold_tables={}; history_buckets={}; runtime_buckets={}; aggregate_best_params={}; aggregate_best_params_report={}; shap_by_method={}; shap_comparisons={}; pfi_by_method={}; lime_by_method={}; xai_comparisons={}; shap_files=[]; failed_zones=[]
    for idx, zone_value in enumerate(zone_values, start=1):
        zone_label=_sanitize_zone_value(zone_value)
        zone_df=df[df[zone_column]==zone_value].copy().drop(columns=[zone_column], errors='ignore')
        min_required_rows = max(12, folds + 2)
        if len(zone_df) < min_required_rows:
            msg = f"Zone skipped because it has only {len(zone_df)} rows; minimum required is {min_required_rows}."
            skipped = warning_payload('TOO_FEW_ROWS', msg, stage='multi_zone')
            skipped.update({
                'zone': str(zone_value),
                'zone_label': zone_label,
                'error': skipped.pop('code'),
                'n_rows': int(len(zone_df)),
                'min_required_rows': int(min_required_rows),
            })
            failed_zones.append(skipped)
            log(f"⚠️  Zone {zone_value} skipped: {msg}")
            continue
        zone_out_dir=os.path.join(root_out_dir,'zones',f'zone_{zone_label}')
        log(f'🌐 Running zone {zone_value} ({idx}/{len(zone_values)})')
        zres=run_experiment_safe(df=zone_df, feature_cols=[c for c in feature_cols if c != zone_column], target_cols=target_cols, date_column=date_column, datetime_format=datetime_format, folds=folds, seed=seed, iters=iters, models_selected=models_selected, optimizers_selected=optimizers_selected, include_cyclical=include_cyclical, time_series_cv=time_series_cv, nested_cv=nested_cv, strict_time_order=strict_time_order, deterministic=deterministic, generate_figures=generate_figures, compute_diagnostics=compute_diagnostics, write_metadata=write_metadata, with_shap=with_shap, xai_methods=xai_methods, xai_build_comparison=xai_build_comparison, shap_rows=shap_rows, progress_cb=progress_cb, log_cb=log_cb, out_dir=zone_out_dir, fixed_out_dir=zone_out_dir, ui_metadata={**(ui_metadata or {}), 'zone_context': {'zone_column': zone_column, 'zone_value': zone_value, 'zone_label': zone_label}})
        if not isinstance(zres, dict):
            failed_zones.append({'zone': str(zone_value), 'zone_label': zone_label, 'error': 'INVALID_RESULT', 'message': 'Zone run returned a non-dict result.'})
            log(f"❌ Zone {zone_value} failed: invalid non-dict result")
            continue
        if 'error' in zres:
            failed_zones.append({'zone': str(zone_value), 'zone_label': zone_label, 'error': str(zres.get('error')), 'message': str(zres.get('message', ''))})
            log(f"❌ Zone {zone_value} failed: {zres.get('error')} — {zres.get('message', '')}")
            continue
        zres['zone_value']=zone_value; zres['zone_label']=zone_label
        zone_results.append(zres)
        zmetrics=zres.get('metrics_df', pd.DataFrame()).copy()
        if not zmetrics.empty:
            zmetrics.insert(0, 'zone', str(zone_value)); zone_metrics_frames.append(zmetrics)
        for key, val in (zres.get('best_params', {}) or {}).items():
            aggregate_best_params[f'{zone_label}::{key}']=val
        for key, val in (zres.get('best_params_report', {}) or {}).items():
            aggregate_best_params_report[f'{zone_label}::{key}']=val
        zone_y=[float(v) for v in (zres.get('y_true', []) or [])]; aggregate_y_true.extend(zone_y)
        for key in [f'{m}__{o}' for m in models_selected for o in optimizers_selected]:
            vals=(zres.get('predictions', {}) or {}).get(key, [float('nan')]*len(zone_y))
            aggregate_predictions.setdefault(key, []).extend([float(x) if (x == x) else float('nan') for x in vals])
        for key, val in (zres.get('fold_tables', {}) or {}).items():
            if isinstance(val, pd.DataFrame) and not val.empty:
                tmp=val.copy(); tmp.insert(0, 'zone', str(zone_value)); aggregate_fold_tables.setdefault(key, []).append(tmp)
        for key, val in (zres.get('histories', {}) or {}).items():
            history_buckets.setdefault(key, []).append(list(val))
        for key, val in (zres.get('runtime_detail', {}) or {}).items():
            runtime_buckets.setdefault(key, []).append(float(val))
        for key, val in (zres.get('shap_by_method', {}) or {}).items():
            shap_by_method[f'{zone_label}::{key}']=val
        shap_files.extend(zres.get('shap_files', []) or [])
        for key, val in (zres.get('shap_comparisons', {}) or {}).items():
            shap_comparisons[f'{zone_label}::{key}']=val
        for key, val in (zres.get('pfi_by_method', {}) or {}).items():
            pfi_by_method[f'{zone_label}::{key}']=val
        for key, val in (zres.get('lime_by_method', {}) or {}).items():
            lime_by_method[f'{zone_label}::{key}']=val
        for key, val in (zres.get('xai_comparisons', {}) or {}).items():
            xai_comparisons[f'{zone_label}::{key}']=val
    if not zone_results:
        raise ValueError('No zones could be processed successfully.')
    zone_metrics_df = pd.concat(zone_metrics_frames, ignore_index=True) if zone_metrics_frames else pd.DataFrame()
    aggregate_metrics = pd.DataFrame()
    macro_zone_metrics = pd.DataFrame()
    if not zone_metrics_df.empty:
        zone_metrics_df.to_csv(os.path.join(root_out_dir, 'zone_metrics.csv'), index=False)
        macro_zone_metrics = zone_metrics_df.groupby(['model', 'optimizer'], as_index=False).agg({'mae':'mean','mse':'mean','rmse':'mean','r2':'mean','medae':'mean','explained_variance':'mean','mape':'mean','max_error':'mean','bias':'mean','nrmse':'mean','runtime_s':'mean'})
        macro_zone_metrics['MAE']=macro_zone_metrics['mae']; macro_zone_metrics['MSE']=macro_zone_metrics['mse']; macro_zone_metrics['RMSE']=macro_zone_metrics['rmse']; macro_zone_metrics['R2']=macro_zone_metrics['r2']; macro_zone_metrics['MedianAE']=macro_zone_metrics['medae']; macro_zone_metrics['ExplainedVar']=macro_zone_metrics['explained_variance']; macro_zone_metrics['MAPE']=macro_zone_metrics['mape']; macro_zone_metrics['MaxError']=macro_zone_metrics['max_error']; macro_zone_metrics['Bias']=macro_zone_metrics['bias']; macro_zone_metrics['NRMSE']=macro_zone_metrics['nrmse']
        macro_zone_metrics['aggregation_mode'] = 'macro_zone_mean'
        macro_zone_metrics = macro_zone_metrics.sort_values(['rmse','mae']).reset_index(drop=True)
        macro_zone_metrics.to_csv(os.path.join(root_out_dir, 'metrics_macro_zone_mean.csv'), index=False)
    aggregate_fold_tables = {k:(pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k,v in aggregate_fold_tables.items()}
    aggregate_histories={}
    for key, bucket in history_buckets.items():
        max_len=max(len(seq) for seq in bucket)
        arr=np.full((len(bucket), max_len), np.nan, dtype=float)
        for i, seq in enumerate(bucket): arr[i,:len(seq)] = np.asarray(seq, dtype=float)
        aggregate_histories[key]=np.nanmean(arr, axis=0).tolist()
    aggregate_runtime_detail={k: float(np.nansum(np.asarray(v, dtype=float))) for k,v in runtime_buckets.items() if v}
    if aggregate_predictions and aggregate_y_true:
        pred_df=pd.DataFrame({'y_true': aggregate_y_true})
        for k,v in aggregate_predictions.items(): pred_df[k]=v
        pred_df.to_csv(os.path.join(root_out_dir,'oof_predictions.csv'), index=False)
    aggregate_metrics = _build_global_weighted_metrics(aggregate_y_true, aggregate_predictions, aggregate_runtime_detail)
    if not aggregate_metrics.empty:
        aggregate_metrics.to_csv(os.path.join(root_out_dir, 'metrics.csv'), index=False)
        log('📐 Wrote global weighted OOF metrics to metrics.csv')
    if not macro_zone_metrics.empty:
        log('📊 Wrote macro zone mean metrics to metrics_macro_zone_mean.csv')
    if generate_figures and aggregate_histories:
        import matplotlib.pyplot as plt
        for key, hist in aggregate_histories.items():
            _fig_mz = plt.figure()
            try:
                plt.plot(hist); plt.xlabel('iteration'); plt.ylabel('best RMSE'); plt.title(f'Convergence: {key} (zone-mean)'); plt.tight_layout(); plt.savefig(os.path.join(root_fig_dir,f'convergence_{key}.png'), dpi=200)
            finally:
                plt.close(_fig_mz)
    with open(os.path.join(root_out_dir,'best_params.json'),'w',encoding='utf-8') as f: json.dump(aggregate_best_params_report, f, indent=2, default=str)
    try:
        write_paper_tables(aggregate_metrics, aggregate_best_params_report, root_out_dir, nested_cv=nested_cv, time_series_cv=time_series_cv, with_shap=with_shap)
        if generate_figures:
            write_paper_figures(aggregate_y_true, aggregate_predictions, aggregate_metrics, aggregate_fold_tables, root_fig_dir, root_out_dir)
    except Exception as _e:
        log(f'Aggregate paper artifacts skipped: {_e}')
    with open(os.path.join(root_out_dir,'zone_summary.json'),'w',encoding='utf-8') as f: json.dump({'zone_column': zone_column, 'zones':[str(z) for z in zone_values], 'n_zones_processed': len(zone_results), 'n_zones_failed': len(failed_zones), 'failed_zones': failed_zones, 'per_zone_dirs': {str(z.get('zone_value')): z.get('out_dir') for z in zone_results}}, f, indent=2, default=str)
    if write_metadata:
        try:
            root_meta = {
                'run_id': root_run_id,
                'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'seed': int(seed),
                'folds': int(folds),
                'iters': int(iters),
                'time_series_cv': bool(time_series_cv),
                'nested_cv': bool(nested_cv),
                'strict_time_order': bool(strict_time_order),
                'generate_figures': bool(generate_figures),
                'compute_diagnostics': bool(compute_diagnostics),
                'with_shap': bool(with_shap),
                'shap_rows': int(shap_rows),
                'zone_column': zone_column,
                'zone_values': [str(z) for z in zone_values],
                'feature_cols': [c for c in feature_cols if c != zone_column],
                'target_cols': list(target_cols),
                'n_rows': int(len(df)),
                'models_selected': list(models_selected),
                'optimizers_selected': list(optimizers_selected),
                'sidebar_selections': dict(ui_metadata or {}),
                'import_errors': dict(_IMPORT_ERRORS),
            }
            with open(os.path.join(root_out_dir, 'run_metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(root_meta, f, indent=2, default=str)
        except Exception as _e:
            log(f'Run metadata could not be written: {_e}')
    if failed_zones:
        pd.DataFrame(failed_zones).to_csv(os.path.join(root_out_dir, 'failed_zones.csv'), index=False)
    if write_metadata:
        try:
            _write_run_metadata_file(
                root_out_dir,
                run_id=root_run_id,
                seed=seed,
                folds=folds,
                iters=iters,
                time_series_cv=time_series_cv,
                nested_cv=nested_cv,
                strict_time_order=strict_time_order,
                generate_figures=generate_figures,
                compute_diagnostics=compute_diagnostics,
                with_shap=with_shap,
                shap_rows=shap_rows,
                date_column=date_column,
                datetime_format=datetime_format,
                feature_cols=[c for c in feature_cols if c != zone_column],
                target_cols=target_cols,
                n_rows=len(df),
                models_selected=models_selected,
                optimizers_selected=optimizers_selected,
                ui_metadata={**(ui_metadata or {}), 'zone_column_confirmed': zone_column, 'zone_values': [str(z) for z in zone_values], 'aggregate_metric_primary': 'global_weighted_oof', 'aggregate_metric_secondary': 'macro_zone_mean'},
            )
        except Exception as _e:
            log(f"Run metadata could not be written: {_e}")
    _aggregate_zone_wilcoxon(zone_results, root_out_dir, log)
    return {'metrics_df': aggregate_metrics, 'macro_zone_metrics_df': macro_zone_metrics, 'zone_metrics_df': zone_metrics_df, 'zone_results': zone_results, 'failed_zones': failed_zones, 'fold_tables': aggregate_fold_tables, 'predictions': aggregate_predictions, 'y_true': aggregate_y_true, 'best_params': aggregate_best_params, 'best_params_report': aggregate_best_params_report, 'histories': aggregate_histories, 'runtime_detail': aggregate_runtime_detail, 'out_dir': root_out_dir, 'fig_dir': root_fig_dir, 'shap_files': shap_files, 'shap_by_method': shap_by_method, 'shap_comparisons': shap_comparisons, 'pfi_by_method': pfi_by_method, 'lime_by_method': lime_by_method, 'xai_comparisons': xai_comparisons, 'has_shap': _HAS_SHAP, 'has_lime': _HAS_LIME, 'has_lgbm': _HAS_LGBM, 'has_xgb': _HAS_XGB, 'time_axis_used': bool(date_column), 'nested_cv_used': bool(nested_cv), 'import_errors': dict(_IMPORT_ERRORS), 'zone_column': zone_column, 'zone_values': [str(z) for z in zone_values], 'multi_zone_mode': True}


def parse_seed_list(seeds: Optional[Any], default_seed: int = 42) -> List[int]:
    """Parse a comma/space/semicolon separated seed specification.

    Accepts strings such as ``"42, 101, 202"``, an integer, or any iterable
    of integers/strings. The returned list is de-duplicated while preserving
    order. Empty input falls back to ``default_seed``.
    """
    if seeds is None:
        raw_items: List[Any] = [default_seed]
    elif isinstance(seeds, (int, np.integer)):
        raw_items = [int(seeds)]
    elif isinstance(seeds, str):
        cleaned = seeds.replace(";", ",").replace("\n", ",").replace("\t", ",")
        raw_items = []
        for part in cleaned.split(","):
            raw_items.extend(part.strip().split())
        if not raw_items:
            raw_items = [default_seed]
    else:
        try:
            raw_items = list(seeds)
        except TypeError:
            raw_items = [seeds]

    parsed: List[int] = []
    seen = set()
    for item in raw_items:
        if item is None or str(item).strip() == "":
            continue
        try:
            val = int(item)
        except Exception as exc:
            raise ValueError(f"Invalid seed value: {item!r}. Use integers separated by commas, spaces, or semicolons.") from exc
        if val not in seen:
            parsed.append(val)
            seen.add(val)
    if not parsed:
        parsed = [int(default_seed)]
    return parsed


def _aggregate_multi_seed_metrics(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Return method-level mean/std summaries from concatenated per-seed metrics."""
    if per_seed_df.empty:
        return per_seed_df.copy()
    group_cols = [c for c in ["model", "optimizer"] if c in per_seed_df.columns]
    if not group_cols:
        return pd.DataFrame()
    metric_cols = [
        c for c in [
            "mae", "mse", "rmse", "r2", "medae", "explained_variance", "mape",
            "max_error", "bias", "nrmse", "runtime_s", "MAE", "MSE", "RMSE", "R2",
            "MedianAE", "ExplainedVar", "MAPE", "MaxError", "Bias", "NRMSE",
        ]
        if c in per_seed_df.columns and pd.api.types.is_numeric_dtype(per_seed_df[c])
    ]
    rows: List[Dict[str, Any]] = []
    for keys, grp in per_seed_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = {col: val for col, val in zip(group_cols, keys)}
        row["n_seeds"] = int(grp["seed"].nunique()) if "seed" in grp.columns else int(len(grp))
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        # Canonical display columns: mean values keep Streamlit/LaTeX display compatible.
        alias_pairs = {
            "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R2",
            "medae": "MedianAE", "explained_variance": "ExplainedVar", "mape": "MAPE",
            "max_error": "MaxError", "bias": "Bias", "nrmse": "NRMSE",
        }
        for lower, upper in alias_pairs.items():
            mean_key = f"{lower}_mean" if f"{lower}_mean" in row else f"{upper}_mean"
            std_key = f"{lower}_std" if f"{lower}_std" in row else f"{upper}_std"
            if mean_key in row:
                row[upper] = row[mean_key]
            if std_key in row:
                row[f"{upper}_std"] = row[std_key]
        if "runtime_s_mean" in row:
            row["runtime_s"] = row["runtime_s_mean"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        sort_cols = [c for c in ["rmse_mean", "mae_mean", "RMSE", "MAE"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
    return out


def run_experiment_multi_seed(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    *,
    seeds: Any,
    date_column: Optional[str],
    datetime_format: Optional[str] = None,
    folds: int,
    iters: int,
    models_selected: List[str],
    optimizers_selected: List[str],
    include_cyclical: bool = False,
    time_series_cv: bool = True,
    nested_cv: bool = False,
    strict_time_order: bool = False,
    deterministic: bool = False,
    generate_figures: bool = True,
    compute_diagnostics: bool = True,
    write_metadata: bool = True,
    with_shap: bool = True,
    xai_methods: Optional[Any] = None,
    xai_build_comparison: bool = True,
    shap_rows: int = 800,
    progress_cb: Optional[ProgressCallback] = None,
    log_cb: Optional[LogCallback] = None,
    out_dir: str = "ard_outputs",
    ui_metadata: Optional[Dict[str, Any]] = None,
    zone_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full experiment repeatedly over multiple random seeds.

    Output layout:
        <out_dir>/multi_seed_<timestamp>_<id>/
            seed_42/...
            seed_101/...
            per_seed_metrics.csv
            multi_seed_summary.csv
            multi_seed_best_methods.csv
    """
    seed_values = parse_seed_list(seeds, default_seed=42)
    if len(seed_values) == 1:
        # Preserve single-seed semantics while accepting a one-item seed list.
        kwargs: Dict[str, Any] = dict(
            df=df, feature_cols=feature_cols, target_cols=target_cols,
            date_column=date_column, datetime_format=datetime_format, folds=folds,
            seed=int(seed_values[0]), iters=iters, models_selected=models_selected,
            optimizers_selected=optimizers_selected, include_cyclical=include_cyclical,
            time_series_cv=time_series_cv, nested_cv=nested_cv,
            strict_time_order=strict_time_order, deterministic=deterministic,
            generate_figures=generate_figures, compute_diagnostics=compute_diagnostics,
            write_metadata=write_metadata, with_shap=with_shap, xai_methods=xai_methods, xai_build_comparison=xai_build_comparison, shap_rows=shap_rows,
            progress_cb=progress_cb, log_cb=log_cb, out_dir=out_dir, ui_metadata=ui_metadata,
        )
        if zone_column:
            kwargs["zone_column"] = zone_column
            return run_experiment_multi_zone(**kwargs)
        return run_experiment(**kwargs)

    root_run_id = time.strftime("multi_seed_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root_out_dir = os.path.join(str(out_dir), root_run_id)
    os.makedirs(root_out_dir, exist_ok=True)
    root_fig_dir = os.path.join(root_out_dir, "figures")
    os.makedirs(root_fig_dir, exist_ok=True)

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    def prog(p: float, msg: Optional[str] = None) -> None:
        if progress_cb:
            progress_cb(p, msg)

    log(f"🔁 Multi-seed mode enabled: seeds={seed_values}")
    seed_results: List[Dict[str, Any]] = []
    failed_seeds: List[Dict[str, Any]] = []
    per_seed_frames: List[pd.DataFrame] = []

    for idx, seed_val in enumerate(seed_values, start=1):
        base_progress = (idx - 1) / len(seed_values)
        span = 1.0 / len(seed_values)
        prog(base_progress, f"Running seed {seed_val} ({idx}/{len(seed_values)})…")

        def child_progress(p: float, msg: Optional[str] = None, *, _base=base_progress, _span=span, _seed=seed_val) -> None:
            prog(min(0.999, _base + max(0.0, min(1.0, p)) * _span), f"Seed {_seed}: {msg or ''}".strip())

        seed_out_dir = os.path.join(root_out_dir, f"seed_{seed_val}")
        seed_metadata = dict(ui_metadata or {})
        seed_metadata.update({"multi_seed_mode": True, "multi_seed_root": root_out_dir, "seed_index": idx, "seed_values": seed_values})
        kwargs = dict(
            df=df, feature_cols=feature_cols, target_cols=target_cols,
            date_column=date_column, datetime_format=datetime_format, folds=folds,
            seed=int(seed_val), iters=iters, models_selected=models_selected,
            optimizers_selected=optimizers_selected, include_cyclical=include_cyclical,
            time_series_cv=time_series_cv, nested_cv=nested_cv,
            strict_time_order=strict_time_order, deterministic=deterministic,
            generate_figures=generate_figures, compute_diagnostics=compute_diagnostics,
            write_metadata=write_metadata, with_shap=with_shap, xai_methods=xai_methods, xai_build_comparison=xai_build_comparison, shap_rows=shap_rows,
            progress_cb=child_progress, log_cb=log_cb, out_dir=seed_out_dir,
            fixed_out_dir=seed_out_dir, ui_metadata=seed_metadata,
        )
        if zone_column:
            kwargs["zone_column"] = zone_column
            kwargs.pop("fixed_out_dir", None)  # multi-zone creates its own nested zone folders
        try:
            res = run_experiment_multi_zone(**kwargs) if zone_column else run_experiment(**kwargs)
            seed_results.append(res)
            mdf = res.get("metrics_df")
            if isinstance(mdf, pd.DataFrame) and not mdf.empty:
                tmp = mdf.copy()
                tmp.insert(0, "seed", int(seed_val))
                tmp.insert(1, "seed_index", idx)
                tmp["seed_out_dir"] = res.get("out_dir", seed_out_dir)
                per_seed_frames.append(tmp)
        except Exception as exc:
            payload = make_error_payload(exc, stage="multi_seed", engine_file=__file__)
            failed_seeds.append({"seed": int(seed_val), **payload})
            log(f"❌ Seed {seed_val} failed: {payload.get('message')}")

    if not per_seed_frames:
        if failed_seeds:
            pd.DataFrame(failed_seeds).to_csv(os.path.join(root_out_dir, "failed_seeds.csv"), index=False)
        raise RuntimeError("All multi-seed runs failed. See failed_seeds.csv for details.")

    per_seed_metrics = pd.concat(per_seed_frames, ignore_index=True)
    summary = _aggregate_multi_seed_metrics(per_seed_metrics)
    per_seed_metrics.to_csv(os.path.join(root_out_dir, "per_seed_metrics.csv"), index=False)
    summary.to_csv(os.path.join(root_out_dir, "multi_seed_summary.csv"), index=False)
    if not summary.empty:
        summary.head(10).to_csv(os.path.join(root_out_dir, "multi_seed_best_methods.csv"), index=False)
    if failed_seeds:
        pd.DataFrame(failed_seeds).to_csv(os.path.join(root_out_dir, "failed_seeds.csv"), index=False)

    if generate_figures:
        try:
            create_multi_seed_figures(per_seed_metrics=per_seed_metrics, summary=summary, fig_dir=root_fig_dir, out_dir=root_out_dir)
        except Exception as _e:
            log(f"Multi-seed stability figures skipped: {_e}")

    metadata = {
        "multi_seed_mode": True,
        "seed_values": seed_values,
        "n_seeds_requested": len(seed_values),
        "n_seeds_completed": int(per_seed_metrics["seed"].nunique()),
        "n_seeds_failed": len(failed_seeds),
        "nested_cv": bool(nested_cv),
        "time_series_cv": bool(time_series_cv),
        "deterministic": bool(deterministic),
        "models_selected": list(models_selected),
        "optimizers_selected": list(optimizers_selected),
        "zone_column": zone_column,
        "ui_metadata": ui_metadata or {},
    }
    with open(os.path.join(root_out_dir, "multi_seed_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    prog(1.0, f"Multi-seed completed: {per_seed_metrics['seed'].nunique()}/{len(seed_values)} seed(s)")
    return {
        "metrics_df": summary,
        "multi_seed_summary_df": summary,
        "per_seed_metrics_df": per_seed_metrics,
        "seed_results": seed_results,
        "failed_seeds": failed_seeds,
        "out_dir": root_out_dir,
        "fig_dir": root_fig_dir,
        "has_shap": _HAS_SHAP,
        "has_lgbm": _HAS_LGBM,
        "has_xgb": _HAS_XGB,
        "time_axis_used": bool(date_column),
        "nested_cv_used": bool(nested_cv),
        "import_errors": dict(_IMPORT_ERRORS),
        "zone_column": zone_column,
        "multi_zone_mode": bool(zone_column),
        "multi_seed_mode": True,
        "seed_values": seed_values,
    }

# -----------------------------
# CLI entry point
# -----------------------------
def run_experiment_from_path(
    data_path: str,
    target: str,
    features: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    delimiter: str = ",",
    datetime_format: Optional[str] = None,
    folds: int = 3,
    seed: int = 42,
    seeds: Optional[Any] = None,
    multi_seed: bool = False,
    iters: int = 40,
    models_selected: Optional[List[str]] = None,
    optimizers_selected: Optional[List[str]] = None,
    time_series_cv: bool = True,
    nested_cv: bool = False,
    with_shap: bool = False,
    xai_methods: Optional[Any] = None,
    shap_rows: int = 800,
    strict_time_order: bool = False,
    deterministic: bool = False,
    generate_figures: bool = True,
    compute_diagnostics: bool = True,
    write_metadata: bool = True,
    out_dir: str = "ard_outputs",
    ui_metadata: Optional[Dict[str, Any]] = None,
    zone_column: Optional[str] = None,
) -> Dict[str, Any]:
    ext = os.path.splitext(data_path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = _read_csv_with_fallbacks(data_path, sep=delimiter)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path)
    elif ext == ".mat":
        mat = loadmat(data_path)
        candidates = []
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if hasattr(v, "ndim") and getattr(v, "ndim", 0) == 2 and getattr(v, "dtype", None) is not None:
                if v.dtype.kind in "iuf" and v.size > 0:
                    candidates.append((k, v))
        if not candidates:
            raise ValueError(
                "No 2D numeric array found in .mat file. Please export as CSV/XLSX or include a 2D numeric matrix variable."
            )
        var_name, arr = max(candidates, key=lambda kv: kv[1].size)
        df = pd.DataFrame(arr)
        df.columns = [f"{var_name}_{i}" for i in range(df.shape[1])]
    else:
        raise ValueError("Unsupported file format. Use CSV, TXT, XLSX, or MAT.")

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    if features is None:
        features = [c for c in df.columns if c != target and c != date_col]

    if models_selected is None:
        models_selected = list(default_models(seed).keys())
    if optimizers_selected is None:
        optimizers_selected = ["None"]

    run_kwargs = dict(
        df=df,
        feature_cols=features,
        target_cols=[target],
        date_column=date_col,
        datetime_format=datetime_format,
        folds=int(folds),
        seed=int(seed),
        seeds=seeds,
        multi_seed=bool(multi_seed),
        iters=int(iters),
        models_selected=models_selected,
        optimizers_selected=optimizers_selected,
        time_series_cv=bool(time_series_cv),
        nested_cv=bool(nested_cv),
        strict_time_order=bool(strict_time_order),
        deterministic=bool(deterministic),
        with_shap=bool(with_shap),
        xai_methods=xai_methods,
        shap_rows=int(shap_rows),
        generate_figures=bool(generate_figures),
        compute_diagnostics=bool(compute_diagnostics),
        write_metadata=bool(write_metadata),
        out_dir=str(out_dir),
        ui_metadata=ui_metadata,
    )
    if zone_column:
        run_kwargs["zone_column"] = zone_column
    return run_experiment_safe(**run_kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# run_experiment sub-functions
# Each is independently testable and has a single responsibility.
# run_experiment() orchestrates them in order.
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    date_column: Optional[str],
    datetime_format: Optional[str],
    include_cyclical: bool,
    time_series_cv: bool,
    strict_time_order: bool,
    log: LogCallback,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, List[str]]:
    """Sort chronologically, add cyclical features, return (df2, y, X, effective_feature_cols).

    Raises ValueError when strict_time_order=True and date_column is missing/invalid.
    """
    df0 = df.copy()

    if time_series_cv and (not date_column or date_column not in df0.columns):
        msg = (
            "Time-series CV (TimeSeriesSplit) is enabled but no valid date_column was provided. "
            "The dataset will be split using its current row order, which can introduce temporal leakage "
            "if the rows are not already sorted chronologically. Provide date_column (recommended), "
            "or disable time_series_cv. You can also set strict_time_order=True to raise an error."
        )
        if strict_time_order:
            raise ValueError(msg)
        log(f"⚠️  {msg}")

    if time_series_cv and date_column and date_column in df0.columns:
        dt = pd.to_datetime(df0[date_column], format=datetime_format, errors="coerce")
        n_bad = int(dt.isna().sum())
        if n_bad > 0 and strict_time_order:
            raise ValueError(
                f"strict_time_order=True but {n_bad} rows in date column '{date_column}' could not be parsed. "
                "Fix or remove invalid timestamps before running a time-series experiment."
            )
        if n_bad < len(dt):
            if n_bad > 0:
                log(f"⚠️  {n_bad} rows have unparsable datetime values in '{date_column}'. They will sort to the end.")
            df0 = (
                df0.assign(_ard_dt_sort=dt)
                .sort_values("_ard_dt_sort", kind="mergesort")
                .drop(columns=["_ard_dt_sort"])
                .reset_index(drop=True)
            )
        else:
            msg = f"Date column '{date_column}' could not be parsed; dataset order is left unchanged."
            if strict_time_order:
                raise ValueError(msg)
            log(f"⚠️  {msg}")

    if include_cyclical:
        before = len(df0)
        df2 = ensure_cyclical_time_features(df0, date_column, datetime_format)
        dropped = before - len(df2)
        if dropped > 0:
            log(f"⚠️  Dropped {dropped} rows due to datetime parsing failures while creating cyclical features.")
    else:
        df2 = df0

    if not target_cols:
        raise ValueError("At least one target must be selected.")

    effective_feature_cols = list(dict.fromkeys(feature_cols))
    if date_column and date_column in effective_feature_cols:
        effective_feature_cols = [c for c in effective_feature_cols if c != date_column]
        log(f"⚠️  Date column '{date_column}' was removed from feature_cols to avoid temporal leakage and dtype errors.")
    if include_cyclical:
        cyc_cols = [c for c in ["hour_sin", "hour_cos", "doy_sin", "doy_cos"] if c in df2.columns]
        effective_feature_cols = effective_feature_cols + [c for c in cyc_cols if c not in effective_feature_cols]

    missing_features = [c for c in effective_feature_cols if c not in df2.columns]
    if missing_features:
        raise ValueError(f"Selected feature columns are missing from the dataset after preprocessing: {missing_features}")
    if target_cols[0] not in df2.columns:
        raise ValueError(f"Selected target column is missing from the dataset: {target_cols[0]}")

    y = df2[target_cols[0]].astype(float).to_numpy()
    X = df2[effective_feature_cols].copy()
    log(f"🧪 Dataset ready • rows={len(df2)} • features={len(effective_feature_cols)} • target={target_cols[0]}")
    return df2, y, X, effective_feature_cols


def _run_cv_loop(
    models: Dict[str, Any],
    optimizers_selected: List[str],
    opt_map: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    preprocess: Any,
    outer_cv: Any,
    folds: int,
    iters: int,
    seed: int,
    nested_cv: bool,
    time_series_cv: bool,
    log: LogCallback,
    prog: ProgressCallback,
    total_jobs: int,
) -> Tuple[
    List[Dict[str, Any]],         # rows
    Dict[str, pd.DataFrame],    # fold_tables
    Dict[str, List[float]],       # predictions
    Dict[str, Dict[str, Any]],    # best_params
    Dict[str, Dict[str, Any]],    # best_params_report
    Dict[str, List[float]],       # histories
    Dict[str, float],             # runtime_detail
    List[Dict[str, Any]],         # failed_jobs
]:
    """Execute the model × optimizer CV matrix and return all collected results.

    Each (model, optimizer) combination is evaluated independently; failures are
    captured in failed_jobs rather than aborting the entire run.
    """
    rows: List[Dict[str, Any]] = []
    fold_tables: Dict[str, "pd.DataFrame"] = {}
    predictions: Dict[str, List[float]] = {}
    best_params: Dict[str, Dict[str, Any]] = {}
    best_params_report: Dict[str, Dict[str, Any]] = {}
    histories: Dict[str, List[float]] = {}
    runtime_detail: Dict[str, float] = {}
    failed_jobs: List[Dict[str, Any]] = []
    done = 0

    def _inner_cv(n_train: int):
        n_splits = max(2, min(int(folds), 3))
        if n_train < (n_splits + 1) * 5:
            n_splits = 2
        if time_series_cv:
            return TimeSeriesSplit(n_splits=n_splits)
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _run_optimizer(obj: Objective, space: Dict[str, ParamSpec], oname: str) -> OptResult:
        opt_cls = opt_map[oname]
        opt = opt_cls(space, seed=seed)
        if oname == "PSO":
            cfg = OPTIMIZER_RUNTIME_DEFAULTS["PSO"]
            n_particles = max(int(cfg["min_particles"]), min(int(cfg["max_particles"]), int(cfg["particles_per_dim"]) * len(space) + int(cfg["particle_offset"])))
            return opt.run(obj, iters=iters, n_particles=n_particles)
        elif oname == "GA":
            cfg = OPTIMIZER_RUNTIME_DEFAULTS["GA"]
            pop = max(int(cfg["min_pop"]), min(int(cfg["max_pop"]), int(cfg["pop_per_dim"]) * len(space) + int(cfg["pop_offset"])))
            return opt.run(obj, iters=iters, pop=pop)
        elif oname == "ABC":
            cfg = OPTIMIZER_RUNTIME_DEFAULTS["ABC"]
            pop = max(int(cfg["min_pop"]), min(int(cfg["max_pop"]), int(cfg["pop_per_dim"]) * len(space) + int(cfg["pop_offset"])))
            limit = max(int(cfg["min_limit"]), iters // int(cfg["limit_fraction"]))
            return opt.run(obj, iters=iters, pop=pop, limit=limit)
        else:
            return opt.run(obj, iters=iters)

    for mname, model in models.items():
        space = default_spaces(mname)
        base_pipe = make_pipeline(model, preprocess)

        for oname in optimizers_selected:
            done += 1
            prog(done / total_jobs, f"{mname} / {oname}…")
            log(f"▶ {mname} + {oname} (nested_cv={nested_cv})")
            t0 = time.time()
            params_per_outer_fold = None
            inner_best_scores_per_outer_fold = None
            selected_outer_fold = None

            try:
                if not nested_cv:
                    if oname == "None" or len(space) == 0:
                        params: Dict[str, Any] = {}
                        best_pipe = clone(base_pipe)
                        fold_df, mean, oof_pred = eval_cv_with_oof(best_pipe, X, y, outer_cv)
                        hist = [float(mean["rmse"])]
                    else:
                        obj = Objective(base_pipe, space, X, y, outer_cv)
                        opt_res = _run_optimizer(obj, space, oname)
                        params = opt_res.best_params
                        hist = opt_res.history
                        best_pipe = clone(base_pipe)
                        best_pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
                        fold_df, mean, oof_pred = eval_cv_with_oof(best_pipe, X, y, outer_cv)
                else:
                    outer_rows: List[Dict[str, Any]] = []
                    history_all: List[float] = []
                    optimizer_histories_per_outer_fold: List[List[float]] = []
                    chosen_params_per_fold: List[Dict[str, Any]] = []
                    oof_pred = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)
                    for ofold, (tr, te) in enumerate(outer_cv.split(X)):
                        Xtr, ytr = X.iloc[tr], y[tr]
                        Xte, yte = X.iloc[te], y[te]
                        if oname == "None" or len(space) == 0:
                            params_fold: Dict[str, Any] = {}
                            est = clone(base_pipe)
                            est.fit(Xtr, ytr)
                            pred = est.predict(Xte)
                            oof_pred[te] = pred
                            m = _metrics(yte, pred)
                            outer_rows.append({"fold": ofold, **m})
                            fold_rmse = float(m["rmse"])
                            history_all.append(fold_rmse)
                            optimizer_histories_per_outer_fold.append([fold_rmse])
                            chosen_params_per_fold.append(params_fold)
                        else:
                            inner_cv = _inner_cv(len(tr))
                            obj_inner = Objective(base_pipe, space, Xtr, ytr, inner_cv)
                            opt_res = _run_optimizer(obj_inner, space, oname)
                            params_fold = opt_res.best_params
                            pipe_fold = clone(base_pipe)
                            pipe_fold.set_params(**{f"model__{k}": v for k, v in params_fold.items()})
                            est = clone(pipe_fold)
                            est.fit(Xtr, ytr)
                            pred = est.predict(Xte)
                            oof_pred[te] = pred
                            m = _metrics(yte, pred)
                            outer_rows.append({"fold": ofold, **m})
                            history_all.append(float(opt_res.best_score))
                            optimizer_histories_per_outer_fold.append([float(v) for v in (opt_res.history or [])])
                            chosen_params_per_fold.append(params_fold)

                    fold_df = pd.DataFrame(outer_rows)
                    mean = {col: float(fold_df[col].mean()) for col in fold_df.columns if col != "fold"}
                    params_per_outer_fold = chosen_params_per_fold
                    inner_best_scores_per_outer_fold = [float(x) for x in history_all]
                    params = _aggregate_params_median_mode(params_per_outer_fold, space)
                    selected_outer_fold = None
                    if optimizer_histories_per_outer_fold:
                        max_hist_len = max(len(seq) for seq in optimizer_histories_per_outer_fold)
                        hist_arr = np.full((len(optimizer_histories_per_outer_fold), max_hist_len), np.nan, dtype=float)
                        for i_hist, seq in enumerate(optimizer_histories_per_outer_fold):
                            if seq:
                                hist_arr[i_hist, :len(seq)] = np.asarray(seq, dtype=float)
                        hist = np.nanmean(hist_arr, axis=0).tolist()
                    else:
                        hist = inner_best_scores_per_outer_fold

                fold_df = fold_df.copy()
                fold_df.columns = [str(c).strip().lower() for c in fold_df.columns]
                rt = time.time() - t0
                key = f"{mname}__{oname}"
                fold_tables[key] = fold_df
                best_params[key] = params
                rep: Dict[str, Any] = {
                    "nested_cv": bool(nested_cv),
                    "selected_params": params,
                    "selection_rule": ("aggregate_median_mode" if nested_cv else "single_cv_best"),
                }
                if nested_cv:
                    rep["selected_outer_fold"] = selected_outer_fold
                    rep["params_per_outer_fold"] = params_per_outer_fold
                    rep["inner_best_scores_per_outer_fold"] = inner_best_scores_per_outer_fold
                    rep["optimizer_history_per_outer_fold"] = optimizer_histories_per_outer_fold
                    rep["convergence_history"] = list(hist)
                    rep["history_aggregation"] = "mean_optimizer_history_across_outer_folds"
                best_params_report[key] = rep
                histories[key] = list(hist)
                runtime_detail[key] = float(rt)
                predictions[key] = [float(x) if (x == x) else float("nan") for x in oof_pred]
                rows.append({
                    "model": mname, "optimizer": oname,
                    "mae":   float(mean.get("mae",   float("nan"))),
                    "mse":   float(mean.get("mse",   float("nan"))),
                    "rmse":  float(mean.get("rmse",  float("nan"))),
                    "r2":    float(mean.get("r2",    float("nan"))),
                    "medae": float(mean.get("medae", float("nan"))),
                    "explained_variance": float(mean.get("explained_variance", float("nan"))),
                    "mape":      float(mean.get("mape",      float("nan"))),
                    "max_error": float(mean.get("max_error", float("nan"))),
                    "bias":      float(mean.get("bias",      float("nan"))),
                    "nrmse":     float(mean.get("nrmse",     float("nan"))),
                    "runtime_s": float(rt),
                })
                log(f"✅ {mname} + {oname} → RMSE={mean.get('rmse', float('nan')):.4f}  R²={mean.get('r2', float('nan')):.4f}  [{rt:.1f}s]")
            except Exception as exc:
                rt = time.time() - t0
                failed_jobs.append({
                    "model": mname, "optimizer": oname,
                    "error": type(exc).__name__, "message": str(exc), "runtime_s": float(rt),
                })
                runtime_detail[f"{mname}__{oname}"] = float(rt)
                log(f"❌ {mname} + {oname} failed: {type(exc).__name__}: {exc}")
                continue

    return rows, fold_tables, predictions, best_params, best_params_report, histories, runtime_detail, failed_jobs


def _build_ensemble_comparison(
    y: np.ndarray,
    predictions: Dict[str, List[float]],
    metrics_df: "pd.DataFrame",
    out_dir: str,
    log,
) -> None:
    """Compute mean-ensemble metrics for top-1/3/5 methods and write CSV."""
    try:
        y_true_arr = np.asarray(y, dtype=float)
        method_keys_ord = [f"{r['model']}__{r['optimizer']}" for _, r in metrics_df.iterrows()]
        used_keys = [k for k in method_keys_ord if k in predictions]

        def _sm(y_t, y_p):
            return _metrics_core(y_t, y_p, drop_non_finite=True, include_n=True)

        def _valid_prediction_array(key: str) -> Optional[np.ndarray]:
            arr = np.asarray(predictions.get(key, []), dtype=float)
            if arr.ndim != 1 or arr.size != y_true_arr.size or not np.isfinite(arr).any():
                return None
            return arr

        valid_keys: List[str] = []
        valid_arrays: Dict[str, np.ndarray] = {}
        for key in used_keys:
            arr = _valid_prediction_array(key)
            if arr is not None:
                valid_keys.append(key)
                valid_arrays[key] = arr

        ens_rows: List[Dict[str, Any]] = []
        if valid_keys:
            ens_rows.append({"name": "Best single method", "method": valid_keys[0],
                             **_sm(y_true_arr, valid_arrays[valid_keys[0]])})
        for n_top, label in [(3, "Top-3 mean ensemble"), (5, "Top-5 mean ensemble")]:
            if len(valid_keys) >= n_top:
                keys = valid_keys[:n_top]
                stack = np.vstack([valid_arrays[k] for k in keys])
                valid_counts = np.sum(np.isfinite(stack), axis=0)
                sums = np.nansum(stack, axis=0)
                avg = np.divide(sums, valid_counts, out=np.full(stack.shape[1], np.nan), where=valid_counts > 0)
                ens_rows.append({"name": label, "method": ", ".join(keys), **_sm(y_true_arr, avg)})

        if not ens_rows:
            ens_rows = [{"name": "N/A", "method": "", "mae": float("nan"), "mse": float("nan"),
                         "rmse": float("nan"), "r2": float("nan"), "medae": float("nan"),
                         "explained_variance": float("nan"), "mape": float("nan"),
                         "max_error": float("nan"), "bias": float("nan"), "nrmse": float("nan"), "n": 0}]
        pd.DataFrame(ens_rows).to_csv(os.path.join(out_dir, "metrics_ensemble_comparison.csv"), index=False)
    except Exception as _e:
        log(f"Ensemble comparison skipped: {_e}")


def _write_run_artifacts(
    out_dir: str,
    y: np.ndarray,
    metrics_df: "pd.DataFrame",
    predictions: Dict[str, List[float]],
    best_params_report: Dict[str, Dict[str, Any]],
    fold_tables: Dict[str, "pd.DataFrame"],
    compute_diagnostics: bool,
    log,
) -> None:
    """Write metrics.csv, oof_predictions.csv, best_params.json, supplementary S1, fold stability."""
    metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    try:
        pred_df = pd.DataFrame({"y_true": [float(v) for v in y]})
        for k, v in predictions.items():
            pred_df[k] = v
        pred_df.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    except Exception as _e:
        log(f"OOF predictions file skipped: {_e}")

    try:
        with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params_report, f, indent=2, default=str)
    except Exception as _e:
        log(f"best_params.json skipped: {_e}")

    try:
        export_supplementary_table(os.path.join(out_dir, "supplementary_S1_param_space.csv"))
    except Exception as _e:
        log(f"Supplementary table export skipped: {_e}")

    if compute_diagnostics:
        _compute_fold_stability(fold_tables, out_dir, log)



def _normalize_xai_methods(xai_methods: Optional[Any], *, with_shap: bool = True) -> set:
    """Normalize user-selected XAI methods to a canonical uppercase set."""
    if not with_shap:
        return set()
    if xai_methods is None:
        return {"SHAP", "PFI", "LIME"}
    if isinstance(xai_methods, str):
        raw = [p.strip() for p in re.split(r"[,;\s]+", xai_methods) if p.strip()]
    else:
        raw = [str(p).strip() for p in xai_methods if str(p).strip()]
    aliases = {"PERMUTATION": "PFI", "PERMUTATION_IMPORTANCE": "PFI"}
    allowed = {"SHAP", "PFI", "LIME"}
    out = set()
    for item in raw:
        key = aliases.get(item.upper(), item.upper())
        if key in allowed:
            out.add(key)
    return out

def _run_shap_pipeline(
    metrics_df: "pd.DataFrame",
    best_params: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    effective_feature_cols: List[str],
    out_dir: str,
    seed: int,
    shap_rows: int,
    generate_figures: bool,
    deterministic: bool,
    log: LogCallback,
    xai_methods: Optional[Any] = None,
    xai_build_comparison: bool = True,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Retrain each model on the full dataset and compute SHAP, PFI, LIME values + comparisons.

    Returns (shap_files, shap_by_method, shap_comparisons, pfi_by_method, lime_by_method, xai_comparisons).

    Note: models are retrained on the FULL dataset (X, y). This is intentional —
    SHAP here serves global explainability, not generalisation estimation.
    Generalisation is already captured by the nested CV metrics.
    """
    shap_files: List[str] = []
    shap_by_method: Dict[str, Dict[str, Any]] = {}
    shap_comparisons: Dict[str, Dict[str, Any]] = {}
    pfi_by_method: Dict[str, Dict[str, Any]] = {}
    lime_by_method: Dict[str, Dict[str, Any]] = {}
    xai_comparisons: Dict[str, Dict[str, Any]] = {}

    selected_xai = _normalize_xai_methods(xai_methods, with_shap=True)
    if not selected_xai:
        log("XAI requested but no XAI method was selected; skipping SHAP/PFI/LIME.")
        return shap_files, shap_by_method, shap_comparisons, pfi_by_method, lime_by_method, xai_comparisons
    if "SHAP" in selected_xai and not _HAS_SHAP:
        log("SHAP requested but unavailable in this environment; continuing with other selected XAI methods if available.")

    if len(metrics_df) == 0:
        return shap_files, shap_by_method, shap_comparisons, pfi_by_method, lime_by_method, xai_comparisons

    try:
        _deterministic = bool(deterministic)
        all_default_models = default_models(seed, _deterministic)
        models_present = list(dict.fromkeys(metrics_df["model"].tolist()))

        for bm in models_present:
            model_best = all_default_models.get(bm)
            if model_best is None:
                continue
            preprocess_best = make_preprocess(effective_feature_cols)
            model_rows = metrics_df[metrics_df["model"] == bm].sort_values("rmse").reset_index(drop=True)
            best_non_none = model_rows[model_rows["optimizer"] != "None"].head(1)

            variants: List[Tuple[str, str, Dict[str, Any]]] = [("None", f"{bm}__None", {})]
            if len(best_non_none) > 0:
                bo = str(best_non_none.iloc[0]["optimizer"])
                mk = f"{bm}__{bo}"
                variants.append((bo, mk, best_params.get(mk, {})))
            else:
                best_any = model_rows.head(1)
                if len(best_any) > 0 and str(best_any.iloc[0]["optimizer"]) != "None":
                    bo = str(best_any.iloc[0]["optimizer"])
                    mk = f"{bm}__{bo}"
                    variants.append((bo, mk, best_params.get(mk, {})))

            seen: set = set()
            for opt_name, method_key, bp in variants:
                if method_key in seen:
                    continue
                seen.add(method_key)
                log(f"🧠 XAI: {bm}/{opt_name} — retrained on full dataset ({len(X)} rows); methods={sorted(selected_xai)}")
                pipe = make_pipeline(clone(model_best), preprocess_best)
                if bp:
                    pipe.set_params(**{f"model__{k}": v for k, v in bp.items()})
                pipe.fit(X, y)  # full-data retrain — intentional, see docstring

                importance_csv: Optional[str] = None
                if "SHAP" in selected_xai:
                    if _HAS_SHAP:
                        try:
                            files = compute_shap(pipe, X, out_dir=out_dir, max_rows=int(shap_rows),
                                                 seed=seed, subdir=method_key, generate_figures=bool(generate_figures))
                            importance_csv = os.path.join(out_dir, "shap", method_key, "shap_importance.csv")
                            shap_by_method[method_key] = {
                                "model": bm, "optimizer": opt_name, "files": files,
                                "dir": os.path.join(out_dir, "shap", method_key),
                                "out_dir": os.path.join(out_dir, "shap", method_key),
                                "importance_csv": importance_csv, "method_key": method_key,
                            }
                            shap_files.extend(files)
                            if not os.path.exists(importance_csv):
                                log(f"SHAP importance CSV missing for {method_key}: {importance_csv}")
                        except Exception as _shap_exc:
                            log(f"SHAP failed for {method_key}: {_shap_exc}")
                    else:
                        log(f"SHAP skipped for {method_key}: optional shap package unavailable.")

                # PFI and LIME use the same full-data retrained pipeline. They are
                # saved separately so Streamlit can render each XAI family in its own section.
                if "PFI" in selected_xai:
                    try:
                        log(f"🧪 PFI: {bm}/{opt_name} — permutation importance on sampled rows")
                        pfi_meta = create_pfi_artifacts(
                            pipe, X, y, out_dir=out_dir, method_key=method_key,
                            max_rows=int(shap_rows), seed=seed, generate_figures=bool(generate_figures)
                        )
                        pfi_meta.update({"model": bm, "optimizer": opt_name, "method_key": method_key})
                        pfi_by_method[method_key] = pfi_meta
                    except Exception as _pfi_exc:
                        log(f"PFI failed for {method_key}: {_pfi_exc}")

                if "LIME" in selected_xai:
                    try:
                        log(f"🧩 LIME: {bm}/{opt_name} — local explanations on selected samples")
                        lime_meta = create_lime_artifacts(
                            pipe, X, y, out_dir=out_dir, method_key=method_key,
                            max_rows=int(shap_rows), seed=seed, generate_figures=bool(generate_figures)
                        )
                        lime_meta.update({"model": bm, "optimizer": opt_name, "method_key": method_key})
                        lime_by_method[method_key] = lime_meta
                    except Exception as _lime_exc:
                        log(f"LIME failed for {method_key}: {_lime_exc}")

                if bool(xai_build_comparison) and len(selected_xai.intersection({"SHAP", "PFI", "LIME"})) >= 2:
                    try:
                        xai_meta = create_xai_agreement_artifacts(
                            out_dir=out_dir, method_key=method_key,
                            shap_importance_csv=importance_csv,
                            pfi_importance_csv=(pfi_by_method.get(method_key, {}) or {}).get("importance_csv"),
                            lime_local_csv=(lime_by_method.get(method_key, {}) or {}).get("local_csv"),
                            generate_figures=bool(generate_figures),
                        )
                        xai_meta.update({"model": bm, "optimizer": opt_name, "method_key": method_key})
                        xai_comparisons[method_key] = xai_meta
                    except Exception as _xai_exc:
                        log(f"XAI agreement failed for {method_key}: {_xai_exc}")

        # Comparison pass — second pass ensures ordering independence
        if bool(xai_build_comparison) and "SHAP" in selected_xai:
            for bm in models_present:
                base_key = f"{bm}__None"
                if base_key not in shap_by_method:
                    continue
                hpo_candidates = [
                    k for k, v in shap_by_method.items()
                    if v.get("model") == bm and str(v.get("optimizer")) != "None"
                ]
                if not hpo_candidates:
                    continue

                def _rmse_for(mk: str) -> float:
                    opt = str(shap_by_method[mk].get("optimizer"))
                    r = metrics_df.loc[(metrics_df["model"] == bm) & (metrics_df["optimizer"] == opt), "rmse"]
                    return float(r.iloc[0]) if len(r) else float("inf")

                hpo_key = sorted(hpo_candidates, key=_rmse_for)[0]
                cmp_meta = create_shap_comparison_artifacts(bm, shap_by_method[base_key], shap_by_method[hpo_key], out_dir)
                if cmp_meta and not cmp_meta.get("error"):
                    shap_comparisons[bm] = cmp_meta
                    log(f"SHAP comparison ready for {bm}: {base_key} vs {hpo_key}")
                else:
                    err = cmp_meta.get("error", "unknown") if isinstance(cmp_meta, dict) else "unknown"
                    log(f"SHAP comparison failed for {bm}: {err}")

    except Exception as exc:
        err = make_error_payload(exc, stage="shap")
        log(f"SHAP pipeline failed: {err.get('error')}: {err.get('message')}")

    return shap_files, shap_by_method, shap_comparisons, pfi_by_method, lime_by_method, xai_comparisons


# -----------------------------
# Main experiment
# -----------------------------
def run_experiment(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    *,
    date_column: Optional[str],
    datetime_format: Optional[str] = None,
    folds: int,
    seed: int,
    iters: int,
    models_selected: List[str],
    optimizers_selected: List[str],
    include_cyclical: bool = False,
    time_series_cv: bool = True,
    nested_cv: bool = False,
    strict_time_order: bool = False,
    deterministic: bool = False,
    generate_figures: bool = True,
    compute_diagnostics: bool = True,
    write_metadata: bool = True,
    with_shap: bool = True,
    xai_methods: Optional[Any] = None,
    xai_build_comparison: bool = True,
    shap_rows: int = 800,
    progress_cb: Optional[ProgressCallback] = None,
    log_cb: Optional[LogCallback] = None,
    out_dir: str = "ard_outputs",
    fixed_out_dir: Optional[str] = None,
    ui_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Orchestrate a full ARD experiment run.

    This function is intentionally thin — it delegates each concern to a
    dedicated sub-function so that every stage is independently testable:

        _prepare_dataset()        — sort, cyclical features, X/y extraction
        _run_cv_loop()            — model × optimizer CV matrix
        _build_ensemble_comparison() — top-N mean ensembles
        _write_run_artifacts()    — CSV / JSON disk writes
        _run_shap_pipeline()      — full-data retrain + SHAP + comparison
        _generate_standard_figures() — all matplotlib/plotly figures
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    def prog(p: float, msg: Optional[str] = None):
        if progress_cb:
            progress_cb(p, msg)

    t_all = time.time()
    _deterministic = bool(deterministic)
    if _deterministic:
        _set_best_effort_determinism(seed)

    if _IMPORT_ERRORS:
        for dep_name, dep_err in _IMPORT_ERRORS.items():
            log(f"⚠️  Optional dependency unavailable: {dep_name} — {dep_err}")

    # ── 1. Dataset preparation ─────────────────────────────────────────────
    prog(0.02, "Preparing dataset…")
    df2, y, X, effective_feature_cols = _prepare_dataset(
        df=df, feature_cols=feature_cols, target_cols=target_cols,
        date_column=date_column, datetime_format=datetime_format,
        include_cyclical=include_cyclical, time_series_cv=time_series_cv,
        strict_time_order=strict_time_order, log=log,
    )

    preprocess = make_preprocess(effective_feature_cols)
    outer_cv = TimeSeriesSplit(n_splits=folds) if time_series_cv else KFold(n_splits=folds, shuffle=True, random_state=seed)

    opt_map = {
        "None": None, "RandomSearch": RandomSearchOptimizer,
        "TPE": TPEOptimizer, "PSO": PSOOptimizer, "GA": GAOptimizer, "ABC": ABCOptimizer,
    }
    all_models = default_models(seed, _deterministic)
    _validate_selection_inputs(models_selected, optimizers_selected, all_models, opt_map)
    models = {k: v for k, v in all_models.items() if k in models_selected}

    run_id, out_dir, fig_dir = _prepare_run_artifacts(out_dir, fixed_out_dir, generate_figures, compute_diagnostics)

    # ── 2. Run metadata ────────────────────────────────────────────────────
    if write_metadata:
        try:
            _write_run_metadata_file(
                out_dir, run_id=run_id, seed=seed, folds=folds, iters=iters,
                time_series_cv=time_series_cv, nested_cv=nested_cv,
                strict_time_order=strict_time_order, generate_figures=generate_figures,
                compute_diagnostics=compute_diagnostics, with_shap=with_shap,
                shap_rows=shap_rows, date_column=date_column, datetime_format=datetime_format,
                feature_cols=effective_feature_cols, target_cols=target_cols,
                n_rows=len(df), models_selected=models_selected,
                optimizers_selected=optimizers_selected, ui_metadata=ui_metadata,
            )
        except Exception as _e:
            log(f"Run metadata could not be written: {_e}")

    # ── 3. CV loop ─────────────────────────────────────────────────────────
    prog(0.05, "Starting CV loop…")
    total_jobs = max(1, len(models) * len(optimizers_selected))
    (rows, fold_tables, predictions, best_params, best_params_report,
     histories, runtime_detail, failed_jobs) = _run_cv_loop(
        models=models, optimizers_selected=optimizers_selected, opt_map=opt_map,
        X=X, y=y, preprocess=preprocess, outer_cv=outer_cv,
        folds=folds, iters=iters, seed=seed, nested_cv=nested_cv,
        time_series_cv=time_series_cv, log=log, prog=prog, total_jobs=total_jobs,
    )

    if failed_jobs:
        pd.DataFrame(failed_jobs).to_csv(os.path.join(out_dir, "failed_jobs.csv"), index=False)
    if not rows:
        raise RuntimeError("All model/optimizer jobs failed. See failed_jobs.csv for details.")

    metrics_df = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)

    # ── 4. Artifact writes ─────────────────────────────────────────────────
    prog(0.80, "Writing artifacts…")
    _write_run_artifacts(
        out_dir=out_dir, y=y, metrics_df=metrics_df, predictions=predictions,
        best_params_report=best_params_report, fold_tables=fold_tables,
        compute_diagnostics=compute_diagnostics, log=log,
    )
    try:
        write_paper_tables(metrics_df, best_params_report, out_dir, nested_cv=nested_cv, time_series_cv=time_series_cv, with_shap=with_shap)
    except Exception as _e:
        log(f"Paper tables skipped: {_e}")
    _build_ensemble_comparison(y=y, predictions=predictions, metrics_df=metrics_df, out_dir=out_dir, log=log)

    # ── 5. Figures ─────────────────────────────────────────────────────────
    if generate_figures:
        prog(0.88, "Generating figures…")
        _generate_standard_figures(
            fig_dir=fig_dir, out_dir=out_dir, df2=df2, y=y,
            target_cols=target_cols, feature_cols=feature_cols,
            date_column=date_column, metrics_df=metrics_df,
            fold_tables=fold_tables, histories=histories,
            predictions=predictions, log=log,
        )
        try:
            date_values = df2[date_column].to_numpy() if (date_column and date_column in df2.columns) else None
            create_prediction_diagnostic_figures(
                fig_dir=fig_dir, out_dir=out_dir, y=y, predictions=predictions,
                metrics_df=metrics_df, fold_tables=fold_tables, histories=histories,
                date_values=date_values, log=log,
            )
        except Exception as _e:
            log(f"Advanced diagnostic figures skipped: {_e}")
        try:
            write_paper_figures(y, predictions, metrics_df, fold_tables, fig_dir, out_dir)
        except Exception as _e:
            log(f"Paper figures skipped: {_e}")
    else:
        log("Figure generation disabled (generate_figures=False); skipping plots.")

    # ── 6. SHAP ────────────────────────────────────────────────────────────
    shap_files: List[str] = []
    shap_by_method: Dict[str, Dict[str, Any]] = {}
    shap_comparisons: Dict[str, Dict[str, Any]] = {}
    pfi_by_method: Dict[str, Dict[str, Any]] = {}
    lime_by_method: Dict[str, Dict[str, Any]] = {}
    xai_comparisons: Dict[str, Dict[str, Any]] = {}
    if with_shap:
        prog(0.93, "Computing SHAP/PFI/LIME…")
        (shap_files, shap_by_method, shap_comparisons,
         pfi_by_method, lime_by_method, xai_comparisons) = _run_shap_pipeline(
            metrics_df=metrics_df, best_params=best_params,
            X=X, y=y, effective_feature_cols=effective_feature_cols,
            out_dir=out_dir, seed=seed, shap_rows=shap_rows,
            generate_figures=generate_figures, deterministic=_deterministic, log=log,
            xai_methods=xai_methods,
            xai_build_comparison=xai_build_comparison,
        )

    prog(1.0, f"Done in {time.time() - t_all:.1f}s")
    log(f"✅ Experiment suite completed in {time.time() - t_all:.1f}s • {len(metrics_df)} method(s) evaluated.")

    return {
        "metrics_df": metrics_df,
        "fold_tables": fold_tables,
        "predictions": predictions,
        "y_true": [float(v) for v in y],
        "best_params": best_params,
        "best_params_report": best_params_report,
        "histories": histories,
        "runtime_detail": runtime_detail,
        "failed_jobs": failed_jobs,
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "shap_files": shap_files,
        "shap_by_method": shap_by_method,
        "shap_comparisons": shap_comparisons,
        "pfi_by_method": pfi_by_method,
        "lime_by_method": lime_by_method,
        "xai_comparisons": xai_comparisons,
        "has_shap": _HAS_SHAP,
        "has_lime": _HAS_LIME,
        "has_lgbm": _HAS_LGBM,
        "has_xgb": _HAS_XGB,
        "time_axis_used": bool(date_column),
        "nested_cv_used": bool(nested_cv),
        "import_errors": dict(_IMPORT_ERRORS),
        "zone_column": None,
        "zone_values": [],
        "multi_zone_mode": False,
    }

# -----------------------------
# Safe wrapper
# -----------------------------
def run_experiment_safe(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Calls run_experiment and guarantees a dict result even on exception.

    Streamlit will never crash with an unhandled engine error.
    
    The Streamlit layer uses a few user-facing aliases (``compute_xai`` and
    ``xai_methods_selected``).  Normalize them here so all engine entry points
    keep a stable public API and multi-seed/single-seed runs receive identical
    XAI settings.
    """
    try:
        compute_xai_alias = kwargs.pop("compute_xai", None)
        xai_methods_alias = kwargs.pop("xai_methods_selected", None)
        xai_sampling_mode_alias = kwargs.pop("xai_sampling_mode", None)
        xai_sampling_rows_alias = kwargs.pop("xai_sampling_rows", None)
        xai_build_comparison_alias = kwargs.pop("xai_build_comparison", None)

        if compute_xai_alias is not None:
            kwargs["with_shap"] = bool(compute_xai_alias)
        if xai_methods_alias is not None and kwargs.get("xai_methods") is None:
            kwargs["xai_methods"] = xai_methods_alias
        if xai_build_comparison_alias is not None:
            kwargs["xai_build_comparison"] = bool(xai_build_comparison_alias)
        # ``xai_sampling_mode`` is a UI/metadata concept.  The engine only needs
        # the resolved row count, stored as ``shap_rows`` for backward
        # compatibility with existing SHAP/PFI/LIME code paths.
        if xai_sampling_rows_alias is not None and kwargs.get("shap_rows") is None:
            kwargs["shap_rows"] = int(xai_sampling_rows_alias)
        elif xai_sampling_rows_alias is not None:
            # Keep the explicit engine value when both are present, but make sure
            # user-facing aliases never leak into run_experiment_multi_seed().
            pass
        if xai_sampling_mode_alias is not None:
            ui_meta = kwargs.get("ui_metadata")
            if isinstance(ui_meta, dict):
                ui_meta.setdefault("xai_sampling_mode", xai_sampling_mode_alias)

        seeds_arg = kwargs.pop("seeds", None)
        multi_seed_flag = bool(kwargs.pop("multi_seed", False))
        parsed_seeds = parse_seed_list(seeds_arg, kwargs.get("seed", 42)) if (seeds_arg is not None or multi_seed_flag) else []
        if parsed_seeds and (multi_seed_flag or len(parsed_seeds) > 1):
            kwargs.pop("seed", None)
            kwargs["seeds"] = parsed_seeds
            res = run_experiment_multi_seed(*args, **kwargs)
        elif 'zone_column' in kwargs:
            res = run_experiment_multi_zone(*args, **kwargs)
        else:
            res = run_experiment(*args, **kwargs)
        if res is None:
            return {
                "error": "ENGINE_RETURNED_NONE",
                "message": "Engine returned no results (res=None). This usually indicates an outdated ard_engine.py.",
                "engine_file": __file__,
            }
        return res
    except Exception as exc:
        return make_error_payload(exc, stage="engine", engine_file=__file__)

