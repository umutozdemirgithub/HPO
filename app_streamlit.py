"""Streamlit entrypoint for the Advanced Research Dashboard.

This version keeps the UI modular while preserving the richer, tab-based
visual and tabular reporting from the earlier dashboard revision.
"""

from __future__ import annotations

_IMPORT_ERRORS = {}

import glob
import importlib.util
import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", message="Calling float on a single element Series is deprecated.*", category=FutureWarning)

try:
    import plotly.express as px
except Exception as e:
    px = None
    _IMPORT_ERRORS["plotly.express"] = str(e)

try:
    import plotly.graph_objects as go
except Exception as e:
    go = None
    _IMPORT_ERRORS["plotly.graph_objects"] = str(e)

try:
    import ard_engine as eng
except Exception as e:
    eng = None
    _IMPORT_ERRORS["ard_engine"] = str(e)

from ard.constants import (
    DEFAULT_FOLDS,
    DEFAULT_ITERS,
    DEFAULT_SEED,
    DEFAULT_SHAP_ROWS,
    FIGURE_GROUPS,
    METRIC_RENAME_MAP,
)
from ard.streamlit_helpers import (
    MAX_UPLOAD_MB,
    _append_process_log,
    _init_process_log_state,
    _make_result_fingerprint,
    _normalize_for_json,
    _render_process_log_controls,
    _render_process_logs,
    _validate_uploaded_file_size,
    build_shap_comparison_if_possible,
    get_output_dir,
    load_data,
    render_wilcoxon_section,
    render_engine_error,
)

logger = logging.getLogger("Advanced Research Dashboard")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

@dataclass
class SidebarState:
    df: pd.DataFrame
    uploaded_file: Any
    load_info: Dict[str, Any]
    detected_delimiter: Optional[str]
    target_cols: List[str]
    feature_cols: List[str]
    date_column: str
    datetime_format: Optional[str]
    detected_zone_col: Optional[str]
    folds: int
    time_series_cv: bool
    nested_cv: bool
    iters: int
    seed: int
    multi_seed_enabled: bool
    seeds: List[int]
    seed_list_text: str
    use_experiment_cache: bool
    sel_models: List[str]
    selected_metric_columns: List[str]
    sel_opts: List[str]
    with_shap: bool
    xai_methods: List[str]
    xai_build_comparison: bool
    xai_sampling_mode: str
    shap_rows: int
    auto_detected_zone_candidate: Optional[str]



def _xai_sampling_config(mode: str, n_rows: int) -> Dict[str, Any]:
    """Return dynamic XAI row slider bounds/defaults for the selected mode."""
    n = max(1, int(n_rows or 1))
    presets: Dict[str, Dict[str, Any]] = {
        "fast": {
            "label": "Fast",
            "min": 25,
            "default": 200,
            "max": 500,
            "help": "Fast preview: small sample, quickest SHAP/PFI/LIME generation; best while tuning settings.",
        },
        "balanced": {
            "label": "Balanced",
            "min": 50,
            "default": int(DEFAULT_SHAP_ROWS),
            "max": 1000,
            "help": "Recommended default: good balance between runtime and stable visual explanations.",
        },
        "full": {
            "label": "Full",
            "min": 100,
            "default": max(int(DEFAULT_SHAP_ROWS), 1200),
            "max": 2000,
            "help": "Paper/final mode: richer explanations but substantially slower, especially with multi-seed.",
        },
    }
    cfg = dict(presets.get(str(mode).lower(), presets["balanced"]))
    cfg["max"] = max(1, min(int(cfg["max"]), n))
    cfg["min"] = max(1, min(int(cfg["min"]), int(cfg["max"])))
    cfg["default"] = max(int(cfg["min"]), min(int(cfg["default"]), int(cfg["max"])))
    return cfg

def setup_page() -> None:
    st.set_page_config(page_title="Advanced Research Dashboard", layout="wide", page_icon="🔬")
    st.title("🔬 Advanced Research Dashboard")
    st.markdown("##### *Integrated Framework for Performance, Optimization, and Explainable AI*")

def render_environment_diagnostics() -> None:
    if _IMPORT_ERRORS:
        with st.sidebar.expander("Environment diagnostics", expanded=False):
            for dep_name, dep_err in _IMPORT_ERRORS.items():
                st.warning(f"{dep_name}: {dep_err}")

def _guess_time_col(cols: List[str], df_ref: pd.DataFrame) -> Optional[str]:
    if not cols:
        return None
    for kw in ("date", "time", "timestamp", "datetime"):
        for c in cols:
            if kw in c.lower():
                return c
    return sorted(cols, key=lambda c: df_ref[c].notna().mean(), reverse=True)[0]

def _ensure_datetime_column(df: pd.DataFrame, date_column: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if date_column is None:
        st.error("Strict locked mode: No datetime column was detected.")
        st.stop()
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        parsed = pd.to_datetime(df[date_column].astype(str).str.strip(), errors="coerce")
        if parsed.notna().sum() <= len(df) * 0.7:
            st.error(f"Strict locked mode: '{date_column}' could not be parsed as datetime for most rows.")
            st.stop()
        df = df.copy()
        df[date_column] = parsed
    return df.sort_values(date_column).reset_index(drop=True), date_column

def load_uploaded_dataframe() -> Optional[Tuple[pd.DataFrame, Any, Optional[str], Dict[str, Any]]]:
    with st.sidebar:
        st.header("1. Data Ingestion")
        uploaded_file = st.file_uploader(f"Upload dataset (max {MAX_UPLOAD_MB} MB)", type=["csv", "txt", "xlsx", "mat"])
    if not uploaded_file:
        return None
    if not _validate_uploaded_file_size(uploaded_file):
        st.stop()
    df, detected_delimiter, load_info = load_data(uploaded_file, None)
    return df, uploaded_file, detected_delimiter, load_info

def render_sidebar_controls(
    df: pd.DataFrame,
    uploaded_file: Any,
    detected_delimiter: Optional[str],
    load_info: Dict[str, Any],
) -> SidebarState:
    with st.sidebar:
        st.success("Locked mode: TimeSeriesSplit + Nested CV enforced.")
        st.header("2. Experiment Control")
        folds = st.number_input("Cross-Validation Folds", 2, 10, DEFAULT_FOLDS)
        iters = st.number_input("Optimizer Iterations", 5, 300, DEFAULT_ITERS)
        seed = st.number_input("Random Seed", value=DEFAULT_SEED)
        multi_seed_enabled = st.checkbox("Enable multi-seed evaluation", value=False)
        default_seed_list = f"{int(seed)}, 101, 202"
        seed_list_text = st.text_input("Seed list", value=default_seed_list, disabled=not multi_seed_enabled, help="Use comma, space, or semicolon separated integers, e.g. 42, 101, 202.")
        try:
            seeds = eng.parse_seed_list(seed_list_text if multi_seed_enabled else int(seed), int(seed))
        except Exception as exc:
            st.error(f"Invalid seed list: {exc}")
            st.stop()
        if multi_seed_enabled:
            st.caption(f"Multi-seed will run {len(seeds)} repeated experiment(s): {seeds}")
        use_experiment_cache = st.checkbox("Reuse cached experiment results for identical inputs", value=True)

    all_cols = df.columns.tolist()
    default_target = ["POWER"] if "POWER" in all_cols else [all_cols[0]]
    dt_cols = df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns.tolist()
    date_column_guess = _guess_time_col(dt_cols, df)

    with st.sidebar.expander("2A. Feature Engineering Setup", expanded=True):
        target_cols = st.multiselect("Target (Y):", all_cols, default=default_target)
        df, date_column = _ensure_datetime_column(df, date_column_guess)
        # Prevent temporal leakage and dtype errors: the raw datetime column is a time axis, not an X feature.
        available_features = [c for c in all_cols if c not in target_cols and c != date_column]
        feature_cols = st.multiselect("Features (X):", available_features, default=available_features[: min(5, len(available_features))])
        auto_detected_zone_col = eng.detect_zone_column(
            df,
            target=target_cols[0] if target_cols else None,
            feature_cols=feature_cols,
            date_col=date_column,
        )
        detected_zone_col = None
        if auto_detected_zone_col and auto_detected_zone_col in df.columns and df[auto_detected_zone_col].nunique(dropna=True) > 1:
            if st.checkbox("Enable multi-zone evaluation using detected zone column", value=False):
                detected_zone_col = auto_detected_zone_col
                feature_cols = [c for c in feature_cols if c != detected_zone_col]
        st.caption(f"Detected delimiter: {detected_delimiter or 'N/A'} | Time axis: {date_column}")

    with st.sidebar.expander("2B. Model & Optimizer Selection", expanded=True):
        model_opts = ["HistGB"]
        if getattr(eng, "_HAS_LGBM", False):
            model_opts.insert(0, "LightGBM")
        if getattr(eng, "_HAS_XGB", False):
            model_opts.insert(1 if "LightGBM" in model_opts else 0, "XGBoost")
        sel_models = st.multiselect("Algorithms:", model_opts, default=model_opts)
        selected_metric_columns = st.multiselect(
            "Summary Metrics:",
            ["RMSE", "MAE", "R2", "MSE", "MedianAE", "ExplainedVar", "MAPE", "MaxError", "Bias", "NRMSE", "runtime_s"],
            default=["RMSE", "MAE", "R2", "MSE", "runtime_s"],
        ) or ["RMSE", "MAE", "R2", "runtime_s"]
        optuna_available = importlib.util.find_spec("optuna") is not None
        opt_choices = ["None", "RandomSearch", "PSO", "GA", "ABC"] + (["TPE"] if optuna_available else [])
        sel_opts = st.multiselect("Meta-Heuristic Optimizers:", opt_choices, default=["None", "GA"])

    with st.sidebar.expander("2C. Explainable AI Selection", expanded=True):
        with_shap = st.checkbox(
            "Compute XAI explanations",
            value=True,
            help="Turn this on to generate SHAP, PFI and/or LIME artifacts. Turn it off for the fastest model-comparison run.",
        )
        xai_methods = st.multiselect(
            "XAI Methods:",
            ["SHAP", "PFI", "LIME"],
            default=["SHAP", "PFI", "LIME"],
            disabled=not with_shap,
            help="Only the selected XAI methods will be computed, saved under ard_outputs, and rendered in the Explainable AI tab.",
        )
        if with_shap and not xai_methods:
            st.warning("Select at least one XAI method or disable Compute XAI.")
        xai_build_comparison = st.checkbox(
            "Build XAI method-comparison artifacts",
            value=True,
            disabled=(not with_shap) or len(xai_methods) < 2,
            help=(
                "Optional. Enable this only when you want SHAP/PFI/LIME agreement plots "
                "and normalized importance comparisons. You can compute one or more XAI "
                "methods without building cross-method comparison artifacts."
            ),
        )
        if (not with_shap) or len(xai_methods) < 2:
            xai_build_comparison = False

        mode_labels = {
            "fast": "fast — quickest preview",
            "balanced": "balanced — recommended",
            "full": "full — final/paper run",
        }
        shap_mode = st.selectbox(
            "XAI Sampling Mode",
            ["fast", "balanced", "full"],
            index=1,
            disabled=not with_shap,
            format_func=lambda m: mode_labels.get(m, m),
            help="Controls the automatic default and maximum for the XAI row sampler. The slider bounds are also capped by the dataset row count.",
        )
        sampling_cfg = _xai_sampling_config(shap_mode, len(df))
        shap_rows = st.slider(
            "XAI Sampling Rows",
            min_value=int(sampling_cfg["min"]),
            max_value=int(sampling_cfg["max"]),
            value=int(sampling_cfg["default"]),
            step=max(1, int(max(1, sampling_cfg["max"] - sampling_cfg["min"]) // 20)),
            disabled=not with_shap,
            help=str(sampling_cfg["help"]),
        )
        if with_shap:
            st.caption(
                f"Mode: {shap_mode} | Rows: {int(shap_rows):,} / {len(df):,} | "
                f"Selected XAI methods: {', '.join(xai_methods) if xai_methods else 'none'} | "
                f"Comparison: {'on' if xai_build_comparison else 'off'}"
            )
            if multi_seed_enabled and shap_mode == "full":
                st.warning("Full XAI sampling is expensive in multi-seed mode. Use balanced/fast for exploration, full only for final paper artifacts.")
            if "LIME" in xai_methods and int(shap_rows) > 1000:
                st.info("LIME can be slow with large samples. Consider balanced mode unless this is the final run.")
        else:
            xai_methods = []
            xai_build_comparison = False

    return SidebarState(
        df=df,
        uploaded_file=uploaded_file,
        load_info=load_info,
        detected_delimiter=detected_delimiter,
        target_cols=target_cols,
        feature_cols=feature_cols,
        date_column=date_column,
        datetime_format=None,
        detected_zone_col=detected_zone_col,
        folds=int(folds),
        time_series_cv=True,
        nested_cv=True,
        iters=int(iters),
        seed=int(seed),
        multi_seed_enabled=bool(multi_seed_enabled),
        seeds=list(seeds),
        seed_list_text=str(seed_list_text),
        use_experiment_cache=use_experiment_cache,
        sel_models=sel_models,
        selected_metric_columns=selected_metric_columns,
        sel_opts=sel_opts,
        with_shap=bool(with_shap and bool(xai_methods)),
        xai_methods=list(xai_methods),
        xai_build_comparison=bool(xai_build_comparison),
        xai_sampling_mode=str(shap_mode),
        shap_rows=int(shap_rows),
        auto_detected_zone_candidate=auto_detected_zone_col,
    )

def build_sidebar_run_metadata(state: SidebarState) -> Dict[str, Any]:
    return {
        "uploaded_file_name": getattr(state.uploaded_file, "name", None),
        "uploaded_file_size_bytes": int(getattr(state.uploaded_file, "size", 0) or 0),
        "dataset_shape": {"rows": int(len(state.df)), "columns": int(len(state.df.columns))},
        "csv_encoding": state.load_info.get("csv_encoding"),
        "detected_delimiter": state.detected_delimiter,
        "datetime_columns_auto": list(state.load_info.get("datetime_columns", [])),
        "numeric_columns_auto": list(state.load_info.get("numeric_columns", [])),
        "date_column": state.date_column,
        "feature_cols_selected": list(state.feature_cols),
        "target_cols": list(state.target_cols),
        "detected_zone_column": state.detected_zone_col,
        "zone_detection_candidate": state.auto_detected_zone_candidate,
        "folds": state.folds,
        "iters": state.iters,
        "seed": state.seed,
        "multi_seed_enabled": bool(state.multi_seed_enabled),
        "seed_values": list(state.seeds),
        "models_selected": list(state.sel_models),
        "optimizers_selected": list(state.sel_opts),
        "compute_xai": bool(state.with_shap),
        "xai_methods_selected": list(state.xai_methods),
        "xai_build_comparison": bool(state.xai_build_comparison),
        "xai_sampling_mode": state.xai_sampling_mode,
        "xai_sampling_rows": int(state.shap_rows),
    }

def build_run_config(state: SidebarState, sidebar_run_metadata: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "feature_cols": list(state.feature_cols),
        "target_cols": list(state.target_cols),
        "date_column": state.date_column,
        "datetime_format": state.datetime_format,
        "folds": state.folds,
        "seed": state.seed,
        "seeds": list(state.seeds),
        "multi_seed": bool(state.multi_seed_enabled),
        "iters": state.iters,
        "models_selected": list(state.sel_models),
        "optimizers_selected": list(state.sel_opts),
        "compute_xai": bool(state.with_shap),
        "xai_methods_selected": list(state.xai_methods),
        "xai_build_comparison": bool(state.xai_build_comparison),
        "xai_sampling_mode": state.xai_sampling_mode,
        "time_series_cv": state.time_series_cv,
        "nested_cv": state.nested_cv,
        "strict_time_order": True,
        "generate_figures": True,
        "compute_diagnostics": True,
        "write_metadata": True,
        "with_shap": state.with_shap,
        "xai_methods": list(state.xai_methods),
        "xai_build_comparison": bool(state.xai_build_comparison),
        "shap_rows": state.shap_rows,
        "out_dir": "ard_outputs",
        "ui_metadata": sidebar_run_metadata,
    }
    if state.detected_zone_col and state.detected_zone_col in state.df.columns and state.df[state.detected_zone_col].nunique(dropna=True) > 1:
        cfg["zone_column"] = state.detected_zone_col
    return cfg

def execute_run(df: pd.DataFrame, run_config: Dict[str, Any], use_experiment_cache: bool, ui_log=None, ui_progress=None) -> Dict[str, Any]:
    cache_json = json.dumps(_normalize_for_json(run_config), sort_keys=True)
    if use_experiment_cache:
        fingerprint = _make_result_fingerprint(df, cache_json)
        ss_key = f"_ard_result_{fingerprint}"
        if ss_key in st.session_state:
            if ui_log:
                ui_log("Session cache hit.")
            return dict(st.session_state[ss_key])
    runner = eng.run_experiment_safe if hasattr(eng, "run_experiment_safe") else eng.run_experiment
    result = runner(df=df, progress_cb=ui_progress, log_cb=ui_log, **run_config)
    result = dict(result) if isinstance(result, dict) else {"error": "INVALID_RESULT", "message": "Engine returned a non-dict result."}
    if use_experiment_cache and "error" not in result:
        st.session_state[ss_key] = result
    return result

def render_process_console() -> Tuple[Any, Any, Any]:
    _init_process_log_state()
    panel = st.empty()
    c1, c2 = st.columns([1, 1])
    with c1:
        download_placeholder = st.empty()
    with c2:
        clear_placeholder = st.empty()
    body = st.empty()
    _render_process_logs(panel, body)
    _render_process_log_controls(download_placeholder, clear_placeholder, panel, body)
    return panel, body, st.empty()

def run_experiment_from_ui(state: SidebarState) -> Optional[Dict[str, Any]]:
    run_btn = st.sidebar.button("🚀 EXECUTE EXPERIMENT", type="primary", width="stretch")
    panel, body, status_placeholder = render_process_console()
    if not run_btn:
        return st.session_state.get("run_result")
    progress_bar = st.progress(0)
    st.session_state["run_logs"] = []
    st.session_state["run_started_at"] = time.time()
    st.session_state["run_in_progress"] = True

    def ui_progress(p: float, msg: Optional[str] = None) -> None:
        progress_bar.progress(int(p * 100))
        if msg:
            status_placeholder.info(f"Status: {msg}")

    def ui_log(msg: str) -> None:
        _append_process_log(str(msg))
        _render_process_logs(panel, body)

    result = execute_run(
        state.df,
        build_run_config(state, build_sidebar_run_metadata(state)),
        state.use_experiment_cache,
        ui_log=ui_log,
        ui_progress=ui_progress,
    )
    st.session_state["run_in_progress"] = False
    st.session_state["run_result"] = result
    st.session_state["out_dir"] = result.get("out_dir")
    return result

def _safe_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if obj is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()

def _prepare_metrics_display(metrics_df: pd.DataFrame, selected_metric_columns: Sequence[str]) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    display_df = metrics_df.copy()
    for src, dst in METRIC_RENAME_MAP.items():
        if dst not in display_df.columns and src in display_df.columns:
            display_df = display_df.rename(columns={src: dst})
    dupes = [src for src, dst in METRIC_RENAME_MAP.items() if src in display_df.columns and dst in display_df.columns]
    if dupes:
        display_df = display_df.drop(columns=dupes)
    fixed_cols = [c for c in ["zone", "model", "optimizer"] if c in display_df.columns]
    selected_existing = [c for c in selected_metric_columns if c in display_df.columns]
    if not selected_existing:
        selected_existing = [c for c in ["RMSE", "MAE", "R2", "MSE", "runtime_s"] if c in display_df.columns]
    extra_cols = [c for c in ["rank"] if c in display_df.columns]
    keep = list(dict.fromkeys(fixed_cols + selected_existing + extra_cols))
    return display_df[keep] if keep else display_df

def _dedupe_existing_paths(paths: Iterable[str]) -> List[str]:
    """Return existing paths once, preserving order.

    Several UI sections combine canonical filenames with glob patterns; this
    helper prevents the same PNG from being rendered twice when both match.
    """
    seen: set[str] = set()
    valid: List[str] = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        key = os.path.realpath(path)
        if key in seen:
            continue
        seen.add(key)
        valid.append(path)
    return valid


def _show_image_gallery(paths: Iterable[str], columns: int = 2) -> None:
    valid = _dedupe_existing_paths(paths)
    if not valid:
        st.info("No figures available in this section.")
        return
    cols = st.columns(max(1, int(columns)))
    for idx, path in enumerate(valid):
        with cols[idx % len(cols)]:
            st.markdown(f"**{os.path.basename(path)}**")
            st.image(path, width="stretch")


def _show_single_image(path: str) -> None:
    if not path or not os.path.exists(path):
        st.info("No figure available in this section.")
        return
    st.markdown(f"**{os.path.basename(path)}**")
    st.image(path, width="stretch")

def _glob_many(fig_dir: Optional[str], patterns: Sequence[str]) -> List[str]:
    if not fig_dir or not os.path.isdir(fig_dir):
        return []
    out: List[str] = []
    for pattern in patterns:
        out.extend(sorted(glob.glob(os.path.join(fig_dir, pattern))))
    return list(dict.fromkeys(out))



def _safe_filename(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_") or "method"


def _save_actual_predicted_png(dfm: pd.DataFrame, fig_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    vals = dfm[["y_true", "y_pred"]].to_numpy(dtype=float)
    min_v = float(np.nanmin(vals)); max_v = float(np.nanmax(vals))
    plt.figure(figsize=(6.8, 5.6))
    plt.scatter(dfm["y_true"].to_numpy(dtype=float), dfm["y_pred"].to_numpy(dtype=float), s=14, alpha=0.45)
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.2)
    try:
        coef = np.polyfit(dfm["y_true"].to_numpy(dtype=float), dfm["y_pred"].to_numpy(dtype=float), 1)
        xline = np.linspace(min_v, max_v, 100)
        plt.plot(xline, coef[0] * xline + coef[1], linewidth=1.1)
    except Exception:
        pass
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title)
    plt.tight_layout(); plt.savefig(fig_path, dpi=260, bbox_inches="tight"); plt.close()


def _save_temporal_png(plot_df: pd.DataFrame, fig_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure(figsize=(10.5, 4.8))
    plt.plot(plot_df["time"].to_list(), pd.to_numeric(plot_df["actual"], errors="coerce").to_numpy(dtype=float), linewidth=1.0, label="actual")
    plt.plot(plot_df["time"].to_list(), pd.to_numeric(plot_df["predicted"], errors="coerce").to_numpy(dtype=float), linewidth=1.0, label="predicted")
    plt.xlabel("Time"); plt.ylabel("Value"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(fig_path, dpi=260, bbox_inches="tight"); plt.close()


def _save_corr_heatmap_png(corr: pd.DataFrame, fig_path: str, title: str = "Feature Correlation Heatmap") -> None:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure(figsize=(max(7.0, 0.45 * len(corr.columns) + 4), max(5.8, 0.45 * len(corr.index) + 3)))
    im = plt.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Correlation")
    plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    plt.title(title); plt.tight_layout(); plt.savefig(fig_path, dpi=260, bbox_inches="tight"); plt.close()


def _load_predictions_df(res: Dict[str, Any]) -> pd.DataFrame:
    out_dir = res.get("out_dir") or st.session_state.get("out_dir")
    csv_path = os.path.join(out_dir, "oof_predictions.csv") if out_dir else None
    if csv_path and os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    y_true = res.get("y_true") or []
    preds = res.get("predictions") or {}
    if not y_true and not preds:
        return pd.DataFrame()
    df = pd.DataFrame({"y_true": y_true})
    for key, values in preds.items():
        df[key] = values
    return df

def _best_method_key(metrics_df: pd.DataFrame) -> Optional[str]:
    if metrics_df.empty:
        return None
    df = metrics_df.copy()
    if "rmse" not in df.columns and "RMSE" in df.columns:
        df["rmse"] = pd.to_numeric(df["RMSE"], errors="coerce")
    elif "rmse" in df.columns:
        df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    else:
        return None
    df = df.dropna(subset=["rmse"])
    if df.empty:
        return None
    row = df.sort_values("rmse", ascending=True).iloc[0]
    return f"{row['model']}__{row['optimizer']}"

def render_input_preview(state: SidebarState) -> None:
    st.subheader("Dataset Preview")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(state.df):,}")
    b.metric("Columns", f"{len(state.df.columns):,}")
    c.metric("Date column", state.date_column)
    st.dataframe(state.df.head(25), width="stretch")

def render_metric_summary_cards(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        return
    work = metrics_df.copy()
    if "rmse" not in work.columns and "RMSE" in work.columns:
        work["rmse"] = pd.to_numeric(work["RMSE"], errors="coerce")
    if "r2" not in work.columns and "R2" in work.columns:
        work["r2"] = pd.to_numeric(work["R2"], errors="coerce")
    best = work.sort_values([c for c in ["rmse", "mae"] if c in work.columns], ascending=True).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", str(best.get("model", "-")))
    c2.metric("Optimizer", str(best.get("optimizer", "-")))
    if "rmse" in best:
        c3.metric("RMSE", f"{float(best['rmse']):.4f}")
    if "r2" in best and pd.notna(best["r2"]):
        c4.metric("R²", f"{float(best['r2']):.4f}")

def render_metrics_and_tables_tab(res: Dict[str, Any], state: SidebarState) -> None:
    st.subheader("Quantitative Performance Summary")
    metrics_df = _safe_dataframe(res.get("metrics_df"))
    if metrics_df.empty:
        st.info("No metrics are available for this run.")
        return

    render_metric_summary_cards(metrics_df)
    display_df = _prepare_metrics_display(metrics_df, state.selected_metric_columns)

    if res.get("multi_zone_mode"):
        st.info(f"Aggregate results across zone column `{res.get('zone_column')}` • zones: {', '.join(res.get('zone_values', []))}")
        zone_metrics_df = _safe_dataframe(res.get("zone_metrics_df"))
        if not zone_metrics_df.empty:
            with st.expander("Per-zone metrics", expanded=False):
                st.dataframe(_prepare_metrics_display(zone_metrics_df, state.selected_metric_columns), width="stretch", hide_index=True)

    highlight_subset = [c for c in ["RMSE", "MAE", "MSE", "MedianAE", "MAPE", "MaxError", "NRMSE"] if c in display_df.columns]
    if highlight_subset:
        st.dataframe(display_df.style.highlight_min(axis=0, subset=highlight_subset, color="lightgreen"), width="stretch")
    else:
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Download Research Data")
    d1, d2 = st.columns(2)
    d1.download_button(
        "📂 Export Metrics (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="ard_metrics.csv",
        mime="text/csv",
        key="download_metrics_csv",
    )
    d2.download_button(
        "⚙️ Export Hyperparameters (JSON)",
        data=json.dumps(res.get("best_params", {}), indent=2, default=str).encode("utf-8"),
        file_name="ard_params.json",
        mime="application/json",
        key="download_params_json",
    )

    st.divider()
    st.subheader("Supplementary Material")
    try:
        sup_rows = []
        for model_name in ["LightGBM", "XGBoost", "HistGB"]:
            for param_name, spec in eng.default_spaces(model_name).items():
                sup_rows.append({
                    "Model": model_name,
                    "Parameter": param_name,
                    "Type": spec.kind,
                    "Low": spec.low if spec.kind != "cat" else "",
                    "High": spec.high if spec.kind != "cat" else "",
                    "Choices": ", ".join(str(c) for c in spec.choices) if spec.choices else "",
                    "Notes": "log-uniform" if spec.kind == "logfloat" else ("integer" if spec.kind == "int" else ""),
                })
        sup_df = pd.DataFrame(sup_rows)
        st.caption("Table S1 — Hyperparameter search space used in the study.")
        st.dataframe(sup_df, width="stretch", hide_index=True)
        st.download_button(
            "📋 Download Table S1 (Supplementary CSV)",
            data=sup_df.to_csv(index=False).encode("utf-8"),
            file_name="supplementary_S1_param_space.csv",
            mime="text/csv",
            key="download_supplementary_csv",
        )
    except Exception as exc:
        st.info(f"Supplementary table unavailable: {exc}")

    out_dir = res.get("out_dir") or get_output_dir()
    artifacts = []
    if out_dir and os.path.isdir(out_dir):
        for root, _, files in os.walk(out_dir):
            for name in files:
                p = os.path.join(root, name)
                artifacts.append({"artifact": os.path.relpath(p, out_dir), "size_bytes": os.path.getsize(p)})
    if artifacts:
        with st.expander("Artifact inventory", expanded=False):
            st.dataframe(pd.DataFrame(artifacts).sort_values("artifact"), width="stretch", hide_index=True)

def _run_key_scope(res: Dict[str, Any], suffix: str = "") -> str:
    base = str(res.get("out_dir") or res.get("fig_dir") or "run")
    return _safe_filename(base)[-80:] + (f"_{suffix}" if suffix else "")


def render_fidelity_subtab(res: Dict[str, Any], state: SidebarState) -> None:
    st.markdown("### 🎯 Predictive Accuracy & Model Fidelity")
    pred_df = _load_predictions_df(res)
    if pred_df.empty or "y_true" not in pred_df.columns:
        st.info("OOF predictions were not found for this run.")
        return

    candidate_cols = [c for c in pred_df.columns if c != "y_true"]
    if not candidate_cols:
        st.info("Prediction columns are not available.")
        return

    selected_methods = st.multiselect(
        "Methods to compare",
        options=candidate_cols,
        default=candidate_cols[: min(4, len(candidate_cols))],
        key=f"vis_fidelity_methods_{_run_key_scope(res)}",
    ) or candidate_cols[:1]

    # Output directory preparation
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cols = st.columns(2)
    for idx, method in enumerate(selected_methods):
        dfm = pred_df[["y_true", method]].copy().dropna()
        if dfm.empty:
            continue
        dfm = dfm.rename(columns={method: "y_pred"})
        
        with cols[idx % 2]:
            st.markdown(f"#### {method}")
            if px is not None:
                fig = px.scatter(
                    dfm, 
                    x="y_true", 
                    y="y_pred", 
                    trendline="ols", 
                    height=420, 
                    template="plotly_white"
                )
                min_v = float(np.nanmin(dfm[["y_true", "y_pred"]].to_numpy()))
                max_v = float(np.nanmax(dfm[["y_true", "y_pred"]].to_numpy()))
                fig.add_shape(
                    type="line", 
                    x0=min_v, y0=min_v, 
                    x1=max_v, y1=max_v, 
                    line=dict(dash="dash")
                )
                fig.update_layout(
                    xaxis_title="Actual", 
                    yaxis_title="Predicted",
                    title=f"Actual vs Predicted — {method}"
                )
                
                # Save a static PNG using matplotlib so Kaleido is not required.
                fig_path = os.path.join(fig_dir, f"actual_vs_predicted_{_safe_filename(method)}.png")
                try:
                    _save_actual_predicted_png(dfm, fig_path, f"Actual vs Predicted — {method}")
                    st.caption(f"✅ Saved: {os.path.basename(fig_path)}")
                except Exception:
                    pass

                st.plotly_chart(fig, width="stretch", key=f"plot_fidelity_{_run_key_scope(res)}_{idx}_{_safe_filename(method)}")

            # Metrics
            rmse = float(np.sqrt(np.mean((dfm["y_true"] - dfm["y_pred"]) ** 2)))
            mae = float(np.mean(np.abs(dfm["y_true"] - dfm["y_pred"])))
            corr = float(dfm["y_true"].corr(dfm["y_pred"])) if len(dfm) > 1 else float("nan")
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{rmse:.4f}")
            m2.metric("MAE", f"{mae:.4f}")
            m3.metric("Corr", f"{corr:.4f}" if corr == corr else "nan")

def render_temporal_subtab(res: Dict[str, Any], state: SidebarState) -> None:
    st.markdown("### 🕒 Temporal & Trend Dynamics")
    
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    temporal_paths = _glob_many(fig_dir, ["prediction_vs_time_*.png", "timeseries_power.png"])
    if temporal_paths:
        _show_image_gallery(temporal_paths, columns=2)
        return

    pred_df = _load_predictions_df(res)
    candidate_cols = [c for c in pred_df.columns if c != "y_true"]
    if pred_df.empty or not candidate_cols:
        st.info("Temporal figures are not available.")
        return

    x_axis = state.df[state.date_column].iloc[: len(pred_df)].reset_index(drop=True)
    method = st.selectbox("Method for time profile", candidate_cols, key=f"temporal_method_{_run_key_scope(res)}")
    
    plot_df = pd.DataFrame({
        "time": x_axis, 
        "actual": pred_df["y_true"], 
        "predicted": pred_df[method]
    })

    if px is not None:
        fig = px.line(
            plot_df, 
            x="time", 
            y=["actual", "predicted"], 
            height=480, 
            template="plotly_white",
            title=f"Temporal Prediction — {method}"
        )
        
        # Save a static PNG using matplotlib so Kaleido is not required.
        fig_path = os.path.join(fig_dir, f"temporal_prediction_{_safe_filename(method)}.png")
        try:
            _save_temporal_png(plot_df, fig_path, f"Temporal Prediction — {method}")
            st.caption(f"✅ Saved: {os.path.basename(fig_path)}")
        except Exception:
            pass

        st.plotly_chart(fig, width="stretch", key=f"plot_temporal_{_run_key_scope(res)}_{_safe_filename(method)}")

def render_correlation_subtab(res: Dict[str, Any], state: SidebarState) -> None:
    st.markdown("### 🔗 Correlation & Distribution")
    
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Try pre-generated figures first
    paths = _glob_many(fig_dir, ["correlation_heatmap.png", "ssrd_scatter.png", "timeseries_power.png"])
    if paths:
        _show_image_gallery(paths, columns=2)
        return

    numeric_cols = [c for c in state.feature_cols + state.target_cols 
                    if c in state.df.columns and pd.api.types.is_numeric_dtype(state.df[c])]
    
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for on-the-fly correlation analysis.")
        return

    corr = state.df[numeric_cols].corr(numeric_only=True)
    st.dataframe(corr, width="stretch")

    if px is not None:
        fig = px.imshow(
            corr, 
            text_auto=".2f", 
            aspect="auto", 
            template="plotly_white", 
            height=550,
            title="Feature Correlation Heatmap"
        )
        
        # Save a static PNG using matplotlib so Kaleido is not required.
        fig_path = os.path.join(fig_dir, "correlation_heatmap_plotly.png")
        try:
            _save_corr_heatmap_png(corr, fig_path)
            st.caption(f"✅ Saved: {os.path.basename(fig_path)}")
        except Exception:
            pass

        st.plotly_chart(fig, width="stretch", key=f"plot_corr_heatmap_{_run_key_scope(res)}")

    # SSRD vs Target scatter (if available)
    if "SSRD" in numeric_cols and state.target_cols:
        target_col = state.target_cols[0]
        fig_ssrd = px.scatter(
            state.df, 
            x="SSRD", 
            y=target_col, 
            trendline="ols",
            template="plotly_white",
            title="SSRD vs Target"
        )
        ssrd_path = os.path.join(fig_dir, "ssrd_scatter_plotly.png")
        # Interactive chart is displayed; static export is skipped here to avoid Kaleido dependency.
        st.plotly_chart(fig_ssrd, width="stretch", key=f"plot_ssrd_target_{_run_key_scope(res)}")

def render_diagnostics_subtab(res: Dict[str, Any]) -> None:
    st.markdown("### 🧪 Diagnostics")
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    diag_tabs = st.tabs(["Residuals", "Prediction", "Optimization", "Metric/Fold Heatmaps", "Multi-seed", "Wilcoxon"])
    with diag_tabs[0]:
        _show_image_gallery(_glob_many(fig_dir, ["residual_*.png", "absolute_error_*.png"]), columns=2)
    with diag_tabs[1]:
        _show_image_gallery(_glob_many(fig_dir, ["actual_vs_predicted_*.png", "prediction_vs_time_*.png", "top_methods_error_bar.png"]), columns=2)
    with diag_tabs[2]:
        _show_image_gallery(_glob_many(fig_dir, ["convergence_*.png", "optimizer_convergence_overlay.png", "pareto_frontier.png", "rmse_runtime.png", "ablation_iters_*.png"]), columns=2)
    with diag_tabs[3]:
        _show_image_gallery(_glob_many(fig_dir, ["metric_performance_heatmap.png", "fold_rmse_heatmap.png", "fold_stability_rmse_box.png"]), columns=2)
    with diag_tabs[4]:
        _show_image_gallery(_glob_many(fig_dir, ["multi_seed_*.png"]), columns=2)
        per_seed = res.get("per_seed_metrics_df")
        if isinstance(per_seed, pd.DataFrame) and not per_seed.empty:
            st.dataframe(per_seed.head(500), width="stretch", hide_index=True)
    with diag_tabs[5]:
        render_wilcoxon_section(output_dir=res.get("out_dir"), fig_dir=fig_dir, key_prefix=f"wilcoxon_{_run_key_scope(res)}")

def _render_visual_analytics_summary_for_multiseed(res: Dict[str, Any]) -> None:
    st.markdown("### 🔁 Multi-seed visual summary")
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    paths = _glob_many(fig_dir, ["multi_seed_*.png", "metric_performance_heatmap.png", "top_methods_error_bar.png"])
    if paths:
        _show_image_gallery(paths, columns=2)
    else:
        st.info("No aggregate multi-seed visual summary was found. Open a Seed tab below for per-seed prediction diagnostics.")
    per_seed = res.get("per_seed_metrics_df")
    if isinstance(per_seed, pd.DataFrame) and not per_seed.empty:
        st.markdown("#### Per-seed metrics")
        st.dataframe(per_seed.head(500), width="stretch", hide_index=True)
    # Avoid boolean evaluation of pandas DataFrames (``df_a or df_b`` raises
    # ``ValueError: The truth value of a DataFrame is ambiguous``).
    summary_obj = res.get("multi_seed_summary_df")
    summary = _safe_dataframe(summary_obj)
    if summary.empty:
        summary = _safe_dataframe(res.get("metrics_df"))
    if not summary.empty:
        st.markdown("#### Multi-seed summary")
        st.dataframe(summary.head(500), width="stretch", hide_index=True)


def _render_visual_analytics_single_run(res: Dict[str, Any], state: SidebarState) -> None:
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    st.caption(f"📁 Figures for this run are saved to: `{fig_dir}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", f"{len(state.df):,}")
    target_mean = pd.to_numeric(state.df[state.target_cols[0]], errors="coerce").mean() if state.target_cols else float("nan")
    c2.metric("Target Mean", f"{target_mean:.2f}" if target_mean == target_mean else "nan")
    c3.metric("Feature Count", f"{len(state.feature_cols)}")
    c4.metric("Selected Models", f"{len(state.sel_models)}")

    sub_tabs = st.tabs([
        "🎯 Model Fidelity (Actual vs Predicted)",
        "🕒 Temporal & Trend Dynamics",
        "🔗 Correlation & Distribution",
        "🧪 Diagnostics (OOF, Residuals, Wilcoxon, Ablation)",
    ])
    with sub_tabs[0]:
        render_fidelity_subtab(res, state)
    with sub_tabs[1]:
        render_temporal_subtab(res, state)
    with sub_tabs[2]:
        render_correlation_subtab(res, state)
    with sub_tabs[3]:
        render_diagnostics_subtab(res)


def render_visual_analytics_tab(res: Dict[str, Any], state: SidebarState) -> None:
    st.header("📈 Advanced Visual Data Profiling & Prediction Fidelity")

    if _multi_seed_seed_results(res):
        st.info("Multi-seed mode stores prediction and diagnostic figures inside each seed folder. Use the seed tabs below to inspect per-seed OOF, residual, optimization, and fold diagnostics.")
        handled = _render_multi_seed_seed_tabs(
            res,
            summary_title="Multi-seed summary",
            summary_renderer=_render_visual_analytics_summary_for_multiseed,
            seed_renderer=lambda seed_res: _render_visual_analytics_single_run(seed_res, state),
        )
        if handled:
            return

    _render_visual_analytics_single_run(res, state)


def _render_fold_stability_summary_for_multiseed(res: Dict[str, Any]) -> None:
    st.markdown("### 🔁 Multi-seed fold/stability summary")
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    paths = _glob_many(fig_dir, ["multi_seed_*.png", "fold_rmse_heatmap.png", "fold_stability_rmse_box.png"])
    if paths:
        _show_image_gallery(paths, columns=2)
    else:
        st.info("No aggregate fold-stability figure was found. Open a Seed tab below for per-seed fold tables and fold heatmaps.")
    per_seed = res.get("per_seed_metrics_df")
    if isinstance(per_seed, pd.DataFrame) and not per_seed.empty:
        st.dataframe(per_seed.head(500), width="stretch", hide_index=True)


def _render_fold_stability_single_run(res: Dict[str, Any]) -> None:
    st.subheader("Fold Stability & Cross-Validation Forensics")
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    fold_tables = res.get("fold_tables") or {}

    top_tabs = st.tabs(["Overview", "Per-method tables", "Artifacts"])
    with top_tabs[0]:
        paths = _glob_many(fig_dir, ["fold_stability_rmse_box.png", "fold_rmse_heatmap.png", "wilcoxon_fold_heatmap_*.png", "rmse_runtime.png"])
        if paths:
            _show_image_gallery(paths, columns=2)
        else:
            st.info("No fold-stability figures found for this run.")
        metrics_df = _safe_dataframe(res.get("metrics_df"))
        if not metrics_df.empty:
            view = metrics_df.copy()
            rmse_col = "rmse" if "rmse" in view.columns else ("RMSE" if "RMSE" in view.columns else None)
            if rmse_col is not None and "runtime_s" in view.columns and px is not None:
                fig = px.scatter(
                    view,
                    x="runtime_s",
                    y=rmse_col,
                    color="model" if "model" in view.columns else None,
                    symbol="optimizer" if "optimizer" in view.columns else None,
                    template="plotly_white",
                    height=440,
                    title="Runtime vs RMSE Trade-off",
                )
                st.plotly_chart(fig, width="stretch", key=f"plot_fold_runtime_rmse_{_run_key_scope(res)}")

    with top_tabs[1]:
        if not fold_tables:
            st.info("Fold tables are not available for this run.")
        else:
            method_tabs = st.tabs([k for k in fold_tables.keys()])
            for method_key, method_tab in zip(fold_tables.keys(), method_tabs):
                with method_tab:
                    ft = _safe_dataframe(fold_tables.get(method_key))
                    if ft.empty:
                        st.info("No fold table available.")
                        continue
                    st.dataframe(ft, width="stretch", hide_index=True)
                    if {"fold", "rmse"}.issubset(ft.columns) and px is not None:
                        fig = px.line(ft, x="fold", y=[c for c in ["rmse", "mae"] if c in ft.columns], markers=True, template="plotly_white", height=380)
                        st.plotly_chart(fig, width="stretch", key=f"plot_fold_line_{_run_key_scope(res)}_{_safe_filename(method_key)}")
                    if "rmse" in ft.columns:
                        st.metric("Mean RMSE", f"{pd.to_numeric(ft['rmse'], errors='coerce').mean():.4f}")

    with top_tabs[2]:
        out_dir = res.get("out_dir") or get_output_dir()
        candidates = []
        if out_dir and os.path.isdir(out_dir):
            for name in ["fold_stability.csv", "oof_predictions.csv", "metrics.csv", "pareto_frontier_points.csv"]:
                p = os.path.join(out_dir, name)
                if os.path.exists(p):
                    candidates.append(p)
        if not candidates:
            st.info("No fold-level artifact files found.")
        else:
            for pth in candidates:
                st.markdown(f"**{os.path.basename(pth)}**")
                try:
                    st.dataframe(pd.read_csv(pth).head(200), width="stretch", hide_index=True)
                except Exception:
                    st.caption("Preview unavailable.")


def render_fold_stability_tab(res: Dict[str, Any]) -> None:
    if _multi_seed_seed_results(res):
        handled = _render_multi_seed_seed_tabs(
            res,
            summary_title="Multi-seed summary",
            summary_renderer=_render_fold_stability_summary_for_multiseed,
            seed_renderer=_render_fold_stability_single_run,
        )
        if handled:
            return
    _render_fold_stability_single_run(res)


def _read_csv_preview(path: str, *, rows: int = 500) -> None:
    if os.path.exists(path):
        st.markdown(f"**{os.path.basename(path)}**")
        try:
            st.dataframe(pd.read_csv(path).head(rows), width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"Could not preview `{os.path.basename(path)}`: {exc}")
    else:
        st.info(f"`{os.path.basename(path)}` was not found.")


def _xai_method_maps(res: Dict[str, Any], family: str) -> Dict[str, Dict[str, Any]]:
    maps: Dict[str, Dict[str, Dict[str, Any]]] = {
        "PFI": res.get("pfi_by_method") or {},
        "LIME": res.get("lime_by_method") or {},
        "Agreement": res.get("xai_comparisons") or {},
    }
    return maps.get(family, {})


def _render_method_variant_tabs(
    xmap: Dict[str, Dict[str, Any]],
    *,
    family: str,
    renderer: Callable[[str, Dict[str, Any]], None],
) -> None:
    if not xmap:
        st.info(f"No {family} artifacts were produced for this run.")
        return
    model_names = sorted({str(v.get("model", "Unknown")) for v in xmap.values()})
    for model_name in model_names:
        st.markdown(f"#### {model_name}")
        method_keys = [k for k, v in xmap.items() if str(v.get("model", "Unknown")) == model_name]
        if not method_keys:
            continue
        method_tabs = st.tabs([f"{xmap[k].get('optimizer', k)}" for k in method_keys])
        for mk, tab in zip(method_keys, method_tabs):
            with tab:
                renderer(mk, xmap[mk])




def _shap_dependence_label(path: str) -> str:
    """Create a compact tab label from a saved SHAP dependence filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = "shap_dependence_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    # Expected names are often 01_FEATURE or feature names with underscores.
    parts = stem.split("_", 1)
    if parts and parts[0].isdigit() and len(parts) > 1:
        stem = parts[1]
    return stem.replace("__", "_")[:32] or os.path.basename(path)


def _render_dependence_png_tabs(dep_pngs: Sequence[str]) -> bool:
    dep_pngs = _dedupe_existing_paths(dep_pngs)
    if not dep_pngs:
        return False
    labels = [_shap_dependence_label(p) for p in dep_pngs]
    # Streamlit allows duplicate tab labels, but unique labels are clearer.
    used: Dict[str, int] = {}
    unique_labels: List[str] = []
    for label in labels:
        count = used.get(label, 0) + 1
        used[label] = count
        unique_labels.append(label if count == 1 else f"{label} #{count}")
    feature_tabs = st.tabs(unique_labels)
    for path, tab in zip(dep_pngs, feature_tabs):
        with tab:
            _show_single_image(path)
    return True


def _render_dependence_npz_tabs(npz_path: str, *, shap_key: str) -> bool:
    if not os.path.exists(npz_path):
        return False
    try:
        npz = np.load(npz_path, allow_pickle=True)
        shap_vals = np.asarray(npz["shap_values"], dtype=float)
        Xt = np.asarray(npz["data"], dtype=float)
        feat_names = [str(x) for x in npz["feature_names"].tolist()]
        if shap_vals.ndim != 2 or Xt.ndim != 2 or shap_vals.shape != Xt.shape or not feat_names:
            st.warning("SHAP NPZ has incompatible shapes for dependence plotting.")
            return False
        feature_tabs = st.tabs([f[:32] for f in feat_names])
        for idx, (feat_name, tab) in enumerate(zip(feat_names, feature_tabs)):
            with tab:
                dep_df = pd.DataFrame({"FeatureValue": Xt[:, idx], "SHAP": shap_vals[:, idx]})
                if px is not None:
                    fig = px.scatter(
                        dep_df,
                        x="FeatureValue",
                        y="SHAP",
                        template="plotly_white",
                        height=520,
                        trendline="lowess",
                        title=f"SHAP dependence — {feat_name}",
                    )
                    fig.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig, width="stretch", key=f"dep_plot_{shap_key}_{idx}")
                st.dataframe(dep_df.head(300), width="stretch", hide_index=True)
        return True
    except Exception as exc:
        st.error(f"Failed to load SHAP NPZ for dependence plotting: {exc}")
        return False

def render_shap_variant(shap_key: str, shap_meta: Dict[str, Any]) -> None:
    """Render SHAP artifacts with clear second-level tabs."""
    shap_dir = shap_meta.get("out_dir") or shap_meta.get("dir")
    if not shap_dir or not os.path.isdir(shap_dir):
        st.info("SHAP artifacts are not available for this variant.")
        return

    importance_csv = os.path.join(shap_dir, "shap_importance.csv")
    figs_dir = os.path.join(shap_dir, "figures")
    tabs = st.tabs(["🌍 Global", "📍 Local", "🔁 Dependence", "📋 Tables"])

    with tabs[0]:
        st.caption("Global explanation: feature-level contribution patterns across the sampled dataset.")
        _show_image_gallery([
            os.path.join(figs_dir, "shap_summary.png"),
            os.path.join(figs_dir, "shap_bar.png"),
        ], columns=2)

    with tabs[1]:
        st.caption("Local explanation: how individual samples move the prediction away from the baseline.")
        decision_force = [
            os.path.join(figs_dir, "shap_decision.png"),
            os.path.join(figs_dir, "shap_force.png"),
        ]
        st.markdown("**Decision / force plots**")
        _show_image_gallery(decision_force, columns=2)
        st.markdown("**Waterfall plots**")
        _show_image_gallery(sorted(glob.glob(os.path.join(figs_dir, "shap_waterfall_*.png"))), columns=2)

    with tabs[2]:
        st.caption("Dependence analysis: each available feature is shown as its own tab; no manual feature selector is needed.")
        dep_pngs = sorted(glob.glob(os.path.join(figs_dir, "shap_dependence_*.png")))
        if dep_pngs:
            _render_dependence_png_tabs(dep_pngs)
        else:
            npz_path = os.path.join(shap_dir, "shap_values.npz")
            if not _render_dependence_npz_tabs(npz_path, shap_key=shap_key):
                st.info("No dependence plots or SHAP value matrix were found for this variant.")

    with tabs[3]:
        _read_csv_preview(importance_csv, rows=500)
        npz_path = os.path.join(shap_dir, "shap_values.npz")
        if os.path.exists(npz_path):
            st.caption(f"Raw SHAP matrix archive: `{os.path.basename(npz_path)}`")


def render_shap_comparison_variant(model_name: str, cmp_meta: Dict[str, Any]) -> None:
    tabs = st.tabs(["📊 Importance comparison", "Δ Difference", "📋 Tables"])
    fig_map = cmp_meta.get("figures", {}) if isinstance(cmp_meta, dict) else {}
    with tabs[0]:
        _show_image_gallery([fig_map.get("side_by_side", "")], columns=1)
    with tabs[1]:
        _show_image_gallery([fig_map.get("delta", ""), fig_map.get("rank_change", "")], columns=2)
    with tabs[2]:
        cmp_csv = cmp_meta.get("comparison_csv") if isinstance(cmp_meta, dict) else None
        if cmp_csv:
            _read_csv_preview(cmp_csv, rows=500)
        else:
            st.info(f"No SHAP comparison table found for {model_name}.")


def render_pfi_variant(method_key: str, pfi_meta: Dict[str, Any]) -> None:
    pfi_dir = pfi_meta.get("out_dir") or pfi_meta.get("dir")
    if not pfi_dir or not os.path.isdir(pfi_dir):
        st.info("PFI artifacts are not available for this variant.")
        return
    if pfi_meta.get("error"):
        st.warning(f"PFI error: {pfi_meta.get('error')}")
    figs_dir = os.path.join(pfi_dir, "figures")
    tabs = st.tabs(["📊 Importance", "📦 Distribution", "🔁 Cumulative", "📋 Tables"])
    with tabs[0]:
        st.caption("Permutation feature importance with uncertainty from repeated shuffles.")
        _show_image_gallery([
            os.path.join(figs_dir, "pfi_bar_mean_std.png"),
            *(_glob_many(figs_dir, ["pfi_importance*.png", "*mean_std*.png"])),
        ], columns=2)
    with tabs[1]:
        st.caption("Repeat-level PFI distribution. Wider boxes indicate less stable importance estimates.")
        _show_image_gallery([
            os.path.join(figs_dir, "pfi_repeat_boxplot.png"),
            *(_glob_many(figs_dir, ["pfi_repeat*.png", "*boxplot*.png"])),
        ], columns=2)
    with tabs[2]:
        st.caption("Cumulative concentration: how many features explain most of the normalized PFI mass.")
        _show_image_gallery([
            os.path.join(figs_dir, "pfi_cumulative_importance.png"),
            *(_glob_many(figs_dir, ["pfi_cumulative*.png"])),
        ], columns=2)
    with tabs[3]:
        _read_csv_preview(os.path.join(pfi_dir, "pfi_importance.csv"), rows=500)
        _read_csv_preview(os.path.join(pfi_dir, "pfi_repeats.csv"), rows=500)


def render_lime_variant(method_key: str, lime_meta: Dict[str, Any]) -> None:
    lime_dir = lime_meta.get("out_dir") or lime_meta.get("dir")
    if not lime_dir or not os.path.isdir(lime_dir):
        st.info("LIME artifacts are not available for this variant.")
        return
    if lime_meta.get("skipped"):
        st.warning(f"LIME skipped: {lime_meta.get('reason', 'lime package unavailable')}")
    if lime_meta.get("fallback"):
        st.info(lime_meta.get("method_note", "LIME-style fallback explanations were generated because the optional lime package is unavailable."))
    if lime_meta.get("error"):
        st.warning(f"LIME error: {lime_meta.get('error')}")
    figs_dir = os.path.join(lime_dir, "figures")
    tabs = st.tabs(["📍 Local explanations", "📈 Stability", "🌐 HTML views", "📋 Tables"])
    with tabs[0]:
        _show_image_gallery(_glob_many(figs_dir, ["lime_example_*.png", "lime_fallback_local_*.png", "lime_local_*.png"]), columns=2)
    with tabs[1]:
        _show_image_gallery([
            os.path.join(figs_dir, "lime_term_stability.png"),
            *(_glob_many(figs_dir, ["*stability*.png", "*frequency*.png"])),
        ], columns=2)
    with tabs[2]:
        htmls = sorted(glob.glob(os.path.join(lime_dir, "lime_example_*.html")) + glob.glob(os.path.join(lime_dir, "lime_fallback_*.html")))
        if not htmls:
            st.info("No LIME HTML files found.")
        else:
            st.caption("HTML explanations are saved in the run folder. Open them from the artifacts directory for full interactive rendering.")
            for h in htmls:
                st.markdown(f"- `{os.path.basename(h)}`")
    with tabs[3]:
        _read_csv_preview(os.path.join(lime_dir, "lime_local_explanations.csv"), rows=500)


def render_xai_agreement_variant(method_key: str, xai_meta: Dict[str, Any]) -> None:
    xai_dir = xai_meta.get("out_dir") or xai_meta.get("dir")
    if not xai_dir or not os.path.isdir(xai_dir):
        st.info("XAI agreement artifacts are not available for this variant.")
        return
    if xai_meta.get("skipped"):
        st.warning(f"Agreement skipped: {xai_meta.get('reason', '')}")
    if xai_meta.get("error"):
        st.warning(f"Agreement error: {xai_meta.get('error')}")
    figs_dir = os.path.join(xai_dir, "figures")
    tabs = st.tabs(["🤝 Agreement", "📊 Importance comparison", "🔗 Rank correlation", "📋 Tables"])
    with tabs[0]:
        _show_image_gallery(_glob_many(figs_dir, ["xai_agreement*.png", "*agreement*.png"]), columns=2)
    with tabs[1]:
        _show_image_gallery([
            os.path.join(figs_dir, "xai_normalized_importance_comparison.png"),
            *(_glob_many(figs_dir, ["*normalized*.png", "*importance_comparison*.png", "xai_grouped*.png"])),
        ], columns=2)
    with tabs[2]:
        _show_image_gallery([
            os.path.join(figs_dir, "xai_rank_correlation_heatmap.png"),
            *(_glob_many(figs_dir, ["*rank_correlation*.png", "*correlation_heatmap*.png"])),
        ], columns=2)
    with tabs[3]:
        _read_csv_preview(os.path.join(xai_dir, "xai_agreement.csv"), rows=500)


def render_xai_family_tab(res: Dict[str, Any], family: str) -> None:
    xmap = _xai_method_maps(res, family)
    renderers = {
        "PFI": render_pfi_variant,
        "LIME": render_lime_variant,
        "Agreement": render_xai_agreement_variant,
    }
    renderer = renderers.get(family)
    if renderer is None:
        st.info(f"No renderer is configured for {family}.")
        return
    _render_method_variant_tabs(xmap, family=family, renderer=renderer)


def _render_xai_method_tabs_for_result(res: Dict[str, Any]) -> None:
    """Render the SHAP/PFI/LIME/XAI-agreement tabs for one concrete run result.

    In single-seed mode this is the top-level experiment result. In multi-seed
    mode it is one entry from ``seed_results``. Keeping this helper separate
    lets the UI add Seed-level tabs above the XAI method tabs without changing
    how individual artifacts are displayed.
    """
    xai_tabs = st.tabs(["SHAP", "PFI", "LIME", "SHAP vs PFI vs LIME"])
    with xai_tabs[0]:
        render_shap_tab(res)
    with xai_tabs[1]:
        render_xai_family_tab(res, "PFI")
    with xai_tabs[2]:
        render_xai_family_tab(res, "LIME")
    with xai_tabs[3]:
        render_xai_family_tab(res, "Agreement")


def _seed_label_from_result(seed_res: Dict[str, Any], fallback_idx: int) -> str:
    """Best-effort label for a per-seed result tab."""
    seed_value = seed_res.get("seed")
    if seed_value is None:
        try:
            meta_path = os.path.join(str(seed_res.get("out_dir", "")), "run_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                seed_value = meta.get("seed") or meta.get("random_seed")
        except Exception:
            seed_value = None
    if seed_value is None:
        out_dir = str(seed_res.get("out_dir", ""))
        m = re.search(r"seed[_-]?(\d+)", out_dir, flags=re.IGNORECASE)
        if m:
            seed_value = m.group(1)
    return f"Seed {seed_value}" if seed_value is not None else f"Seed {fallback_idx}"


def _multi_seed_seed_results(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return concrete per-seed result dictionaries for a multi-seed run."""
    if not res.get("multi_seed_mode"):
        return []
    seed_results = res.get("seed_results") or []
    return [r for r in seed_results if isinstance(r, dict) and r.get("out_dir")]


def _render_multi_seed_seed_tabs(
    res: Dict[str, Any],
    *,
    summary_title: str,
    summary_renderer: Callable[[Dict[str, Any]], None],
    seed_renderer: Callable[[Dict[str, Any]], None],
) -> bool:
    """Render a Summary tab plus one tab per seed when multi-seed results exist.

    Parent multi-seed results contain aggregate tables/figures, while most
    prediction/XAI/fold artifacts live under seed_XX subdirectories. This
    helper prevents empty parent sections by routing UI panels to the seed
    result that owns the artifacts. Returns True when it handled rendering.
    """
    seed_results = _multi_seed_seed_results(res)
    if not seed_results:
        return False
    labels = [summary_title] + [_seed_label_from_result(seed_res, i + 1) for i, seed_res in enumerate(seed_results)]
    tabs = st.tabs(labels)
    with tabs[0]:
        summary_renderer(res)
    for seed_res, tab in zip(seed_results, tabs[1:]):
        with tab:
            seed_out = seed_res.get("out_dir")
            if seed_out:
                st.caption(f"Seed output folder: `{seed_out}`")
            seed_renderer(seed_res)
    return True


def render_explainability_tab(res: Dict[str, Any]) -> None:
    st.info("Explainability outputs are saved under `shap/`, `pfi/`, `lime/`, and `xai_comparison/` inside each run folder.")

    if res.get("multi_seed_mode") and isinstance(res.get("seed_results"), list) and res.get("seed_results"):
        seed_results = [r for r in res.get("seed_results", []) if isinstance(r, dict)]
        if not seed_results:
            st.warning("Multi-seed mode completed, but no per-seed results were available for XAI display.")
            return
        seed_labels = [_seed_label_from_result(seed_res, i + 1) for i, seed_res in enumerate(seed_results)]
        seed_tabs = st.tabs(seed_labels)
        for seed_res, seed_tab in zip(seed_results, seed_tabs):
            with seed_tab:
                seed_out = seed_res.get("out_dir")
                if seed_out:
                    st.caption(f"Seed output folder: `{seed_out}`")
                _render_xai_method_tabs_for_result(seed_res)
        return

    _render_xai_method_tabs_for_result(res)


def render_shap_tab(res: Dict[str, Any]) -> None:
    st.info("Utilizing SHAP (SHapley Additive exPlanations) to decode complex algorithmic decisions into human-readable insights.")
    shap_map = res.get("shap_by_method") or {}
    if not shap_map:
        st.warning("SHAP analysis was not enabled or no SHAP data was produced for this run.")
        return

    out_dir = res.get("out_dir") or st.session_state.get("out_dir") or "ard_outputs"
    shap_comparisons = res.get("shap_comparisons") or {}
    model_names = sorted({v.get("model", "") for v in shap_map.values() if v.get("model")})
    if not model_names:
        st.info("No model-level SHAP groups were found.")
        return

    model_tabs = st.tabs([f"🔍 {m}" for m in model_names])
    for model_name, model_tab in zip(model_names, model_tabs):
        with model_tab:
            method_keys = [k for k, v in shap_map.items() if v.get("model") == model_name]
            base_key = next((k for k in method_keys if shap_map[k].get("optimizer") == "None"), None)
            cmp_meta = shap_comparisons.get(model_name, {}) or {}
            hpo_key = cmp_meta.get("hpo_key")
            if not hpo_key:
                hpo_candidates = [k for k in method_keys if shap_map[k].get("optimizer") != "None"]
                if hpo_candidates:
                    metrics_df = _safe_dataframe(res.get("metrics_df"))
                    if not metrics_df.empty and "rmse" in metrics_df.columns:
                        def _rmse_for_key(mk: str) -> float:
                            opt = str(shap_map[mk].get("optimizer"))
                            rows = metrics_df.loc[(metrics_df["model"] == model_name) & (metrics_df["optimizer"] == opt), "rmse"]
                            return float(rows.iloc[0]) if len(rows) else float("inf")
                        hpo_key = sorted(hpo_candidates, key=_rmse_for_key)[0]
                    else:
                        hpo_key = hpo_candidates[0]

            cmp_meta = build_shap_comparison_if_possible(model_name, base_key, hpo_key, shap_map, shap_comparisons, out_dir)
            method_labels: List[str] = []
            method_keys_for_tabs: List[str] = []
            for mk in method_keys:
                opt = shap_map[mk].get("optimizer", "None")
                method_labels.append(f"{model_name} ({opt})")
                method_keys_for_tabs.append(mk)
            if base_key and hpo_key and cmp_meta and not cmp_meta.get("error") and os.path.exists(cmp_meta.get("comparison_csv", "")):
                method_labels.append(f"{model_name} (Comparison)")

            inner_tabs = st.tabs(method_labels)
            for mk, inner_tab in zip(method_keys_for_tabs, inner_tabs[: len(method_keys_for_tabs)]):
                with inner_tab:
                    render_shap_variant(mk, shap_map[mk])
            if len(inner_tabs) > len(method_keys_for_tabs):
                with inner_tabs[-1]:
                    hpo_label = shap_map[hpo_key].get("optimizer") if hpo_key else "HPO"
                    st.success(f"Comparison ready: {model_name} (None) vs {model_name} ({hpo_label})")
                    render_shap_comparison_variant(model_name, cmp_meta)
            elif cmp_meta and cmp_meta.get("error"):
                st.warning(f"SHAP comparison could not be built for this run: {cmp_meta.get('error')}")

def _render_figure_gallery_single_run(res: Dict[str, Any]) -> None:
    fig_dir = res.get("fig_dir") or os.path.join(res.get("out_dir", ""), "figures")
    st.caption(f"Figure folder: `{fig_dir}`")
    subtabs = st.tabs(list(FIGURE_GROUPS.keys()))
    for title, subtab in zip(FIGURE_GROUPS.keys(), subtabs):
        with subtab:
            _show_image_gallery(_glob_many(fig_dir, FIGURE_GROUPS[title]), columns=2)


def _render_figure_gallery_summary_for_multiseed(res: Dict[str, Any]) -> None:
    st.markdown("### 🔁 Multi-seed aggregate figures")
    _render_figure_gallery_single_run(res)


def render_figure_gallery_tab(res: Dict[str, Any]) -> None:
    if _multi_seed_seed_results(res):
        handled = _render_multi_seed_seed_tabs(
            res,
            summary_title="Multi-seed summary",
            summary_renderer=_render_figure_gallery_summary_for_multiseed,
            seed_renderer=_render_figure_gallery_single_run,
        )
        if handled:
            return
    _render_figure_gallery_single_run(res)


def render_results(res: Optional[Dict[str, Any]], state: SidebarState) -> None:
    if not res:
        st.info("Run the experiment to generate results.")
        return
    if "error" in res:
        render_engine_error(res)
        return

    st.success("✅ Experiment Suite Completed.")
    tabs = st.tabs([
        "📊 Metrics & Tables",
        "📈 Visual Analytics",
        "🧩 Fold Stability",
        "🧠 Explainable AI (SHAP/LIME/PFI)",
        "🖼 Figure Gallery",
        "📦 Artifacts",
    ])
    with tabs[0]:
        render_metrics_and_tables_tab(res, state)
    with tabs[1]:
        render_visual_analytics_tab(res, state)
    with tabs[2]:
        render_fold_stability_tab(res)
    with tabs[3]:
        render_explainability_tab(res)
    with tabs[4]:
        render_figure_gallery_tab(res)
    with tabs[5]:
        out_dir = res.get("out_dir") or get_output_dir()
        if not out_dir or not os.path.isdir(out_dir):
            st.info("Artifacts are not available for this run.")
        else:
            rows = []
            for root, _, files in os.walk(out_dir):
                for name in files:
                    p = os.path.join(root, name)
                    rows.append({"artifact": os.path.relpath(p, out_dir), "size_bytes": os.path.getsize(p)})
            st.dataframe(pd.DataFrame(rows).sort_values("artifact"), width="stretch", hide_index=True)
            st.caption(f"Output directory: {out_dir}")


def main() -> None:
    setup_page()
    render_environment_diagnostics()
    if eng is None:
        st.error("ard_engine.py could not be imported.")
        st.stop()

    loaded = load_uploaded_dataframe()
    if loaded is None:
        st.info("Please upload a data file from the sidebar to start the analysis.")
        st.stop()

    df, uploaded_file, detected_delimiter, load_info = loaded
    state = render_sidebar_controls(df, uploaded_file, detected_delimiter, load_info)
    render_input_preview(state)
    result = run_experiment_from_ui(state)
    render_results(result, state)
    st.sidebar.markdown("---")
    st.sidebar.caption("ARD Framework | Built for Researchers")


if __name__ == "__main__":
    main()
