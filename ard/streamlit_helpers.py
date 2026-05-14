"""Streamlit helper functions for ARD.

These functions were extracted from ``app_streamlit.py`` so the UI entrypoint
can stay focused on composition instead of implementation details.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import io
import json
import os
import time
from typing import Any

from .constants import ALLOWED_UPLOAD_EXTENSIONS, ALLOWED_UPLOAD_MIME_HINTS, MAX_UPLOAD_MB
from .errors import format_error_result

import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    import ard_engine as eng
except Exception:
    eng = None

CACHE_TTL_SECONDS = 3600

def build_shap_comparison_if_possible(model_name, base_key, hpo_key, shap_map, shap_comparisons, out_dir):
    cmp_meta = shap_comparisons.get(model_name, {}) or {}
    if cmp_meta and cmp_meta.get("comparison_csv") and os.path.exists(cmp_meta.get("comparison_csv", "")):
        return cmp_meta
    if not base_key or not hpo_key:
        return cmp_meta
    try:
        baseline_meta = shap_map.get(base_key, {}) or {}
        hpo_meta = shap_map.get(hpo_key, {}) or {}
        generated = eng.create_shap_comparison_artifacts(model_name, baseline_meta, hpo_meta, out_dir)
        if generated and not generated.get("error") and os.path.exists(generated.get("comparison_csv", "")):
            shap_comparisons[model_name] = generated
            return generated
        return generated or cmp_meta
    except Exception as e:
        return {"model": model_name, "error": str(e)}


def get_output_dir():
    try:
        run_result = st.session_state.get("run_result")
        if isinstance(run_result, dict):
            out_dir = run_result.get("out_dir")
            if out_dir:
                return out_dir
        out_dir = st.session_state.get("out_dir")
        if out_dir:
            return out_dir
    except Exception:
        pass
    return None


def _init_process_log_state():
    if "run_logs" not in st.session_state:
        st.session_state["run_logs"] = []
    if "run_started_at" not in st.session_state:
        st.session_state["run_started_at"] = None
    if "run_in_progress" not in st.session_state:
        st.session_state["run_in_progress"] = False
    if "run_status_text" not in st.session_state:
        st.session_state["run_status_text"] = "Idle"
    if "run_log_render_nonce" not in st.session_state:
        st.session_state["run_log_render_nonce"] = 0


def _classify_log_line(line: str) -> str:
    s = str(line)
    if any(tok in s for tok in ["❌", "ERROR", "FAIL", "Traceback"]):
        return "error"
    if any(tok in s for tok in ["⚠", "WARN"]):
        return "warning"
    if any(tok in s for tok in ["✅", "DONE", "COMPLETED", "ready", "saved"]):
        return "success"
    return "info"


def _append_process_log(msg: str):
    _init_process_log_state()
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    st.session_state["run_logs"].append(line)
    st.session_state["run_logs"] = st.session_state["run_logs"][-400:]
    st.session_state["run_log_render_nonce"] += 1


def _render_process_logs(panel_placeholder, body_placeholder):
    _init_process_log_state()
    logs = st.session_state.get("run_logs", [])
    total = len(logs)
    warn_count = sum(1 for x in logs if _classify_log_line(x) == "warning")
    err_count = sum(1 for x in logs if _classify_log_line(x) == "error")
    ok_count = sum(1 for x in logs if _classify_log_line(x) == "success")
    started_at = st.session_state.get("run_started_at")
    elapsed = f"{time.time() - started_at:0.1f}s" if started_at else "-"
    status = st.session_state.get("run_status_text", "Idle")
    running = st.session_state.get("run_in_progress", False)

    with panel_placeholder.container():
        st.markdown("### Process Logs")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Status", "Running" if running else status)
        k2.metric("Elapsed", elapsed)
        k3.metric("Lines", str(total))
        k4.metric("Warnings", str(warn_count))
        k5.metric("Errors", str(err_count))
        st.caption(
            f"Live execution journal • success: {ok_count} • warnings: {warn_count} • errors: {err_count} • state: {status}"
        )

    rendered = logs[-300:]
    if rendered:
        html_lines = []
        color_map = {
            "error": "#fecaca",
            "warning": "#fde68a",
            "success": "#86efac",
            "info": "#cbd5e1",
        }
        for line in rendered:
            sev = _classify_log_line(line)
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(
                f'<div style="color:{color_map[sev]}; white-space:pre-wrap; font-family:Consolas, Menlo, monospace; font-size:0.92rem; line-height:1.5;">{safe}</div>'
            )
        body_html = "".join(html_lines)
    else:
        body_html = '<div style="color:#94a3b8; font-family:Consolas, Menlo, monospace;">No logs yet.</div>'

    body_placeholder.markdown(
        f"""
        <div style="background:linear-gradient(180deg,#06121f 0%,#0f172a 100%); border:1px solid #1e293b; border-radius:14px; padding:14px; box-shadow:0 10px 30px rgba(2,6,23,.25); max-height:520px; overflow-y:auto;">
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_process_log_controls(download_placeholder, clear_placeholder, panel_placeholder, body_placeholder):
    _init_process_log_state()
    logs = st.session_state.get("run_logs", [])

    with download_placeholder.container():
        st.download_button(
            "⬇ Download Logs",
            data="\n".join(logs).encode("utf-8"),
            file_name=f"process_logs_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_run_logs_static",
            disabled=(len(logs) == 0),
            width="stretch",
        )

    with clear_placeholder.container():
        if st.button(
            "🧹 Clear Log Panel",
            key="clear_run_logs_static",
            width="stretch",
        ):
            st.session_state["run_logs"] = []
            _render_process_logs(panel_placeholder, body_placeholder)
            st.rerun()


def render_wilcoxon_section(output_dir=None, fig_dir=None, key_prefix=None):
    # Local imports make this function resilient to stale Streamlit reload state
    # and ensure pd/px/go are always bound when the section renders.
    import pandas as pd  # noqa: F811
    try:
        import plotly.express as px  # noqa: F811
    except Exception:
        px = None
    try:
        import plotly.graph_objects as go  # noqa: F811
    except Exception:
        go = None
    st.divider()
    st.markdown("### Wilcoxon signed-rank comparison (optimizer vs None)")

    output_dir = output_dir or get_output_dir()
    if not output_dir or not os.path.exists(output_dir):
        st.info("Wilcoxon results are not available for this run.")
        return

    fig_dir = fig_dir or os.path.join(output_dir, "figures")
    if key_prefix is None:
        key_prefix = "wilcoxon_" + str(abs(hash(str(output_dir))))
    key_prefix = str(key_prefix)
    wilcoxon_csv = os.path.join(output_dir, "wilcoxon_results.csv")
    wilcoxon_detail_csv = os.path.join(output_dir, "wilcoxon_fold_details.csv")

    wilcoxon_files = {
        "distribution": sorted(glob.glob(os.path.join(fig_dir, "wilcoxon_rmse_diffs_*.png"))),
        "effect": sorted(glob.glob(os.path.join(fig_dir, "wilcoxon_effect_sizes_*.png"))),
        "pvalues": sorted(glob.glob(os.path.join(fig_dir, "wilcoxon_pvalues_*.png"))),
        "heatmap": sorted(glob.glob(os.path.join(fig_dir, "wilcoxon_fold_heatmap_*.png"))),
    }

    summary_df = pd.read_csv(wilcoxon_csv) if os.path.exists(wilcoxon_csv) else pd.DataFrame()
    detail_df = pd.read_csv(wilcoxon_detail_csv) if os.path.exists(wilcoxon_detail_csv) else pd.DataFrame()

    has_any_wilcoxon = any(wilcoxon_files.values()) or not summary_df.empty
    if not has_any_wilcoxon:
        st.info("No Wilcoxon results found for this run.")
        return

    if px is None or go is None:
        st.warning("Plotly is not available, so Wilcoxon charts cannot be rendered in this session.")
        return

    tab_w0, tab_w1, tab_w2, tab_w3, tab_w4 = st.tabs([
        "🏛 Executive Overview",
        "📊 Significance Dashboard",
        "📈 Effect Size Gallery",
        "🧬 Fold-Level Forensics",
        "🖼 Artifact Library",
    ])

    with tab_w0:
        if not summary_df.empty:
            dfw = summary_df.copy()
            p_sig_col = "p_value_holm" if "p_value_holm" in dfw.columns else "p_value"
            dfw["significant"] = np.where(
                pd.to_numeric(dfw[p_sig_col], errors="coerce") <= 0.05,
                "Significant",
                "Not Significant",
            )
            dfw["mean_improvement"] = pd.to_numeric(dfw["mean_improvement"], errors="coerce")
            dfw["p_value"] = pd.to_numeric(dfw["p_value"], errors="coerce")
            if "effect_size_r" in dfw.columns:
                dfw["effect_size_r"] = pd.to_numeric(dfw["effect_size_r"], errors="coerce")
            else:
                dfw["effect_size_r"] = np.nan

            n_col = "n_pairs" if "n_pairs" in dfw.columns else ("n_folds" if "n_folds" in dfw.columns else None)
            pair_scope_col = "pair_scope" if "pair_scope" in dfw.columns else None
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Comparisons", f"{len(dfw)}")
            m2.metric("Significant pairs", f"{int((dfw['significant'] == 'Significant').sum())}")
            m3.metric("Best mean gain", f"{dfw['mean_improvement'].max():.4f}")
            m4.metric("Lowest raw p-value", f"{dfw['p_value'].min():.4g}")
            if n_col is not None:
                m5.metric("Max paired n", f"{int(pd.to_numeric(dfw[n_col], errors='coerce').max())}")
            scope_txt = ", ".join(sorted(dfw[pair_scope_col].dropna().astype(str).unique().tolist())) if pair_scope_col is not None else "outer_fold"
            st.caption(f"Paired-observation scope: {scope_txt}. Significance uses Holm-adjusted p-values when available.")

            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                size_vals = pd.to_numeric(dfw.get("n_pairs", dfw.get("n_folds", 3)), errors="coerce").fillna(3).clip(lower=3)
                fig = px.scatter(
                    dfw.sort_values(["model", "optimizer"]),
                    x="effect_size_r",
                    y="mean_improvement",
                    color="significant",
                    symbol="model",
                    size=size_vals,
                    hover_data=[c for c in ["model", "optimizer", "p_value", "p_value_holm", "n_pairs", "wins", "losses", "ties", "pair_scope"] if c in dfw.columns],
                    template="plotly_white",
                    height=460,
                    labels={
                        "effect_size_r": "Effect size (r)",
                        "mean_improvement": "Mean RMSE improvement vs None",
                    },
                    title="Optimizer impact map",
                )
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")

            with c2:
                top_df = dfw.sort_values(["mean_improvement", "p_value"], ascending=[False, True]).copy()
                top_df["comparison"] = top_df["model"].astype(str) + " • " + top_df["optimizer"].astype(str)
                fig = px.bar(
                    top_df.head(12),
                    x="mean_improvement",
                    y="comparison",
                    orientation="h",
                    color="significant",
                    template="plotly_white",
                    height=460,
                    title="Top optimizer lifts",
                )
                fig.add_vline(x=0, line_dash="dash")
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")

            show_cols = [
                c for c in [
                    "model", "optimizer", "pair_scope", "n_pairs", "n_nonzero",
                    "mean_improvement", "median_improvement",
                    "p_value", "p_value_holm", "effect_size_r", "wins", "losses", "ties",
                    "ci_low", "ci_high"
                ] if c in dfw.columns
            ]
            st.dataframe(
                dfw[show_cols].sort_values(["model", "p_value", "mean_improvement"], ascending=[True, True, False]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Wilcoxon summary table not found for this run.")

    with tab_w1:
        if not summary_df.empty:
            model_options = sorted(summary_df["model"].dropna().astype(str).unique().tolist())
            selected_model = st.selectbox("Model for significance analysis", model_options, key=f"{key_prefix}_significance_model")
            sdf = summary_df[summary_df["model"].astype(str) == selected_model].copy()
            p_sig_col = "p_value_holm" if "p_value_holm" in sdf.columns else "p_value"
            sdf["p_value"] = pd.to_numeric(sdf["p_value"], errors="coerce")
            if "p_value_holm" in sdf.columns:
                sdf["p_value_holm"] = pd.to_numeric(sdf["p_value_holm"], errors="coerce")
            sdf["neglog10_p"] = -np.log10(np.clip(pd.to_numeric(sdf[p_sig_col], errors="coerce"), 1e-12, 1.0))
            sdf["significant"] = np.where(pd.to_numeric(sdf[p_sig_col], errors="coerce") <= 0.05, "adjusted p ≤ 0.05", "adjusted p > 0.05")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    sdf.sort_values(p_sig_col, ascending=True),
                    x=p_sig_col,
                    y="optimizer",
                    orientation="h",
                    color="significant",
                    template="plotly_white",
                    height=440,
                    title=f"Wilcoxon p-values — {selected_model}",
                )
                fig.add_vline(x=0.05, line_dash="dash")
                fig.update_xaxes(title_text=("Holm-adjusted p-value" if p_sig_col == "p_value_holm" else "Wilcoxon p-value"))
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")

            with c2:
                fig = px.bar(
                    sdf.sort_values("neglog10_p", ascending=False),
                    x="neglog10_p",
                    y="optimizer",
                    orientation="h",
                    color="significant",
                    template="plotly_white",
                    height=440,
                    title=f"Significance strength (−log10 p) — {selected_model}",
                )
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")
        else:
            st.info("No Wilcoxon summary data available.")

    with tab_w2:
        if not summary_df.empty:
            model_options = sorted(summary_df["model"].dropna().astype(str).unique().tolist())
            selected_model = st.selectbox("Model for effect-size analysis", model_options, key=f"{key_prefix}_effect_model")
            sdf = summary_df[summary_df["model"].astype(str) == selected_model].copy()

            numeric_cols = ["mean_improvement", "ci_low", "ci_high", "effect_size_r", "win_rate"]
            for col in numeric_cols:
                if col in sdf.columns:
                    sdf[col] = pd.to_numeric(sdf[col], errors="coerce")

            c1, c2 = st.columns(2)
            with c1:
                sdf = sdf.sort_values("mean_improvement", ascending=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=sdf["mean_improvement"],
                    y=sdf["optimizer"],
                    orientation="h",
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=((sdf["ci_high"] - sdf["mean_improvement"]).clip(lower=0) if "ci_high" in sdf.columns else None),
                        arrayminus=((sdf["mean_improvement"] - sdf["ci_low"]).clip(lower=0) if "ci_low" in sdf.columns else None),
                    ),
                    name="Mean improvement",
                ))
                fig.add_vline(x=0, line_dash="dash")
                fig.update_layout(
                    template="plotly_white",
                    height=460,
                    title=f"Effect magnitude with 95% CI — {selected_model}",
                    xaxis_title="Mean RMSE improvement vs None",
                    yaxis_title="Optimizer",
                )
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")

            with c2:
                size_vals = pd.to_numeric(sdf.get("n_pairs", sdf.get("n_folds", 3)), errors="coerce").fillna(3).clip(lower=3)
                fig = px.scatter(
                    sdf,
                    x="effect_size_r",
                    y="win_rate",
                    size=size_vals,
                    color="mean_improvement",
                    hover_data=[c for c in ["optimizer", "p_value", "p_value_holm", "n_pairs", "wins", "losses", "pair_scope"] if c in sdf.columns],
                    template="plotly_white",
                    height=460,
                    title=f"Effect size vs fold win rate — {selected_model}",
                    labels={"effect_size_r": "Effect size (r)", "win_rate": "Fold win rate"},
                )
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")
        else:
            st.info("No effect-size data available.")

    with tab_w3:
        if not detail_df.empty:
            model_options = sorted(detail_df["model"].dropna().astype(str).unique().tolist())
            selected_model = st.selectbox("Model for fold-level analysis", model_options, key=f"{key_prefix}_fold_model")
            ddf = detail_df[detail_df["model"].astype(str) == selected_model].copy()
            ddf["improvement"] = pd.to_numeric(ddf["improvement"], errors="coerce")
            if "zone" in ddf.columns:
                ddf["fold_display"] = ddf["zone"].astype(str) + " • F" + ddf["fold"].astype(str)
                heat_df = ddf.pivot(index="optimizer", columns="fold_display", values="improvement")
                x_axis_col = "fold_display"
            else:
                heat_df = ddf.pivot(index="optimizer", columns="fold", values="improvement")
                x_axis_col = "fold"
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                fig = px.imshow(
                    heat_df,
                    text_auto=".3f",
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    origin="lower",
                    template="plotly_white",
                    height=max(420, 70 * len(heat_df.index) + 120),
                    title=f"Fold-by-fold improvement heatmap — {selected_model}",
                )
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")

            with c2:
                fig = px.line(
                    ddf,
                    x=x_axis_col,
                    y="improvement",
                    color="optimizer",
                    markers=True,
                    template="plotly_white",
                    height=max(420, 70 * ddf["optimizer"].nunique() + 120),
                    title=f"Fold trajectory by optimizer — {selected_model}",
                )
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")

            fig = px.box(
                ddf,
                x="optimizer",
                y="improvement",
                points="all",
                template="plotly_white",
                height=460,
                title=f"Fold-level improvement spread — {selected_model}",
            )
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plot_{abs(hash(fig.to_json()))}")
        else:
            st.info("Fold-level Wilcoxon detail file not found for this run.")

    with tab_w4:
        labels = {
            "distribution": "Distribution plots",
            "effect": "Effect-size charts",
            "pvalues": "P-value charts",
            "heatmap": "Fold heatmaps",
        }
        shown_any = False
        for key, file_list in wilcoxon_files.items():
            if not file_list:
                continue
            shown_any = True
            st.markdown(f"**{labels[key]}**")
            for p_w in file_list:
                name = os.path.basename(p_w).replace(".png", "").replace("wilcoxon_", "").replace("_", " ")
                with st.expander(name.title(), expanded=False):
                    st.image(p_w, width="stretch")
        if not shown_any:
            st.info("No saved Wilcoxon artifact images found.")


def _normalize_for_json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_normalize_for_json(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    return value


def _estimate_uploaded_file_size(uploaded_file) -> int:
    if uploaded_file is None:
        return 0
    try:
        return int(getattr(uploaded_file, "size", 0) or 0)
    except Exception:
        pass
    try:
        pos = uploaded_file.tell()
        uploaded_file.seek(0, os.SEEK_END)
        size = uploaded_file.tell()
        uploaded_file.seek(pos)
        return int(size)
    except Exception:
        return 0


def _validate_uploaded_file_size(uploaded_file, max_mb: int = MAX_UPLOAD_MB) -> bool:
    """Validate upload size and lightweight content-type/extension hints.

    This is not a security boundary, but it rejects common accidental uploads
    before pandas/scipy tries to parse them.
    """
    name = str(getattr(uploaded_file, "name", "") or "")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        st.error(f"Unsupported file extension '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}.")
        return False
    mime = str(getattr(uploaded_file, "type", "") or "")
    if mime and mime not in ALLOWED_UPLOAD_MIME_HINTS:
        st.warning(f"Unexpected browser MIME type for this upload: {mime}. The file will still be parsed by extension.")
    size_bytes = _estimate_uploaded_file_size(uploaded_file)
    if size_bytes and size_bytes > max_mb * 1024 * 1024:
        st.error(
            f"Uploaded file is too large: {size_bytes / (1024 * 1024):.1f} MB. "
            f"Maximum allowed size in this app is {max_mb} MB."
        )
        st.caption(
            "For larger files, reduce the dataset size or raise `server.maxUploadSize` in `.streamlit/config.toml`."
        )
        return False
    return True


def _load_data_cached(file_bytes: bytes, file_name: str, selected_delimiter=None):
    file_extension = file_name.split('.')[-1].lower()
    df = None
    detected_sep = None
    load_info = {"csv_encoding": None, "datetime_columns": [], "numeric_columns": []}

    if file_extension in ["csv", "txt"]:
        sep = selected_delimiter
        if sep is None:
            try:
                sample = file_bytes[:8192].decode("utf-8", errors="ignore")
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "	", "|"])
                sep = dialect.delimiter
            except Exception:
                sep = ","
        detected_sep = sep
        bio = io.BytesIO(file_bytes)
        df, used_encoding = eng.read_uploaded_csv_with_fallbacks(bio, sep=sep, on_bad_lines="skip")
        load_info["csv_encoding"] = used_encoding
    elif file_extension == "xlsx":
        df = pd.read_excel(io.BytesIO(file_bytes))
    elif file_extension == "mat":
        if loadmat is None:
            raise ImportError("scipy is required to read .mat files but could not be imported.")
        mat_contents = loadmat(io.BytesIO(file_bytes))
        candidates = []
        for k, v in mat_contents.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 2 and getattr(v, "dtype", None) is not None:
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
        raise ValueError(f"Unsupported file extension: {file_extension}")

    if df is None:
        return None, detected_sep, load_info

    for col in df.columns:
        if df[col].dtype == "object":
            name_hint = any(k in str(col).lower() for k in ("date", "time", "timestamp", "datetime"))
            if not name_hint:
                continue
            s = df[col].astype(str).str.strip()
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            if parsed.notna().sum() > len(df) * 0.7:
                df[col] = parsed
                load_info["datetime_columns"].append(str(col))

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if df[col].dtype == "object":
            s = df[col].astype(str).str.strip()
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() > len(df) * 0.7:
                df[col] = num
                load_info["numeric_columns"].append(str(col))

    df = df.dropna(axis=1, how="all")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, detected_sep, load_info


def _make_result_fingerprint(df: pd.DataFrame, run_config_json: str) -> str:
    """Deterministic fingerprint for (dataset, config) pair — used for session_state caching."""
    data_fp = hashlib.sha1(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]
    cfg_fp = hashlib.sha1(run_config_json.encode("utf-8")).hexdigest()[:16]
    return f"{data_fp}_{cfg_fp}"


def _run_experiment_cached(df: pd.DataFrame, run_config_json: str):
    """Streamlit cross-session cache — used ONLY when session_state cache misses.

    log_cb and progress_cb cannot be passed into @st.cache_data (not serialisable),
    so this function intentionally runs silently. Live log output is handled by the
    session_state cache path in _execute_run, which always runs with full callbacks
    on the first (uncached) call.
    """
    config = json.loads(run_config_json)
    fp = _make_result_fingerprint(df, run_config_json)
    fixed_out_dir = os.path.join("ard_outputs", f"cached_{fp}")
    kwargs = dict(config)
    kwargs["fixed_out_dir"] = fixed_out_dir
    kwargs["out_dir"] = "ard_outputs"
    kwargs["progress_cb"] = None
    kwargs["log_cb"] = None
    runner = eng.run_experiment_safe if hasattr(eng, "run_experiment_safe") else eng.run_experiment
    return runner(df=df, **kwargs)


def load_data(uploaded_file, selected_delimiter=None):
    """Read an uploaded dataset and apply lightweight, leakage-safe cleaning."""
    if uploaded_file is None:
        return None, None, {"csv_encoding": None, "datetime_columns": [], "numeric_columns": []}
    try:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        return _load_data_cached(file_bytes, uploaded_file.name, selected_delimiter)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, {"csv_encoding": None, "datetime_columns": [], "numeric_columns": []}



def render_engine_error(res: dict[str, Any]) -> None:
    """Render a structured engine error consistently in Streamlit."""
    st.error(format_error_result(res))
    engine_file = res.get("engine_file")
    if engine_file:
        st.caption(f"Engine file: {engine_file}")
    tb = res.get("traceback")
    if tb:
        with st.expander("Traceback", expanded=False):
            st.code(str(tb))
