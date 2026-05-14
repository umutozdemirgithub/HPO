from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

_IMPORT_ERRORS: Dict[str, str] = {}

try:
    import shap
    _HAS_SHAP = True
except Exception as e:
    shap = None
    _HAS_SHAP = False
    _IMPORT_ERRORS["shap"] = str(e)

def compute_shap(best_pipe: Pipeline, X: pd.DataFrame, out_dir: str, max_rows: int = 800,
                 seed: int = 42, subdir: str = "", generate_figures: bool = True) -> List[str]:
    if not _HAS_SHAP:
        return []

    if subdir:
        shap_out_dir = os.path.join(out_dir, "shap", subdir)
    else:
        shap_out_dir = out_dir
    os.makedirs(shap_out_dir, exist_ok=True)

    fig_dir = os.path.join(shap_out_dir, "figures")
    if generate_figures:
        os.makedirs(fig_dir, exist_ok=True)

    n = len(X)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    k = int(min(max_rows, n))
    if k <= 0:
        return []
    idx = np.arange(n) if k == n else np.sort(rng.choice(n, size=k, replace=False))
    Xs = X.iloc[idx].copy()

    pre = best_pipe.named_steps["preprocess"]
    model = best_pipe.named_steps["model"]
    Xt = pre.transform(Xs)
    Xt_arr = np.asarray(Xt, dtype=float)

    try:
        feature_names = [str(x) for x in pre.get_feature_names_out()]
    except Exception:
        feature_names = [str(x) for x in X.columns]

    # Keep transformed samples as a DataFrame as well as a numeric array.
    # Some estimators (notably LightGBM) remember feature names during fit and
    # emit repeated sklearn warnings when SHAP calls predict with a nameless
    # ndarray. Passing a DataFrame with the transformed feature names preserves
    # the schema and keeps Streamlit/CLI logs clean without changing results.
    Xt_df = pd.DataFrame(Xt_arr, columns=feature_names, index=Xs.index)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xt_df)
        base_value = explainer.expected_value
    except Exception:
        explainer = shap.Explainer(model, Xt_df)
        exp = explainer(Xt_df)
        shap_values = exp.values
        base_value = exp.base_values

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]
    base_value_arr = np.asarray(base_value)

    np.savez_compressed(
        os.path.join(shap_out_dir, "shap_values.npz"),
        shap_values=shap_values,
        base_values=base_value_arr,
        data=Xt_arr,
        feature_names=np.array(feature_names, dtype=object),
        sample_index=idx,
    )

    files: List[str] = []
    if generate_figures:
        def _save_current(name: str, *, dpi: int = 260, bbox_inches: str = "tight") -> str:
            path = os.path.join(fig_dir, name)
            plt.tight_layout()
            plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            plt.close('all')
            files.append(path)
            return path

        try:
            plt.figure(figsize=(12, 7))
            shap.summary_plot(shap_values, Xt_df, feature_names=feature_names, show=False, max_display=min(20, len(feature_names)))
            _save_current("shap_summary.png")
        except Exception:
            plt.close('all')

        try:
            plt.figure(figsize=(11, 7))
            shap.summary_plot(shap_values, Xt_df, feature_names=feature_names, plot_type="bar", show=False, max_display=min(20, len(feature_names)))
            _save_current("shap_bar.png")
        except Exception:
            plt.close('all')

        try:
            interaction_order = np.argsort(np.mean(np.abs(shap_values), axis=0))[::-1]
            top_k = min(6, len(feature_names))
            generated = set()
            for rank, feat_idx in enumerate(interaction_order[:top_k], start=1):
                plt.figure(figsize=(8.8, 6.0))
                shap.dependence_plot(int(feat_idx), shap_values, Xt_df, feature_names=feature_names, show=False, interaction_index="auto")
                _save_current(f"shap_dependence_{rank:02d}_{feature_names[feat_idx]}.png")
                generated.add(int(feat_idx))

            # Paper-focused dependence plots: always try to create canonical SSRD/STRD figures
            # even when they are not among the top-k ranked features. Transformed names such
            # as 'num__SSRD' are matched by suffix/substring.
            for requested in ["SSRD", "STRD"]:
                matches = [i for i, name in enumerate(feature_names)
                           if str(name).upper() == requested or str(name).upper().endswith("__" + requested) or requested in str(name).upper()]
                if not matches:
                    continue
                feat_idx = int(matches[0])
                plt.figure(figsize=(8.8, 6.0))
                shap.dependence_plot(feat_idx, shap_values, Xt_df, feature_names=feature_names, show=False, interaction_index="auto")
                _save_current(f"shap_dependence_{requested}.png")
        except Exception:
            plt.close('all')

        try:
            local_strength = np.abs(shap_values).sum(axis=1)
            m = min(25, shap_values.shape[0])
            order_idx = np.argsort(local_strength)[-m:]
            plt.figure(figsize=(12, 7))
            shap.decision_plot(
                base_value_arr[0] if np.ndim(base_value_arr) else float(base_value_arr),
                shap_values[order_idx], Xt_df.iloc[order_idx],
                feature_names=feature_names, show=False
            )
            _save_current("shap_decision.png")
        except Exception:
            plt.close('all')

        try:
            i0 = int(np.argmax(np.abs(shap_values).sum(axis=1)))
            _bv = np.ravel(base_value_arr)
            base_val = float(_bv[0]) if _bv.size else 0.0
            plt.figure(figsize=(14, 4.5))
            shap.force_plot(base_val, shap_values[i0], Xt_df.iloc[i0], feature_names=feature_names, matplotlib=True, show=False)
            _save_current("shap_force.png", bbox_inches="tight")
        except Exception:
            plt.close('all')

        try:
            local_scores = np.abs(shap_values).sum(axis=1)
            top_obs = np.argsort(local_scores)[::-1][: min(3, len(local_scores))]
            for j, obs_idx in enumerate(top_obs, start=1):
                exp = shap.Explanation(values=shap_values[obs_idx],
                                       base_values=(float(np.ravel(base_value_arr)[0]) if np.ravel(base_value_arr).size else 0.0),
                                       data=Xt_df.iloc[obs_idx].to_numpy(dtype=float),
                                       feature_names=feature_names)
                plt.figure(figsize=(10.5, 6.5))
                shap.plots.waterfall(exp, max_display=min(15, len(feature_names)), show=False)
                _save_current(f"shap_waterfall_{j}.png")
        except Exception:
            plt.close('all')

    imp = np.mean(np.abs(shap_values), axis=0)
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp}) \
        .sort_values("mean_abs_shap", ascending=False) \
        .to_csv(os.path.join(shap_out_dir, "shap_importance.csv"), index=False)

    return files

def create_shap_comparison_artifacts(model_name: str, baseline_meta: Dict[str, Any], hpo_meta: Dict[str, Any], out_dir: str) -> Optional[Dict[str, Any]]:
    """Create side-by-side SHAP comparison artifacts for baseline vs HPO of the same model."""
    try:
        base_csv = baseline_meta.get("importance_csv")
        hpo_csv = hpo_meta.get("importance_csv")
        if not base_csv or not hpo_csv or not os.path.exists(base_csv) or not os.path.exists(hpo_csv):
            return None

        base_df = pd.read_csv(base_csv).rename(columns={"mean_abs_shap": "mean_abs_shap_none"})
        hpo_df = pd.read_csv(hpo_csv).rename(columns={"mean_abs_shap": "mean_abs_shap_hpo"})
        if "feature" not in base_df.columns or "feature" not in hpo_df.columns:
            return None

        cmp_df = pd.merge(base_df, hpo_df, on="feature", how="outer").fillna(0.0)
        cmp_df["rank_none"] = cmp_df["mean_abs_shap_none"].rank(method="min", ascending=False).astype(int)
        cmp_df["rank_hpo"] = cmp_df["mean_abs_shap_hpo"].rank(method="min", ascending=False).astype(int)
        cmp_df["delta_shap"] = cmp_df["mean_abs_shap_hpo"] - cmp_df["mean_abs_shap_none"]
        cmp_df["delta_rank"] = cmp_df["rank_none"] - cmp_df["rank_hpo"]
        cmp_df = cmp_df.sort_values(["delta_shap", "mean_abs_shap_hpo"], ascending=[False, False]).reset_index(drop=True)

        cmp_dir = os.path.join(out_dir, "shap_comparison", model_name)
        fig_dir = os.path.join(cmp_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        csv_path = os.path.join(cmp_dir, "shap_comparison.csv")
        cmp_df.to_csv(csv_path, index=False)

        top = cmp_df.assign(max_imp=np.maximum(cmp_df["mean_abs_shap_none"], cmp_df["mean_abs_shap_hpo"])) \
                   .sort_values("max_imp", ascending=False).head(15).sort_values("max_imp", ascending=True)

        plt.figure(figsize=(10, max(6, 0.35 * len(top) + 2)))
        y_pos = np.arange(len(top))
        width = 0.38
        plt.barh(y_pos - width/2, top["mean_abs_shap_none"], height=width, label="None")
        plt.barh(y_pos + width/2, top["mean_abs_shap_hpo"], height=width, label=hpo_meta.get("optimizer", "HPO"))
        plt.yticks(y_pos, top["feature"])
        plt.xlabel("Mean(|SHAP|)")
        plt.title(f"{model_name}: SHAP importance — None vs HPO")
        plt.legend()
        plt.tight_layout()
        side_by_side_path = os.path.join(fig_dir, "shap_side_by_side.png")
        plt.savefig(side_by_side_path, dpi=160, bbox_inches="tight")
        plt.close()

        delta_top = cmp_df.reindex(cmp_df["delta_shap"].abs().sort_values(ascending=False).index).head(15)
        delta_top = delta_top.sort_values("delta_shap", ascending=True)
        plt.figure(figsize=(10, max(6, 0.35 * len(delta_top) + 2)))
        plt.barh(delta_top["feature"], delta_top["delta_shap"])
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Δ Mean(|SHAP|) = HPO - None")
        plt.title(f"{model_name}: SHAP change after HPO")
        plt.tight_layout()
        delta_path = os.path.join(fig_dir, "shap_delta.png")
        plt.savefig(delta_path, dpi=160, bbox_inches="tight")
        plt.close()

        rank_top = cmp_df.reindex(cmp_df["delta_rank"].abs().sort_values(ascending=False).index).head(15)
        rank_top = rank_top.sort_values("delta_rank", ascending=True)
        plt.figure(figsize=(10, max(6, 0.35 * len(rank_top) + 2)))
        plt.barh(rank_top["feature"], rank_top["delta_rank"])
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Δ Rank = Rank(None) - Rank(HPO)")
        plt.title(f"{model_name}: feature rank shift after HPO")
        plt.tight_layout()
        rank_path = os.path.join(fig_dir, "shap_rank_change.png")
        plt.savefig(rank_path, dpi=160, bbox_inches="tight")
        plt.close()

        return {
            "model": model_name,
            "baseline_key": baseline_meta.get("method_key"),
            "hpo_key": hpo_meta.get("method_key"),
            "baseline_optimizer": baseline_meta.get("optimizer", "None"),
            "hpo_optimizer": hpo_meta.get("optimizer", "HPO"),
            "dir": cmp_dir,
            "comparison_csv": csv_path,
            "figures": {
                "side_by_side": side_by_side_path,
                "delta": delta_path,
                "rank_change": rank_path,
            },
        }
    except Exception as e:
        return {"model": model_name, "error": str(e)}

# -----------------------------
# Model + search spaces
# -----------------------------

