from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

_HAS_LIME = False
_LIME_IMPORT_ERROR: Optional[str] = None
try:  # optional dependency; the rest of the engine works without it
    from lime.lime_tabular import LimeTabularExplainer
    _HAS_LIME = True
except Exception as exc:  # pragma: no cover - depends on local env
    LimeTabularExplainer = None  # type: ignore
    _LIME_IMPORT_ERROR = str(exc)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))[:160]


def _sample_frame(X: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n = len(X)
    if n == 0:
        return X.copy(), np.asarray([], dtype=float), np.asarray([], dtype=int)
    k = int(min(max(1, max_rows), n))
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if k == n else np.sort(rng.choice(n, size=k, replace=False))
    return X.iloc[idx].copy(), np.asarray(y, dtype=float)[idx], idx


def create_pfi_artifacts(
    pipe: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    out_dir: str,
    method_key: str,
    max_rows: int = 800,
    seed: int = 42,
    n_repeats: int = 8,
    generate_figures: bool = True,
) -> Dict[str, Any]:
    """Create permutation feature importance artifacts for a fitted pipeline.

    The artifacts are saved under ``<out_dir>/pfi/<method_key>/`` and include:
    ``pfi_importance.csv``, ``pfi_repeats.csv``, and several PNG figures.
    """
    pfi_dir = os.path.join(out_dir, "pfi", method_key)
    fig_dir = os.path.join(pfi_dir, "figures")
    os.makedirs(pfi_dir, exist_ok=True)
    if generate_figures:
        os.makedirs(fig_dir, exist_ok=True)

    Xs, ys, sample_idx = _sample_frame(X, y, max_rows=max_rows, seed=seed)
    result: Dict[str, Any] = {"method_key": method_key, "dir": pfi_dir, "out_dir": pfi_dir, "figures": []}
    if len(Xs) < 5:
        result["error"] = "too_few_rows"
        return result

    try:
        pfi = permutation_importance(
            pipe,
            Xs,
            ys,
            scoring="neg_root_mean_squared_error",
            n_repeats=int(max(2, n_repeats)),
            random_state=int(seed),
            n_jobs=1,
        )
        names = [str(c) for c in Xs.columns]
        imp_mean = np.asarray(pfi.importances_mean, dtype=float)
        imp_std = np.asarray(pfi.importances_std, dtype=float)
        repeats = np.asarray(pfi.importances, dtype=float)

        imp_df = pd.DataFrame({
            "feature": names,
            "pfi_mean_rmse_increase": imp_mean,
            "pfi_std": imp_std,
            "pfi_rank": pd.Series(imp_mean).rank(method="min", ascending=False).astype(int),
        }).sort_values("pfi_mean_rmse_increase", ascending=False).reset_index(drop=True)
        imp_path = os.path.join(pfi_dir, "pfi_importance.csv")
        imp_df.to_csv(imp_path, index=False)
        result["importance_csv"] = imp_path

        repeat_df = pd.DataFrame(repeats.T, columns=names)
        repeat_df.insert(0, "repeat", np.arange(1, len(repeat_df) + 1))
        repeat_path = os.path.join(pfi_dir, "pfi_repeats.csv")
        repeat_df.to_csv(repeat_path, index=False)
        result["repeats_csv"] = repeat_path

        meta = {
            "method_key": method_key,
            "sample_rows": int(len(Xs)),
            "sample_index": [int(i) for i in sample_idx.tolist()],
            "n_repeats": int(n_repeats),
            "scoring": "neg_root_mean_squared_error",
        }
        with open(os.path.join(pfi_dir, "pfi_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        if generate_figures and not imp_df.empty:
            top = imp_df.head(min(20, len(imp_df))).iloc[::-1]
            plt.figure(figsize=(10, max(5.5, 0.35 * len(top) + 1.8)))
            plt.barh(top["feature"].astype(str).to_list(), pd.to_numeric(top["pfi_mean_rmse_increase"], errors="coerce").to_numpy(dtype=float), xerr=pd.to_numeric(top["pfi_std"], errors="coerce").to_numpy(dtype=float), alpha=0.9)
            plt.axvline(0.0, linestyle="--", linewidth=1.0)
            plt.xlabel("Permutation importance: RMSE increase")
            plt.title(f"PFI importance — {method_key}")
            plt.tight_layout()
            p = os.path.join(fig_dir, "pfi_bar_mean_std.png")
            plt.savefig(p, dpi=260, bbox_inches="tight")
            plt.close()
            result["figures"].append(p)

            top_features = imp_df.head(min(15, len(imp_df)))["feature"].tolist()
            box_data = [repeat_df[f].dropna().to_numpy(dtype=float) for f in top_features]
            plt.figure(figsize=(max(8, 0.55 * len(top_features) + 3), 5.2))
            plt.boxplot(box_data, tick_labels=top_features, showmeans=True)
            plt.axhline(0.0, linestyle="--", linewidth=1.0)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("RMSE increase across repeats")
            plt.title(f"PFI repeat distribution — {method_key}")
            plt.tight_layout()
            p = os.path.join(fig_dir, "pfi_repeat_boxplot.png")
            plt.savefig(p, dpi=260, bbox_inches="tight")
            plt.close()
            result["figures"].append(p)

            vals = imp_df["pfi_mean_rmse_increase"].clip(lower=0).to_numpy(dtype=float)
            denom = vals.sum()
            if denom > 0:
                cum = np.cumsum(vals) / denom
                plt.figure(figsize=(7.5, 4.6))
                plt.plot(np.arange(1, len(cum) + 1), cum, marker="o", linewidth=1.2)
                plt.axhline(0.8, linestyle="--", linewidth=1.0)
                plt.axhline(0.9, linestyle="--", linewidth=1.0)
                plt.xlabel("Top-k features")
                plt.ylabel("Cumulative normalized PFI")
                plt.title(f"PFI cumulative concentration — {method_key}")
                plt.tight_layout()
                p = os.path.join(fig_dir, "pfi_cumulative_importance.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                result["figures"].append(p)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _lime_training_matrix(X: pd.DataFrame) -> Tuple[np.ndarray, List[str], np.ndarray]:
    numeric = X.copy()
    for c in numeric.columns:
        numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
    med = numeric.median(numeric_only=True).reindex(numeric.columns).fillna(0.0)
    numeric = numeric.fillna(med)
    return numeric.to_numpy(dtype=float), [str(c) for c in numeric.columns], med.to_numpy(dtype=float)


def create_lime_artifacts(
    pipe: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    out_dir: str,
    method_key: str,
    max_rows: int = 800,
    seed: int = 42,
    n_examples: int = 3,
    num_features: int = 12,
    generate_figures: bool = True,
) -> Dict[str, Any]:
    """Create LIME local explanation artifacts for a fitted pipeline.

    LIME is optional. If the ``lime`` package is not installed, this function writes
    a metadata file explaining the skip instead of failing the experiment.
    """
    lime_dir = os.path.join(out_dir, "lime", method_key)
    fig_dir = os.path.join(lime_dir, "figures")
    os.makedirs(lime_dir, exist_ok=True)
    if generate_figures:
        os.makedirs(fig_dir, exist_ok=True)
    result: Dict[str, Any] = {"method_key": method_key, "dir": lime_dir, "out_dir": lime_dir, "figures": []}

    if not _HAS_LIME:
        # Robust fallback: create LIME-style local surrogate artifacts even when
        # the optional `lime` wheel is not installed. This is not a dependency
        # substitute for the original LIME algorithm; it is a deterministic
        # local leave-one-feature-to-baseline sensitivity explanation so the
        # Streamlit XAI panel and xai_comparison/ artifacts are still populated.
        Xs, ys, sample_idx = _sample_frame(X, y, max_rows=max_rows, seed=seed)
        if len(Xs) < 5:
            result["error"] = "too_few_rows"
            return result
        try:
            train_arr, feature_names, med = _lime_training_matrix(Xs)

            def predict_fn(arr: np.ndarray) -> np.ndarray:
                df_pred = pd.DataFrame(arr, columns=Xs.columns)
                return np.asarray(pipe.predict(df_pred), dtype=float)

            preds = predict_fn(train_arr)
            abs_err = np.abs(np.asarray(ys, dtype=float) - preds)
            candidate_positions: List[int] = []
            if len(abs_err):
                candidate_positions.extend([
                    int(np.nanargmin(abs_err)),
                    int(np.nanargmax(abs_err)),
                    int(np.argsort(abs_err)[len(abs_err)//2]),
                ])
            rng = np.random.default_rng(seed)
            while len(candidate_positions) < int(n_examples):
                candidate_positions.append(int(rng.integers(0, len(Xs))))
            seen: set[int] = set()
            positions: List[int] = []
            for pos in candidate_positions:
                if pos not in seen:
                    positions.append(pos)
                    seen.add(pos)
                if len(positions) >= int(n_examples):
                    break

            rows: List[Dict[str, Any]] = []
            for rank, pos in enumerate(positions, start=1):
                base = train_arr[pos].copy()
                pred_val = float(preds[pos]) if pos < len(preds) else float("nan")
                true_val = float(ys[pos]) if pos < len(ys) else float("nan")
                contribs: List[Tuple[str, float]] = []
                for j, feat in enumerate(feature_names):
                    pert = base.copy()
                    pert[j] = med[j]
                    pert_pred = float(predict_fn(pert.reshape(1, -1))[0])
                    # Positive means the observed feature value increases the prediction
                    # relative to a median-baseline replacement.
                    contribs.append((feat, pred_val - pert_pred))
                contribs.sort(key=lambda t: abs(t[1]), reverse=True)
                sample_id = int(sample_idx[pos]) if pos < len(sample_idx) else int(pos)
                for term_rank, (term, weight) in enumerate(contribs[: int(min(num_features, len(feature_names)))], start=1):
                    rows.append({
                        "sample_rank": rank,
                        "sample_index": sample_id,
                        "term_rank": term_rank,
                        "term": term,
                        "weight": float(weight),
                        "abs_weight": float(abs(weight)),
                        "prediction": pred_val,
                        "target": true_val,
                        "method": "fallback_local_median_sensitivity",
                    })

                if generate_figures:
                    fig_rows = pd.DataFrame(rows)
                    one = fig_rows[fig_rows["sample_rank"] == rank].sort_values("abs_weight", ascending=True)
                    if not one.empty:
                        plt.figure(figsize=(8.5, max(4.8, 0.38 * len(one) + 1.6)))
                        plt.barh(one["term"], one["weight"], alpha=0.9)
                        plt.axvline(0.0, linestyle="--", linewidth=1.0)
                        plt.xlabel("Local contribution: prediction minus median-baseline prediction")
                        plt.title(f"LIME-style fallback local explanation — {method_key} — sample {sample_id}")
                        plt.tight_layout()
                        fp = os.path.join(fig_dir, f"lime_fallback_local_{rank}_sample_{sample_id}.png")
                        plt.savefig(fp, dpi=260, bbox_inches="tight")
                        plt.close()
                        result["figures"].append(fp)

                    html_path = os.path.join(lime_dir, f"lime_fallback_local_{rank}_sample_{sample_id}.html")
                    with open(html_path, "w", encoding="utf-8") as hf:
                        hf.write("<html><body><h2>LIME-style fallback local explanation</h2>")
                        hf.write("<p>The lime package was unavailable; this deterministic fallback replaces each feature by its median baseline and measures the prediction change.</p>")
                        hf.write(pd.DataFrame(contribs[: int(min(num_features, len(feature_names)))], columns=["feature", "weight"]).to_html(index=False))
                        hf.write("</body></html>")

            local_df = pd.DataFrame(rows)
            local_path = os.path.join(lime_dir, "lime_local_explanations.csv")
            local_df.to_csv(local_path, index=False)
            result["local_csv"] = local_path
            result["fallback"] = True
            result["reason"] = _LIME_IMPORT_ERROR or "lime package is unavailable"
            result["method_note"] = "LIME-style fallback local median-sensitivity explanations were generated because the optional lime package is unavailable."

            if generate_figures and not local_df.empty:
                stab = local_df.groupby("term", as_index=False)["abs_weight"].mean().sort_values("abs_weight", ascending=False).head(min(20, local_df["term"].nunique()))
                top = stab.iloc[::-1]
                plt.figure(figsize=(9, max(4.8, 0.36 * len(top) + 1.5)))
                plt.barh(top["term"], top["abs_weight"], alpha=0.9)
                plt.xlabel("Mean |local contribution|")
                plt.title(f"LIME-style fallback term stability — {method_key}")
                plt.tight_layout()
                fp = os.path.join(fig_dir, "lime_term_stability.png")
                plt.savefig(fp, dpi=260, bbox_inches="tight")
                plt.close()
                result["figures"].append(fp)

            with open(os.path.join(lime_dir, "lime_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            with open(os.path.join(lime_dir, "lime_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
        return result

    Xs, ys, sample_idx = _sample_frame(X, y, max_rows=max_rows, seed=seed)
    if len(Xs) < 5:
        result["error"] = "too_few_rows"
        return result

    try:
        train_arr, feature_names, med = _lime_training_matrix(Xs)

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            df_pred = pd.DataFrame(arr, columns=Xs.columns)
            return np.asarray(pipe.predict(df_pred), dtype=float)

        preds = predict_fn(train_arr)
        abs_err = np.abs(np.asarray(ys, dtype=float) - preds)
        candidate_positions: List[int] = []
        if len(abs_err):
            candidate_positions.extend([int(np.nanargmin(abs_err)), int(np.nanargmax(abs_err)), int(np.argsort(abs_err)[len(abs_err)//2])])
        rng = np.random.default_rng(seed)
        while len(candidate_positions) < int(n_examples):
            candidate_positions.append(int(rng.integers(0, len(Xs))))
        # unique while preserving order
        seen: set[int] = set()
        positions = []
        for pos in candidate_positions:
            if pos not in seen:
                positions.append(pos)
                seen.add(pos)
            if len(positions) >= int(n_examples):
                break

        explainer = LimeTabularExplainer(  # type: ignore[misc]
            training_data=train_arr,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=True,
            random_state=int(seed),
        )

        rows: List[Dict[str, Any]] = []
        for rank, pos in enumerate(positions, start=1):
            exp = explainer.explain_instance(
                train_arr[pos],
                predict_fn,
                num_features=int(min(num_features, len(feature_names))),
            )
            exp_list = exp.as_list()
            sample_id = int(sample_idx[pos]) if pos < len(sample_idx) else int(pos)
            pred_val = float(preds[pos]) if pos < len(preds) else float("nan")
            true_val = float(ys[pos]) if pos < len(ys) else float("nan")
            for term_rank, (term, weight) in enumerate(exp_list, start=1):
                rows.append({
                    "example_rank": rank,
                    "sample_index": sample_id,
                    "term_rank": term_rank,
                    "term": str(term),
                    "weight": float(weight),
                    "prediction": pred_val,
                    "y_true": true_val,
                    "abs_error": abs(true_val - pred_val) if np.isfinite(true_val) and np.isfinite(pred_val) else float("nan"),
                })
            try:
                html_path = os.path.join(lime_dir, f"lime_example_{rank:02d}_sample_{sample_id}.html")
                exp.save_to_file(html_path)
            except Exception:
                pass

            if generate_figures and exp_list:
                terms = [str(t) for t, _ in exp_list][::-1]
                weights = np.asarray([float(w) for _, w in exp_list], dtype=float)[::-1]
                plt.figure(figsize=(10, max(4.8, 0.38 * len(terms) + 1.6)))
                plt.barh(terms, weights, alpha=0.9)
                plt.axvline(0.0, linestyle="--", linewidth=1.0)
                plt.xlabel("LIME local contribution")
                plt.title(f"LIME local explanation — {method_key} — sample {sample_id}")
                plt.tight_layout()
                p = os.path.join(fig_dir, f"lime_example_{rank:02d}_sample_{sample_id}.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                result["figures"].append(p)

        df_rows = pd.DataFrame(rows)
        csv_path = os.path.join(lime_dir, "lime_local_explanations.csv")
        df_rows.to_csv(csv_path, index=False)
        result["local_csv"] = csv_path

        if generate_figures and not df_rows.empty:
            freq = (
                df_rows.assign(abs_weight=lambda d: d["weight"].abs())
                .groupby("term", as_index=False)
                .agg(mean_abs_weight=("abs_weight", "mean"), frequency=("term", "size"))
                .sort_values(["frequency", "mean_abs_weight"], ascending=[False, False])
                .head(20)
                .iloc[::-1]
            )
            plt.figure(figsize=(10, max(5.2, 0.35 * len(freq) + 1.8)))
            plt.barh(freq["term"], freq["mean_abs_weight"], alpha=0.9)
            plt.xlabel("Mean |LIME contribution|")
            plt.title(f"LIME term stability across selected examples — {method_key}")
            plt.tight_layout()
            p = os.path.join(fig_dir, "lime_term_stability.png")
            plt.savefig(p, dpi=260, bbox_inches="tight")
            plt.close()
            result["figures"].append(p)

        with open(os.path.join(lime_dir, "lime_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"method_key": method_key, "sample_rows": int(len(Xs)), "selected_positions": positions, "sample_indices": [int(sample_idx[p]) for p in positions]}, f, indent=2, default=str)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def create_xai_agreement_artifacts(
    *,
    out_dir: str,
    method_key: str,
    shap_importance_csv: Optional[str] = None,
    pfi_importance_csv: Optional[str] = None,
    lime_local_csv: Optional[str] = None,
    generate_figures: bool = True,
) -> Dict[str, Any]:
    """Compare SHAP, PFI, and LIME importance signals for one method."""
    agree_dir = os.path.join(out_dir, "xai_comparison", method_key)
    fig_dir = os.path.join(agree_dir, "figures")
    os.makedirs(agree_dir, exist_ok=True)
    if generate_figures:
        os.makedirs(fig_dir, exist_ok=True)
    result: Dict[str, Any] = {"method_key": method_key, "dir": agree_dir, "out_dir": agree_dir, "figures": []}

    frames: List[pd.DataFrame] = []
    try:
        if shap_importance_csv and os.path.exists(shap_importance_csv):
            s = pd.read_csv(shap_importance_csv)
            if {"feature", "mean_abs_shap"}.issubset(s.columns):
                frames.append(s[["feature", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "SHAP"}))
        if pfi_importance_csv and os.path.exists(pfi_importance_csv):
            p = pd.read_csv(pfi_importance_csv)
            if {"feature", "pfi_mean_rmse_increase"}.issubset(p.columns):
                frames.append(p[["feature", "pfi_mean_rmse_increase"]].rename(columns={"pfi_mean_rmse_increase": "PFI"}))
        if lime_local_csv and os.path.exists(lime_local_csv):
            l = pd.read_csv(lime_local_csv)
            if {"term", "weight"}.issubset(l.columns):
                # LIME terms may be discretized intervals; keep the term label.
                lime_imp = l.assign(abs_weight=lambda d: d["weight"].abs()).groupby("term", as_index=False)["abs_weight"].mean()
                lime_imp = lime_imp.rename(columns={"term": "feature", "abs_weight": "LIME"})
                frames.append(lime_imp)
        if not frames:
            result["skipped"] = True
            result["reason"] = "no comparable XAI importance files found"
            return result

        merged = frames[0]
        for fr in frames[1:]:
            merged = pd.merge(merged, fr, on="feature", how="outer")
        score_cols = [c for c in ["SHAP", "PFI", "LIME"] if c in merged.columns]
        merged[score_cols] = merged[score_cols].fillna(0.0)
        for col in score_cols:
            vals = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
            max_val = float(vals.abs().max())
            merged[f"{col}_norm"] = vals / max_val if max_val > 0 else vals
            merged[f"{col}_rank"] = vals.rank(method="min", ascending=False).astype(int)
        norm_cols = [f"{c}_norm" for c in score_cols]
        merged["agreement_score"] = merged[norm_cols].mean(axis=1) if norm_cols else 0.0
        rank_cols = [f"{c}_rank" for c in score_cols]
        merged["rank_std"] = merged[rank_cols].std(axis=1) if len(rank_cols) > 1 else 0.0
        merged = merged.sort_values("agreement_score", ascending=False).reset_index(drop=True)
        csv_path = os.path.join(agree_dir, "xai_agreement.csv")
        merged.to_csv(csv_path, index=False)
        result["agreement_csv"] = csv_path

        if generate_figures and not merged.empty:
            top = merged.head(min(20, len(merged))).iloc[::-1]
            y_pos = np.arange(len(top))
            width = 0.8 / max(1, len(score_cols))
            plt.figure(figsize=(11, max(5.5, 0.38 * len(top) + 2.0)))
            for i, col in enumerate(score_cols):
                plt.barh(y_pos + (i - (len(score_cols)-1)/2) * width, pd.to_numeric(top[f"{col}_norm"], errors="coerce").to_numpy(dtype=float), height=width, label=col, alpha=0.9)
            plt.yticks(y_pos, top["feature"].astype(str).to_list())
            plt.xlabel("Normalized importance")
            plt.title(f"XAI agreement: SHAP vs PFI vs LIME — {method_key}")
            plt.legend()
            plt.tight_layout()
            p = os.path.join(fig_dir, "xai_agreement_grouped_bar.png")
            plt.savefig(p, dpi=260, bbox_inches="tight")
            # Alias with explicit paper-friendly name requested by users.
            p_alias = os.path.join(fig_dir, "xai_normalized_importance_comparison.png")
            plt.savefig(p_alias, dpi=260, bbox_inches="tight")
            plt.close()
            result["figures"].extend([p, p_alias])

            if len(score_cols) >= 2:
                mat = merged[rank_cols].corr(method="spearman").to_numpy(dtype=float)
                plt.figure(figsize=(5.8, 5.0))
                im = plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
                labels = [c.replace("_rank", "") for c in rank_cols]
                plt.xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
                plt.yticks(np.arange(len(labels)), labels)
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
                plt.colorbar(im, label="Spearman rank correlation")
                plt.title(f"XAI rank agreement — {method_key}")
                plt.tight_layout()
                p = os.path.join(fig_dir, "xai_rank_correlation_heatmap.png")
                plt.savefig(p, dpi=260, bbox_inches="tight")
                plt.close()
                result["figures"].append(p)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


__all__ = [
    "_HAS_LIME",
    "_LIME_IMPORT_ERROR",
    "create_pfi_artifacts",
    "create_lime_artifacts",
    "create_xai_agreement_artifacts",
]
