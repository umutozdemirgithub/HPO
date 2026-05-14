from __future__ import annotations

import os
import random
from typing import Any, List, Optional, Tuple

from .constants import ZONE_CANDIDATES

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _set_best_effort_determinism(seed: int) -> None:
    """Best-effort determinism."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def make_preprocess(feature_cols: List[str], scale: bool = False) -> ColumnTransformer:
    """Preprocessing pipeline that preserves feature names (fixes LightGBM warnings)."""
    set_config(transform_output="pandas")   # ← LightGBM uyarılarını çözer
    
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    
    num_pipe = Pipeline(steps=steps)
    
    return ColumnTransformer(
        transformers=[("num", num_pipe, feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model, preprocess: ColumnTransformer) -> Pipeline:
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def ensure_cyclical_time_features(
    df: pd.DataFrame, date_col: Optional[str], datetime_format: Optional[str]
) -> pd.DataFrame:
    needed = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    if all(c in df.columns for c in needed):
        return df
    if not date_col or date_col not in df.columns:
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce", 
                       format=datetime_format if datetime_format else None)
    if dt.isna().all():
        return df

    out = df.copy()
    out["_dt"] = dt
    out = out.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

    hour = out["_dt"].dt.hour.astype(float)
    doy = out["_dt"].dt.dayofyear.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    
    return out.drop(columns=["_dt"])  # ← Temizleme eklendi


# ==================== CSV OKUMA FONKSİYONLARI (ÖNEMLİ!) ====================

def _read_csv_buffer_with_fallbacks(source: Any, sep: str = ",", **read_csv_kwargs) -> Tuple[pd.DataFrame, str]:
    """Read CSV with encoding fallbacks."""
    last_error: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "cp1254", "latin1"):
        try:
            if hasattr(source, "read"):
                if hasattr(source, "seek"):
                    source.seek(0)
                return pd.read_csv(source, sep=sep, encoding=enc, **read_csv_kwargs), enc
            return pd.read_csv(source, sep=sep, encoding=enc, **read_csv_kwargs), enc
        except UnicodeDecodeError as exc:
            last_error = exc
    
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, sep=sep, **read_csv_kwargs), "auto"
    
    if last_error is not None:
        raise last_error
    raise RuntimeError("CSV could not be read with any configured encoding fallback.")


def _read_csv_with_fallbacks(path: str, sep: str = ",") -> pd.DataFrame:
    """Legacy compatibility function."""
    df, _ = _read_csv_buffer_with_fallbacks(path, sep=sep)
    return df


def read_uploaded_csv_with_fallbacks(uploaded_file: Any, sep: str = ",", **read_csv_kwargs) -> Tuple[pd.DataFrame, str]:
    """Used by Streamlit helpers."""
    return _read_csv_buffer_with_fallbacks(uploaded_file, sep=sep, **read_csv_kwargs)


# ==================== DİĞER FONKSİYONLAR ====================

def detect_zone_column(
    df: pd.DataFrame, 
    target: Optional[str] = None, 
    feature_cols: Optional[List[str]] = None, 
    date_col: Optional[str] = None
) -> Optional[str]:
    excluded = {c for c in [target, date_col] if c}
    if feature_cols:
        excluded.update(feature_cols)
    for col in ZONE_CANDIDATES:
        if col in df.columns:
            nunique = int(df[col].nunique(dropna=True))
            if 2 <= nunique <= 20:
                return col
    for col in df.columns:
        if col in excluded:
            continue
        nunique = int(df[col].nunique(dropna=True))
        if not (2 <= nunique <= 20):
            continue
        if (pd.api.types.is_object_dtype(df[col]) or 
            isinstance(df[col].dtype, pd.CategoricalDtype) or 
            pd.api.types.is_integer_dtype(df[col])):
            return col
    return None


def _sanitize_zone_value(zone_value: Any) -> str:
    raw = str(zone_value) if zone_value == zone_value else 'missing'
    cleaned = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in raw).strip('_')
    return cleaned or 'zone'