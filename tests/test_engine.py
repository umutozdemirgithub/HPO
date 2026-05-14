"""ARD engine unit & integration tests (v1.8)

Run with:
    pytest tests/test_engine.py -v

All tests are self-contained; no external files or network access required.
Heavy integration tests (run_experiment) use a minimal synthetic dataset and
are marked with @pytest.mark.integration so they can be skipped in fast CI:

    pytest tests/test_engine.py -v -m "not integration"
"""

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the project root importable when running from the repo root or tests/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ard_engine as eng


# ===========================================================================
# 1. ParamSpec primitives
# ===========================================================================
class TestParamSpec:
    def test_int_bounds(self):
        spec = eng.INT(2, 8)
        assert spec.kind == "int"
        assert spec.low == 2.0
        assert spec.high == 8.0

    def test_float_bounds(self):
        spec = eng.FLOAT(0.01, 0.5)
        assert spec.kind == "float"

    def test_logfloat_bounds(self):
        spec = eng.LOGFLOAT(1e-4, 1.0)
        assert spec.kind == "logfloat"

    def test_cat_choices(self):
        spec = eng.CAT(["gbtree", "dart"])
        assert spec.choices == ["gbtree", "dart"]


# ===========================================================================
# 2. sample_param — all kinds, seeded RNG
# ===========================================================================
class TestSampleParam:
    def setup_method(self):
        self.rng = np.random.default_rng(0)

    def test_int_in_range(self):
        spec = eng.INT(3, 7)
        for _ in range(100):
            v = eng.sample_param(spec, self.rng)
            assert isinstance(v, int)
            assert 3 <= v <= 7

    def test_float_in_range(self):
        spec = eng.FLOAT(0.0, 1.0)
        for _ in range(100):
            v = eng.sample_param(spec, self.rng)
            assert 0.0 <= v <= 1.0

    def test_logfloat_in_range(self):
        spec = eng.LOGFLOAT(0.001, 1.0)
        for _ in range(100):
            v = eng.sample_param(spec, self.rng)
            assert 0.001 <= v <= 1.0

    def test_cat_in_choices(self):
        spec = eng.CAT(["a", "b", "c"])
        for _ in range(50):
            v = eng.sample_param(spec, self.rng)
            assert v in ["a", "b", "c"]


# ===========================================================================
# 3. clamp_param
# ===========================================================================
class TestClampParam:
    def test_int_clamp_low(self):
        assert eng.clamp_param(eng.INT(2, 8), 1) == 2

    def test_int_clamp_high(self):
        assert eng.clamp_param(eng.INT(2, 8), 99) == 8

    def test_float_clamp(self):
        assert eng.clamp_param(eng.FLOAT(0.0, 1.0), 1.5) == pytest.approx(1.0)

    def test_cat_fallback(self):
        spec = eng.CAT(["x", "y"])
        assert eng.clamp_param(spec, "z") == "x"

    def test_cat_valid_passthrough(self):
        spec = eng.CAT(["x", "y"])
        assert eng.clamp_param(spec, "y") == "y"


# ===========================================================================
# 4. encode / decode round-trip
# ===========================================================================
class TestEncodeDecodeSingleParam:
    def _roundtrip(self, spec, value):
        space = {"p": spec}
        params = {"p": value}
        vec = eng.encode(space, params)
        recovered = eng.decode(space, vec)
        return recovered["p"]

    def test_int_roundtrip(self):
        assert self._roundtrip(eng.INT(1, 10), 5) == 5

    def test_float_roundtrip(self):
        assert self._roundtrip(eng.FLOAT(0.0, 1.0), 0.3) == pytest.approx(0.3, rel=1e-9)

    def test_logfloat_roundtrip(self):
        # logfloat: encode goes to log-space, decode comes back
        val = self._roundtrip(eng.LOGFLOAT(0.01, 1.0), 0.05)
        assert val == pytest.approx(0.05, rel=1e-9)

    def test_cat_roundtrip(self):
        assert self._roundtrip(eng.CAT(["a", "b", "c"]), "b") == "b"

    def test_multi_param_roundtrip(self):
        space = {
            "n": eng.INT(1, 20),
            "lr": eng.LOGFLOAT(0.001, 0.3),
            "subsample": eng.FLOAT(0.5, 1.0),
            "booster": eng.CAT(["gbtree", "dart"]),
        }
        params = {"n": 10, "lr": 0.05, "subsample": 0.8, "booster": "dart"}
        vec = eng.encode(space, params)
        recovered = eng.decode(space, vec)
        assert recovered["n"] == 10
        assert recovered["lr"] == pytest.approx(0.05, rel=1e-6)
        assert recovered["subsample"] == pytest.approx(0.8, rel=1e-9)
        assert recovered["booster"] == "dart"


# ===========================================================================
# 5. _metrics_core
# ===========================================================================
class TestMetricsCore:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = eng._metrics_core(y, y)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert m["mae"] == pytest.approx(0.0, abs=1e-12)
        assert m["r2"] == pytest.approx(1.0, abs=1e-12)

    def test_constant_prediction_r2(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        m = eng._metrics_core(y_true, y_pred)
        assert m["r2"] < 1.0

    def test_mape_zero_denominator(self):
        # near-zero true value: MAPE should be nan, not crash
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 1.0])
        m = eng._metrics_core(y_true, y_pred)
        # mape may be nan or a finite number — should not raise
        assert "mape" in m

    def test_drop_non_finite(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        m = eng._metrics_core(y_true, y_pred, drop_non_finite=True, include_n=True)
        assert m["n"] == 2
        assert m["rmse"] == pytest.approx(0.0, abs=1e-12)

    def test_all_non_finite_returns_nan(self):
        y_true = np.array([np.nan, np.inf])
        y_pred = np.array([1.0, 2.0])
        m = eng._metrics_core(y_true, y_pred, drop_non_finite=True, include_n=True)
        assert m["n"] == 0
        assert math.isnan(m["rmse"])

    def test_bias_sign(self):
        # predictions systematically above truth → positive bias
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        m = eng._metrics_core(y_true, y_pred)
        assert m["bias"] == pytest.approx(1.0)

    def test_nrmse_finite(self):
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([1.0, 9.0])
        m = eng._metrics_core(y_true, y_pred)
        assert math.isfinite(m["nrmse"])


# ===========================================================================
# 6. Objective cache key — determinism & type-independence
# ===========================================================================
class TestObjectiveCacheKey:
    def test_same_params_same_key(self):
        p = {"n": 5, "lr": 0.1, "booster": "gbtree"}
        assert eng.Objective._make_cache_key(p) == eng.Objective._make_cache_key(p)

    def test_int_vs_npint64_same_key(self):
        p_int = {"n": 5}
        p_np = {"n": np.int64(5)}
        assert eng.Objective._make_cache_key(p_int) == eng.Objective._make_cache_key(p_np)

    def test_float_vs_npfloat64_same_key(self):
        p_float = {"lr": 0.1}
        p_np = {"lr": np.float64(0.1)}
        assert eng.Objective._make_cache_key(p_float) == eng.Objective._make_cache_key(p_np)

    def test_different_values_different_key(self):
        p1 = {"n": 5}
        p2 = {"n": 6}
        assert eng.Objective._make_cache_key(p1) != eng.Objective._make_cache_key(p2)

    def test_empty_params(self):
        # should not raise
        key = eng.Objective._make_cache_key({})
        assert isinstance(key, str) and len(key) == 32  # MD5 hex digest length

    def test_key_is_string(self):
        key = eng.Objective._make_cache_key({"n": 3, "lr": 0.05})
        assert isinstance(key, str)


# ===========================================================================
# 7. _holm_bonferroni_adjust
# ===========================================================================
class TestHolmBonferroni:
    def test_single_pvalue_unchanged(self):
        adjusted = eng._holm_bonferroni_adjust([0.03])
        assert adjusted[0] == pytest.approx(0.03)

    def test_all_nan_returns_nan(self):
        adjusted = eng._holm_bonferroni_adjust([float("nan"), float("nan")])
        assert all(math.isnan(x) for x in adjusted)

    def test_monotone_non_decreasing(self):
        raw = [0.001, 0.01, 0.04, 0.20]
        adj = eng._holm_bonferroni_adjust(raw)
        for i in range(len(adj) - 1):
            assert adj[i] <= adj[i + 1] + 1e-12  # adjusted p cannot decrease

    def test_capped_at_1(self):
        adj = eng._holm_bonferroni_adjust([0.5, 0.6, 0.7])
        assert all(x <= 1.0 for x in adj)

    def test_length_preserved(self):
        raw = [0.01, 0.05, 0.2]
        adj = eng._holm_bonferroni_adjust(raw)
        assert len(adj) == 3


# ===========================================================================
# 8. _bootstrap_mean_ci
# ===========================================================================
class TestBootstrapCI:
    def test_ci_contains_mean(self):
        arr = np.linspace(1.0, 10.0, 50)
        lo, hi = eng._bootstrap_mean_ci(arr, n_boot=500, seed=0)
        assert lo <= arr.mean() <= hi

    def test_single_value(self):
        lo, hi = eng._bootstrap_mean_ci(np.array([5.0]), n_boot=100)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_empty_returns_nan(self):
        lo, hi = eng._bootstrap_mean_ci(np.array([]))
        assert math.isnan(lo) and math.isnan(hi)

    def test_all_infinite_returns_nan(self):
        lo, hi = eng._bootstrap_mean_ci(np.array([np.inf, -np.inf]))
        assert math.isnan(lo) and math.isnan(hi)

    def test_ci_width_positive(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(5.0, 2.0, 200)
        lo, hi = eng._bootstrap_mean_ci(arr, n_boot=500, seed=0)
        assert hi > lo


# ===========================================================================
# 9. detect_zone_column
# ===========================================================================
class TestDetectZoneColumn:
    def _make_df(self, n=100):
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n, freq="h"),
            "power": rng.normal(100, 10, n),
            "zone": rng.choice(["A", "B", "C"], n),
            "temperature": rng.normal(20, 5, n),
        })

    def test_detects_zone_column(self):
        df = self._make_df()
        result = eng.detect_zone_column(df, target="power", feature_cols=["temperature"], date_col="date")
        assert result == "zone"

    def test_excludes_target(self):
        df = pd.DataFrame({"target": [1, 2, 3], "zone": ["A", "B", "C"]})
        result = eng.detect_zone_column(df, target="target")
        assert result != "target"

    def test_returns_none_when_no_candidate(self):
        df = pd.DataFrame({"x": range(100), "y": range(100)})
        result = eng.detect_zone_column(df)
        assert result is None


# ===========================================================================
# 10. _pareto_front
# ===========================================================================
class TestParetoFront:
    def test_single_point_is_pareto(self):
        mask = eng._pareto_front(np.array([1.0]), np.array([1.0]))
        assert mask[0] is True or mask[0] == True  # noqa: E712

    def test_dominated_point_excluded(self):
        # point 0 dominates point 1 on both objectives
        rmse = np.array([1.0, 2.0])
        rt = np.array([1.0, 2.0])
        mask = eng._pareto_front(rmse, rt)
        assert mask[0] == True   # noqa: E712
        assert mask[1] == False  # noqa: E712

    def test_tradeoff_both_pareto(self):
        # point 0 better rmse, point 1 better runtime → both pareto
        rmse = np.array([0.5, 1.5])
        rt = np.array([10.0, 1.0])
        mask = eng._pareto_front(rmse, rt)
        assert mask[0] == True   # noqa: E712
        assert mask[1] == True   # noqa: E712


# ===========================================================================
# 11. export_latex_table
# ===========================================================================
class TestExportLatexTable:
    def _make_metrics_df(self):
        return pd.DataFrame([
            {"model": "LightGBM", "optimizer": "None", "MAE": 0.5, "RMSE": 0.7, "R2": 0.9, "runtime_s": 1.2},
            {"model": "XGBoost",  "optimizer": "PSO",  "MAE": 0.4, "RMSE": 0.6, "R2": 0.92, "runtime_s": 3.1},
        ])

    def test_creates_file(self, tmp_path):
        df = self._make_metrics_df()
        out = str(tmp_path / "table.tex")
        eng.export_latex_table(df, out)
        assert Path(out).exists()

    def test_contains_tabular(self, tmp_path):
        df = self._make_metrics_df()
        out = str(tmp_path / "table.tex")
        eng.export_latex_table(df, out)
        content = Path(out).read_text()
        assert r"\begin{tabular}" in content
        assert r"\end{table}" in content

    def test_missing_column_raises_keyerror(self, tmp_path):
        df = pd.DataFrame([{"model": "X", "optimizer": "None"}])
        with pytest.raises(KeyError):
            eng.export_latex_table(df, str(tmp_path / "t.tex"))


# ===========================================================================
# 12. _normalize_for_json (app_streamlit helper)
# ===========================================================================
class TestNormalizeForJson:
    # Import the function from the UI module without triggering Streamlit
    @pytest.fixture(autouse=True)
    def _patch_streamlit(self, monkeypatch):
        """Stub out streamlit so app_streamlit.py can be imported in tests."""
        st_stub = types.ModuleType("streamlit")
        for attr in ["cache_data", "set_page_config", "sidebar", "stop",
                     "title", "markdown", "info", "error", "warning",
                     "success", "caption", "columns", "metric", "tabs",
                     "expander", "multiselect", "number_input", "checkbox",
                     "slider", "selectbox", "button", "download_button",
                     "file_uploader", "dataframe", "divider", "subheader",
                     "plotly_chart", "image", "write", "rerun", "container",
                     "header", "text_input", "progress"]:
            setattr(st_stub, attr, lambda *a, **kw: None)
        st_stub.session_state = {}
        st_stub.cache_data = lambda *a, **kw: (lambda f: f)
        monkeypatch.setitem(sys.modules, "streamlit", st_stub)
        # Also stub plotly to avoid import errors
        for mod in ["plotly", "plotly.express", "plotly.graph_objects",
                    "plotly.subplots"]:
            monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))

    @pytest.fixture
    def fn(self):
        import importlib
        m = importlib.import_module("app_streamlit")
        return m._normalize_for_json

    def test_int(self, fn):
        assert fn(np.int64(7)) == 7
        assert isinstance(fn(np.int64(7)), int)

    def test_float(self, fn):
        assert fn(np.float32(3.14)) == pytest.approx(3.14, rel=1e-3)

    def test_bool(self, fn):
        assert fn(np.bool_(True)) is True
        assert fn(np.bool_(False)) is False

    def test_ndarray(self, fn):
        result = fn(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_ndarray_2d(self, fn):
        result = fn(np.array([[1, 2], [3, 4]]))
        assert result == [[1, 2], [3, 4]]

    def test_timestamp(self, fn):
        ts = pd.Timestamp("2024-01-15 12:00:00")
        result = fn(ts)
        assert "2024-01-15" in result

    def test_nested_dict(self, fn):
        val = {"a": np.int64(1), "b": [np.float32(0.5)]}
        result = fn(val)
        assert result["a"] == 1
        assert isinstance(result["b"][0], float)

    def test_passthrough_str(self, fn):
        assert fn("hello") == "hello"

    def test_passthrough_none(self, fn):
        assert fn(None) is None

    def test_json_serialisable(self, fn):
        # everything coming out must be JSON-serialisable
        val = {"arr": np.array([1.0, 2.0]), "n": np.int64(5), "flag": np.bool_(True)}
        result = fn(val)
        json.dumps(result)  # should not raise


# ===========================================================================
# 13. Integration: run_experiment on synthetic data
# ===========================================================================
@pytest.mark.integration
class TestRunExperimentIntegration:
    @pytest.fixture
    def synthetic_df(self):
        rng = np.random.default_rng(0)
        n = 120
        dates = pd.date_range("2022-01-01", periods=n, freq="h")
        x1 = rng.normal(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + x2 + rng.normal(0, 0.1, n)
        return pd.DataFrame({"date": dates, "x1": x1, "x2": x2, "target": y})

    def test_returns_dict(self, synthetic_df, tmp_path):
        result = eng.run_experiment_safe(
            df=synthetic_df,
            feature_cols=["x1", "x2"],
            target_cols=["target"],
            date_column="date",
            folds=3,
            seed=42,
            iters=3,
            models_selected=["HistGB"],
            optimizers_selected=["None"],
            with_shap=False,
            generate_figures=False,
            compute_diagnostics=False,
            write_metadata=False,
            out_dir=str(tmp_path),
        )
        assert isinstance(result, dict)
        assert "error" not in result

    def test_metrics_df_has_required_columns(self, synthetic_df, tmp_path):
        result = eng.run_experiment_safe(
            df=synthetic_df,
            feature_cols=["x1", "x2"],
            target_cols=["target"],
            date_column="date",
            folds=3,
            seed=42,
            iters=3,
            models_selected=["HistGB"],
            optimizers_selected=["None"],
            with_shap=False,
            generate_figures=False,
            compute_diagnostics=False,
            write_metadata=False,
            out_dir=str(tmp_path),
        )
        df = result["metrics_df"]
        for col in ["model", "optimizer", "rmse", "mae", "r2"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_rmse_positive(self, synthetic_df, tmp_path):
        result = eng.run_experiment_safe(
            df=synthetic_df,
            feature_cols=["x1", "x2"],
            target_cols=["target"],
            date_column="date",
            folds=3,
            seed=42,
            iters=3,
            models_selected=["HistGB"],
            optimizers_selected=["None"],
            with_shap=False,
            generate_figures=False,
            compute_diagnostics=False,
            write_metadata=False,
            out_dir=str(tmp_path),
        )
        rmse = float(result["metrics_df"]["rmse"].iloc[0])
        assert rmse > 0.0
        assert math.isfinite(rmse)

    def test_nested_cv_runs(self, synthetic_df, tmp_path):
        result = eng.run_experiment_safe(
            df=synthetic_df,
            feature_cols=["x1", "x2"],
            target_cols=["target"],
            date_column="date",
            folds=3,
            seed=42,
            iters=2,
            nested_cv=True,
            models_selected=["HistGB"],
            optimizers_selected=["None", "RandomSearch"],
            with_shap=False,
            generate_figures=False,
            compute_diagnostics=False,
            write_metadata=False,
            out_dir=str(tmp_path),
        )
        assert "error" not in result
        assert len(result["metrics_df"]) == 2  # HistGB × {None, RandomSearch}

    def test_invalid_model_raises(self, synthetic_df, tmp_path):
        result = eng.run_experiment_safe(
            df=synthetic_df,
            feature_cols=["x1", "x2"],
            target_cols=["target"],
            date_column="date",
            folds=3,
            seed=42,
            iters=2,
            models_selected=["NonExistentModel"],
            optimizers_selected=["None"],
            with_shap=False,
            generate_figures=False,
            compute_diagnostics=False,
            write_metadata=False,
            out_dir=str(tmp_path),
        )
        # run_experiment_safe must return dict with "error", never raise
        assert "error" in result

    def test_output_files_written(self, synthetic_df, tmp_path):
        eng.run_experiment_safe(
            df=synthetic_df,
            feature_cols=["x1", "x2"],
            target_cols=["target"],
            date_column="date",
            folds=3,
            seed=42,
            iters=2,
            models_selected=["HistGB"],
            optimizers_selected=["None"],
            with_shap=False,
            generate_figures=False,
            compute_diagnostics=True,
            write_metadata=True,
            out_dir=str(tmp_path),
        )
        # A run directory should have been created under tmp_path
        run_dirs = list(tmp_path.glob("run_*"))
        assert len(run_dirs) >= 1
        run_dir = run_dirs[0]
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "run_metadata.json").exists()


# ===========================================================================
# 14. _prepare_dataset — unit tests (no CV, no fit)
# ===========================================================================
class TestPrepareDataset:
    @pytest.fixture
    def base_df(self):
        rng = np.random.default_rng(1)
        n = 80
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n, freq="h"),
            "x1": rng.normal(0, 1, n),
            "x2": rng.uniform(0, 1, n),
            "y": rng.normal(5, 1, n),
        })

    def test_returns_correct_shapes(self, base_df):
        logs = []
        df2, y, X, feat_cols = eng._prepare_dataset(
            df=base_df, feature_cols=["x1", "x2"], target_cols=["y"],
            date_column="date", datetime_format=None, include_cyclical=False,
            time_series_cv=True, strict_time_order=False, log=logs.append,
        )
        assert len(df2) == len(base_df)
        assert y.shape == (len(base_df),)
        assert list(X.columns) == ["x1", "x2"]
        assert feat_cols == ["x1", "x2"]

    def test_sorts_chronologically(self, base_df):
        # Shuffle the dataframe first
        shuffled = base_df.sample(frac=1, random_state=0).reset_index(drop=True)
        logs = []
        df2, _, _, _ = eng._prepare_dataset(
            df=shuffled, feature_cols=["x1", "x2"], target_cols=["y"],
            date_column="date", datetime_format=None, include_cyclical=False,
            time_series_cv=True, strict_time_order=False, log=logs.append,
        )
        dates = pd.to_datetime(df2["date"])
        assert (dates.diff().dropna() >= pd.Timedelta(0)).all(), "Data not sorted chronologically"

    def test_strict_time_order_raises_without_date(self, base_df):
        df_no_date = base_df.drop(columns=["date"])
        logs = []
        with pytest.raises(ValueError, match="strict_time_order"):
            eng._prepare_dataset(
                df=df_no_date, feature_cols=["x1", "x2"], target_cols=["y"],
                date_column="date", datetime_format=None, include_cyclical=False,
                time_series_cv=True, strict_time_order=True, log=logs.append,
            )

    def test_no_target_raises(self, base_df):
        logs = []
        with pytest.raises(ValueError, match="target"):
            eng._prepare_dataset(
                df=base_df, feature_cols=["x1", "x2"], target_cols=[],
                date_column="date", datetime_format=None, include_cyclical=False,
                time_series_cv=False, strict_time_order=False, log=logs.append,
            )

    def test_warns_without_date_col(self, base_df):
        df_no_date = base_df.drop(columns=["date"])
        logs = []
        eng._prepare_dataset(
            df=df_no_date, feature_cols=["x1", "x2"], target_cols=["y"],
            date_column=None, datetime_format=None, include_cyclical=False,
            time_series_cv=True, strict_time_order=False, log=logs.append,
        )
        assert any("temporal leakage" in str(m) for m in logs), "Expected leakage warning"


# ===========================================================================
# 15. _build_ensemble_comparison — unit tests
# ===========================================================================
class TestBuildEnsembleComparison:
    def _make_inputs(self, n=60, n_methods=3):
        rng = np.random.default_rng(42)
        y = rng.normal(10, 2, n)
        predictions = {f"ModelA__opt{i}": (y + rng.normal(0, 0.5, n)).tolist() for i in range(n_methods)}
        metrics_df = pd.DataFrame([
            {"model": "ModelA", "optimizer": f"opt{i}", "rmse": 0.5 + i * 0.1, "mae": 0.4}
            for i in range(n_methods)
        ])
        return y, predictions, metrics_df

    def test_writes_csv(self, tmp_path):
        y, predictions, metrics_df = self._make_inputs()
        eng._build_ensemble_comparison(y, predictions, metrics_df, str(tmp_path), print)
        assert (tmp_path / "metrics_ensemble_comparison.csv").exists()

    def test_csv_has_rows(self, tmp_path):
        y, predictions, metrics_df = self._make_inputs(n_methods=5)
        eng._build_ensemble_comparison(y, predictions, metrics_df, str(tmp_path), print)
        df = pd.read_csv(tmp_path / "metrics_ensemble_comparison.csv")
        assert len(df) >= 3  # single best + top3 + top5

    def test_single_method_no_crash(self, tmp_path):
        y, predictions, metrics_df = self._make_inputs(n_methods=1)
        eng._build_ensemble_comparison(y, predictions, metrics_df, str(tmp_path), print)
        assert (tmp_path / "metrics_ensemble_comparison.csv").exists()


# ===========================================================================
# 16. _write_run_artifacts — unit tests
# ===========================================================================
class TestWriteRunArtifacts:
    def _make_inputs(self, n=40):
        rng = np.random.default_rng(0)
        y = rng.normal(5, 1, n)
        metrics_df = pd.DataFrame([{"model": "HistGB", "optimizer": "None", "rmse": 0.5, "mae": 0.4, "r2": 0.9}])
        predictions = {"HistGB__None": y.tolist()}
        best_params_report = {"HistGB__None": {"selected_params": {}, "nested_cv": True, "selection_rule": "aggregate_median_mode"}}
        fold_tables = {"HistGB__None": pd.DataFrame([{"fold": 0, "rmse": 0.5, "mae": 0.4, "r2": 0.9}])}
        return y, metrics_df, predictions, best_params_report, fold_tables

    def test_metrics_csv_written(self, tmp_path):
        y, metrics_df, preds, bpr, ft = self._make_inputs()
        eng._write_run_artifacts(str(tmp_path), y, metrics_df, preds, bpr, ft, True, print)
        assert (tmp_path / "metrics.csv").exists()

    def test_oof_predictions_written(self, tmp_path):
        y, metrics_df, preds, bpr, ft = self._make_inputs()
        eng._write_run_artifacts(str(tmp_path), y, metrics_df, preds, bpr, ft, False, print)
        assert (tmp_path / "oof_predictions.csv").exists()

    def test_best_params_json_written(self, tmp_path):
        y, metrics_df, preds, bpr, ft = self._make_inputs()
        eng._write_run_artifacts(str(tmp_path), y, metrics_df, preds, bpr, ft, False, print)
        assert (tmp_path / "best_params.json").exists()

    def test_fold_stability_written_when_diagnostics_on(self, tmp_path):
        y, metrics_df, preds, bpr, ft = self._make_inputs()
        eng._write_run_artifacts(str(tmp_path), y, metrics_df, preds, bpr, ft, True, print)
        assert (tmp_path / "fold_stability_summary.csv").exists()

    def test_fold_stability_skipped_when_diagnostics_off(self, tmp_path):
        y, metrics_df, preds, bpr, ft = self._make_inputs()
        eng._write_run_artifacts(str(tmp_path), y, metrics_df, preds, bpr, ft, False, print)
        assert not (tmp_path / "fold_stability_summary.csv").exists()

# ===========================================================================
# 14. Paper-grade reliability regression tests
# ===========================================================================
class TestPaperGradeReliabilityFixes:
    def test_prepare_dataset_removes_date_column_from_features(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=12, freq="h"),
            "x1": np.arange(12, dtype=float),
            "target": np.arange(12, dtype=float) * 2.0,
        })
        logs = []
        _, _, X, effective = eng._prepare_dataset(
            df=df,
            feature_cols=["date", "x1"],
            target_cols=["target"],
            date_column="date",
            datetime_format=None,
            include_cyclical=False,
            time_series_cv=True,
            strict_time_order=True,
            log=logs.append,
        )
        assert "date" not in effective
        assert list(X.columns) == ["x1"]
        assert any("removed from feature_cols" in msg for msg in logs)

    def test_strict_time_order_rejects_unparseable_datetime_rows(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "bad-date", "2024-01-03"],
            "x1": [1.0, 2.0, 3.0],
            "target": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError, match="could not be parsed"):
            eng._prepare_dataset(
                df=df,
                feature_cols=["x1"],
                target_cols=["target"],
                date_column="date",
                datetime_format=None,
                include_cyclical=False,
                time_series_cv=True,
                strict_time_order=True,
                log=lambda _: None,
            )

    @pytest.mark.integration
    def test_multi_zone_reports_too_few_rows(self, tmp_path):
        n_big, n_small = 14, 5
        dates = pd.date_range("2024-01-01", periods=n_big + n_small, freq="h")
        df = pd.DataFrame({
            "date": dates,
            "zone": ["A"] * n_big + ["B"] * n_small,
            "x1": np.arange(n_big + n_small, dtype=float),
            "target": np.arange(n_big + n_small, dtype=float) + 1.0,
        })
        res = eng.run_experiment_safe(
            df=df,
            feature_cols=["x1"],
            target_cols=["target"],
            date_column="date",
            datetime_format=None,
            folds=2,
            seed=42,
            iters=1,
            models_selected=["HistGB"],
            optimizers_selected=["None"],
            zone_column="zone",
            include_cyclical=False,
            time_series_cv=True,
            nested_cv=True,
            strict_time_order=True,
            deterministic=True,
            generate_figures=False,
            compute_diagnostics=False,
            write_metadata=True,
            with_shap=False,
            shap_rows=100,
            out_dir=str(tmp_path),
        )
        assert "error" not in res
        failed = res.get("failed_zones", [])
        assert any(z.get("error") == "TOO_FEW_ROWS" and z.get("zone") == "B" for z in failed)
        assert (Path(res["out_dir"]) / "failed_zones.csv").exists()

class TestMultiSeedMode:
    def test_parse_seed_list_accepts_common_formats(self):
        assert eng.parse_seed_list("42, 101; 202 303", 42) == [42, 101, 202, 303]
        assert eng.parse_seed_list([42, "42", 7], 42) == [42, 7]
        assert eng.parse_seed_list(None, 99) == [99]

    @pytest.mark.integration
    def test_multi_seed_outputs_mean_std_tables(self, tmp_path):
        n = 18
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x1": np.linspace(0, 1, n),
            "target": np.linspace(1, 3, n),
        })
        res = eng.run_experiment_safe(
            df=df,
            feature_cols=["x1"],
            target_cols=["target"],
            date_column="date",
            datetime_format=None,
            folds=2,
            seed=42,
            seeds=[42, 101],
            multi_seed=True,
            iters=1,
            models_selected=["HistGB"],
            optimizers_selected=["None"],
            include_cyclical=False,
            time_series_cv=True,
            nested_cv=True,
            strict_time_order=True,
            deterministic=True,
            generate_figures=False,
            compute_diagnostics=False,
            write_metadata=True,
            with_shap=False,
            shap_rows=100,
            out_dir=str(tmp_path),
        )
        assert "error" not in res
        assert res.get("multi_seed_mode") is True
        out_dir = Path(res["out_dir"])
        assert (out_dir / "per_seed_metrics.csv").exists()
        assert (out_dir / "multi_seed_summary.csv").exists()
        assert (out_dir / "multi_seed_best_methods.csv").exists()
        summary = pd.read_csv(out_dir / "multi_seed_summary.csv")
        assert "RMSE" in summary.columns
        assert "RMSE_std" in summary.columns
        assert int(summary.loc[0, "n_seeds"]) == 2
