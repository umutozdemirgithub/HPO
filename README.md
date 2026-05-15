# Advanced Research Dashboard (ARD)

Advanced Research Dashboard (ARD) is a Streamlit-based experiment console for time-aware regression benchmarking, nested cross-validation, optimizer comparison, visual diagnostics, and SHAP-based explainability.

## What ARD does

- Enforces **chronological evaluation** with `TimeSeriesSplit`
- Supports **nested CV** for less biased hyperparameter tuning
- Benchmarks multiple algorithms and optimizers in one run
- Produces saved artifacts under `ard_outputs/`
- Surfaces fold-level metrics, visual analytics, Wilcoxon comparisons, and SHAP summaries
- Includes a live **Process Logs** panel in the UI
- Reuses identical experiment results via **Streamlit caching** when enabled

## Supported inputs

ARD accepts the following dataset types:

- `.csv`
- `.txt`
- `.xlsx`
- `.mat`

The app automatically:

- detects CSV delimiters when possible
- attempts datetime parsing for time-like columns
- converts object columns to numeric when conversion is reliable
- removes all-null columns
- replaces infinities with `NaN`

## File-size guardrails

The Streamlit UI enforces an in-app upload limit of **100 MB**.

If you want Streamlit itself to allow larger uploads, add this file:

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 100
```

Increase the value only if your deployment environment has enough memory.

## Main files

- `app_streamlit.py` — Streamlit UI
- `ard_engine.py` — experiment engine
- `ard_outputs/` — generated run artifacts

## Updated project structure

This revision introduces a small internal package to reduce top-level module sprawl while preserving backward compatibility:

- `ard/engine_core.py` — engine implementation
- `ard/cli.py` — CLI implementation
- `ard/streamlit_helpers.py` — extracted Streamlit helpers
- `ard_engine.py` — compatibility shim for existing imports
- `run_cli.py` — compatibility wrapper for the CLI entrypoint

See `ARCHITECTURE.md` for the refactor rationale and next-step decomposition plan.

## Key runtime behavior

### Import diagnostics
The app initializes `_IMPORT_ERRORS` before optional imports and exposes import problems in the sidebar diagnostics panel.

### Cached data loading
Dataset loading is cached with `st.cache_data` so repeated reruns with the same uploaded file avoid reparsing overhead.

### Cached experiment reruns
ARD includes a cache-aware execution mode for identical:

- dataset contents
- feature/target selection
- model selection
- optimizer selection
- CV configuration
- SHAP settings

When enabled, repeated runs with the same inputs can reuse previously generated outputs instead of recomputing the entire experiment.

## How to run

```bash
python -m streamlit run app_streamlit.py
```

Running Streamlit through the same interpreter you use for package installation reduces environment mismatch issues.

## Recommended Python packages

Install the core stack first:

```bash
pip install streamlit pandas numpy scipy scikit-learn plotly openpyxl
```

Optional modeling and explainability packages:

```bash
pip install xgboost lightgbm shap optuna
```

If an optional dependency is unavailable, ARD will report it in the sidebar diagnostics.

## Output structure

Typical output directories are created under:

```text
ard_outputs/
```

Each run may contain:

- metrics tables
- optimizer summaries
- fold-level diagnostics
- figures
- SHAP artifacts
- Wilcoxon comparison files
- metadata snapshots

## SHAP explainability — methodology note

SHAP values in ARD are computed on a model **retrained on the full dataset** after nested CV completes. This is intentional:

- **Nested CV metrics** measure generalisation (no leakage, held-out folds).
- **SHAP values** explain what the final deployable model has learned from all available data.

These are two complementary views. Consumers citing SHAP results in publications should note that the explainer model is a full-data retrain, not a per-fold estimator. The `run_metadata.json` artifact records `shap_model_scope: "full_data_retrain"` for traceability.

## Notes for maintainers

- Keep naming consistent with **ARD** across UI text, documentation, and artifact exports.
- Avoid reintroducing legacy project labels in README text, result exports, or UI copy.
- Prefer caching for deterministic data prep and result reuse, but keep live callback-heavy execution paths available when a fresh trace is needed.

## Changelog

### v1.7 (current)
- **Fix:** `_normalize_for_json` now handles `np.ndarray` values, preventing `TypeError` during JSON serialisation of array-valued metadata fields.
- **Fix:** CLI `export_latex_table` call is now wrapped in `try/except`; a missing column in `metrics_df` no longer crashes the process — a warning is printed and execution continues.
- **Fix:** Baseline (`optimizer=None`) non-nested CV path no longer triggers a redundant `Objective.evaluate()` call before `eval_cv_with_oof`. History is now derived directly from the CV mean RMSE, eliminating one unnecessary full cross-validation pass.
- **Docs:** SHAP full-data retrain behaviour is now explicitly documented in code comments, `run_metadata.json` (`shap_model_scope` field), log output, and this README.

## v1.9.1 hardening updates

- Refactored `app_streamlit.py` to be import-safe by moving UI bootstrapping into `main()`.
- This eliminates import-time side effects and allows unit tests to load UI helpers without executing the full Streamlit app.
- Fast test suite status after the refactor: `73 passed, 6 deselected` with `pytest -m "not integration"`.


## v1.10 paper-alignment patch

- Nested-CV optimizer convergence now stores each outer fold's real optimizer iteration history (`optimizer_history_per_outer_fold`) and exports an averaged `convergence_history` for plotting, instead of using only one best score per outer fold.
- CLI default optimizer budget is aligned with the paper setting: `--iters` now defaults to `40`.
- CLI supports explicit multi-zone reproduction through `--zone_col <COLUMN_NAME>`. When provided, the engine runs zone-wise evaluation and writes aggregate multi-zone outputs.

Example paper-style CLI run:

```bash
python run_cli.py --data solar_dataset.csv --target POWER --date_col Date.1 \
  --features TCWL TCIW SP HUM TCC U V TEMP SSRD STRD TSR TP \
  --models LightGBM XGBoost HistGB --optimizers None PSO GA ABC \
  --folds 3 --iters 40 --nested_cv --zone_col ZONE --compute_shap --shap_rows 387 --out ard_outputs
```

## v1.11 publication-output completion patch

This version adds the remaining paper-facing diagnostic outputs requested for the manuscript package:

- `best_hyperparameters.csv` — clean model/optimizer hyperparameter configuration table exported from `best_params.json`.
- `rank_table.csv` and `critical_difference_rank_table.csv` — method ranking table across RMSE, MAE, R², and runtime.
- `runtime_accuracy_improvement.csv` — default-baseline improvement table with RMSE/MAE/R² deltas and runtime multipliers.
- `ablation_table.csv` — compact ablation summary comparing default baselines with the best HPO configuration per model and the overall best configuration.
- `figures/actual_vs_predicted_best.png` plus method-specific actual-vs-predicted scatter plots.
- `figures/fold_zone_rmse_heatmap_best.png` plus method-specific `fold_zone_rmse_heatmap_*.png` and CSV, when multi-zone mode is used.
- Guaranteed SHAP dependence plots for `SSRD` and `STRD` are now attempted as `shap_dependence_SSRD.png` and `shap_dependence_STRD.png` whenever those features exist in the transformed feature space.

Together with the earlier outputs (`zone_metrics.csv`, `metrics_macro_zone_mean.csv`, `wilcoxon_results.csv`, residual plots, correlation heatmap, Pareto frontier, and SHAP summary/bar plots), the code now exports the full manuscript-support artifact set.


## Paper-grade reliability updates

This build enforces safer defaults for manuscript experiments:

- Streamlit locked mode uses `TimeSeriesSplit + Nested CV`. The CLI now also defaults to nested CV; use `--no_nested_cv` only for quick exploratory runs.
- The raw datetime/date column is automatically removed from feature columns to prevent temporal leakage and dtype-related failures. Use cyclical features instead when time-of-day/year information is needed.
- `strict_time_order=True` now fails fast when timestamps cannot be parsed, rather than silently sorting invalid timestamps to the end.
- Multi-zone runs report skipped zones in `failed_zones.csv` with `TOO_FEW_ROWS` metadata.
- Shared labels/constants live in `ard/constants.py`; structured engine errors live in `ard/errors.py`.
- `deterministic=True` propagates stricter seed/thread controls to LightGBM and XGBoost where supported.

For final paper runs, recommended settings are `time_series_cv=True`, `nested_cv=True`, `strict_time_order=True`, `deterministic=True`, and explicit model/optimizer selections.

## Reproducible setup additions

This package includes three reproducibility paths:

- `requirements.txt` for flexible installs.
- `requirements-lock.txt` for a pinned paper baseline.
- `environment.yml` for conda/mamba environments.
- `Dockerfile` for containerized Streamlit execution.

Recommended final-paper run settings:

```bash
python run_cli.py --data your_dataset.csv --target POWER --date_col Date \
  --features F1 F2 F3 --models LightGBM XGBoost HistGB \
  --optimizers None RandomSearch TPE GA PSO ABC \
  --folds 3 --iters 40 --nested_cv --strict_time_order --deterministic \
  --compute_shap --shap_rows 800 --out ard_outputs
```

`--deterministic` now propagates a single seed across LightGBM/XGBoost seed aliases and restricts common numeric backends to one thread. Exact bitwise reproducibility can still depend on OS, compiler, BLAS/OpenMP, and package versions; use Docker or `requirements-lock.txt` for the strongest reproducibility.

## v1.12 multi-seed evaluation patch

Multi-seed robustness evaluation has been restored as an explicit optional mode. This is separate from single-seed determinism:

- **Determinism**: rerunning the same seed in the same environment should produce the same best-effort result.
- **Multi-seed evaluation**: reruns the full experiment over multiple seeds and reports method stability as mean ± standard deviation.

### Streamlit usage

In the sidebar under **Experiment Control**:

1. Enable **Multi-seed evaluation**.
2. Enter a seed list such as:

```text
42, 101, 202
```

The app writes a root multi-seed output folder with one subfolder per seed:

```text
ard_outputs/
└── multi_seed_YYYYMMDD_HHMMSS_xxxxxxxx/
    ├── seed_42/
    ├── seed_101/
    ├── seed_202/
    ├── per_seed_metrics.csv
    ├── multi_seed_summary.csv
    ├── multi_seed_best_methods.csv
    └── multi_seed_metadata.json
```

### CLI usage

```bash
python run_cli.py --data your_dataset.csv --target POWER --date_col Date \
  --features F1 F2 F3 \
  --models LightGBM XGBoost HistGB \
  --optimizers None RandomSearch TPE GA \
  --folds 3 --iters 30 --nested_cv --strict_time_order --deterministic \
  --seeds "42,101,202" --out ard_outputs
```

For faster development runs, use fewer seeds and fewer optimizers:

```bash
python run_cli.py --data your_dataset.csv --target POWER --date_col Date \
  --features F1 F2 F3 --models HistGB LightGBM \
  --optimizers None RandomSearch TPE \
  --folds 3 --iters 15 --nested_cv --strict_time_order --deterministic \
  --seeds "42,101" --no_figures --out ard_outputs
```

### New multi-seed artifacts

- `per_seed_metrics.csv`: every seed's full method-level metrics with `seed`, `seed_index`, and `seed_out_dir` columns.
- `multi_seed_summary.csv`: model/optimizer-level mean and standard deviation summaries, including canonical display columns such as `RMSE`, `MAE`, `R2`, plus `RMSE_std`, `MAE_std`, and `R2_std`.
- `multi_seed_best_methods.csv`: top-ranked methods from the multi-seed summary.
- `multi_seed_metadata.json`: seed list, number of completed/failed seeds, reproducibility flags, selected models, selected optimizers, and optional zone metadata.
- `failed_seeds.csv`: written only if one or more seed runs fail while at least one seed succeeds.

Recommended manuscript reporting phrase:

> Experiments were repeated across multiple random seeds and results are reported as mean ± standard deviation across seeds using nested time-series cross-validation.

## Advanced XAI and Visual Analytics Outputs

The current release writes both Streamlit-visible figures and persistent artifacts under the selected `ard_outputs` run folder.

When **Compute XAI explanations** is enabled, the Streamlit sidebar section **2C. Explainable AI Selection** lets you choose any subset of **SHAP**, **PFI**, and **LIME**. Only the selected XAI methods are computed and shown. If all three are selected, each evaluated XAI variant produces:

```text
shap/<method_key>/
  shap_values.npz
  shap_importance.csv
  figures/shap_summary.png
  figures/shap_bar.png
  figures/shap_dependence_*.png
  figures/shap_waterfall_*.png
  figures/shap_decision.png
  figures/shap_force.png

pfi/<method_key>/
  pfi_importance.csv
  pfi_repeats.csv
  pfi_metadata.json
  figures/pfi_bar_mean_std.png
  figures/pfi_repeat_boxplot.png
  figures/pfi_cumulative_importance.png

lime/<method_key>/
  lime_local_explanations.csv
  lime_metadata.json
  lime_example_*.html
  figures/lime_example_*.png
  figures/lime_term_stability.png

xai_comparison/<method_key>/
  xai_agreement.csv
  figures/xai_agreement_grouped_bar.png
  figures/xai_rank_correlation_heatmap.png
```

LIME requires the optional `lime` package, which is included in `requirements.txt`, `requirements-lock.txt`, `pyproject.toml`, and `environment.yml`. If LIME is unavailable in the runtime, the engine does not fail; it writes deterministic LIME-style fallback local explanations under the corresponding `lime/<method_key>/` folder.

Additional predictive and diagnostic figures are also produced in the run-level `figures/` folder:

```text
figures/
  actual_vs_predicted_scatter_*.png
  absolute_error_quantile_*.png
  absolute_error_over_time_*.png
  residual_qq_*.png
  metric_performance_heatmap.png
  top_methods_error_bar.png
  optimizer_convergence_overlay.png
  fold_rmse_heatmap.png
  multi_seed_*.png  # when multi-seed mode is used
```

In Streamlit, these outputs are visible in:

- **Explainable AI (SHAP/LIME/PFI)** → SHAP, PFI, LIME, and SHAP-vs-PFI-vs-LIME agreement tabs.
- **Visual Analytics** → prediction fidelity, residuals, optimizer convergence, metric/fold heatmaps, multi-seed stability, and Wilcoxon tabs.
- **Figure Gallery** → grouped access to all saved PNG artifacts.
- **Artifacts** → a complete file listing for the run folder.

For faster exploratory runs, enable only the XAI methods you need, use `fast` XAI sampling, and keep the XAI row count low. For final paper-ready runs, use `balanced` or `full` sampling and enable multi-seed mode.

The Streamlit XAI sampler is dynamic: changing the sampling mode changes both the default value and the maximum value of the **XAI Sampling Rows** slider, and the upper bound is always capped by the uploaded dataset size.

| Mode | Intended use | Default behavior | Maximum behavior | Runtime note |
|---|---|---|---|---|
| `fast` | quick preview/debug run | small sample, usually around 200 rows | capped around 500 rows and by dataset size | fastest; recommended while tuning models/optimizers |
| `balanced` | recommended normal run | uses the project default XAI row count | capped around 1000 rows and by dataset size | good trade-off for SHAP/PFI/LIME visual stability |
| `full` | final paper artifact run | larger sample, usually around 1200 rows when available | capped around 2000 rows and by dataset size | slowest, especially with multi-seed and LIME |

When multi-seed evaluation is enabled, `full` sampling multiplies XAI runtime by the number of seeds. Use `fast` or `balanced` for exploratory runs, then switch to `full` only for final figures.


### Streamlit XAI method selection

The sidebar separates XAI controls from model/optimizer selection:

```text
2B. Model & Optimizer Selection
  - Algorithms
  - Summary Metrics
  - Meta-Heuristic Optimizers

2C. Explainable AI Selection
  - Compute XAI explanations
  - XAI Methods: SHAP, PFI, LIME
  - XAI Sampling Mode: fast, balanced, full
  - XAI Sampling Rows \(dynamic default and max based on mode + dataset row count\)
```

The row slider is intentionally mode-aware: `fast` constrains the maximum for quick previews, `balanced` raises it for stable routine analysis, and `full` allows larger paper-ready samples. Streamlit also shows contextual warnings when a costly combination is selected, such as `full` XAI sampling in multi-seed mode or large LIME samples.

If **Compute XAI explanations** is disabled, no SHAP/PFI/LIME artifacts are produced. If it is enabled, the engine computes only the selected methods; for example, selecting only **PFI** creates `pfi/` outputs without creating `shap/` or `lime/` outputs. Agreement plots under `xai_comparison/` are generated only when at least two selected XAI methods produce comparable importance data.

### XAI visualization availability note

The XAI panel now always attempts to populate SHAP, PFI, LIME, and SHAP-vs-PFI-vs-LIME sections under the run folder. If the optional `lime` package is not installed in the runtime environment, the engine no longer leaves the LIME panel empty: it writes deterministic **LIME-style fallback local median-sensitivity explanations** to `lime/<method>/`. Install the real LIME dependency with `pip install lime` or `pip install -r requirements.txt` to use the original LIME algorithm.

Some visual outputs are conditional by design:

- Optimizer convergence plots require optimizers with iteration history, such as RandomSearch, TPE, GA, PSO, or ABC; `None` alone has no convergence curve.
- Multi-seed stability plots require multi-seed mode with at least two seeds.
- `None vs HPO` SHAP comparison requires both the baseline optimizer `None` and at least one HPO optimizer for the same model.
- Time-based error plots require a valid date/time column.

## XAI and visual artifact completeness

The Streamlit dashboard and `ard_outputs` now use matplotlib-backed PNG export for the advanced visual analytics panels, so Plotly/Kaleido is **not required** for static figure export. Interactive Plotly charts may still be displayed in Streamlit, but saved PNG artifacts are produced through the engine or matplotlib fallbacks.

Generated artifacts include, when the corresponding analysis is applicable:

- `shap/<method>/figures/`: SHAP summary/beeswarm, bar importance, dependence, waterfall, decision and force plots.
- `shap_comparison/<model>/`: baseline `None` vs best HPO SHAP comparison when both variants were run.
- `pfi/<method>/`: `pfi_importance.csv`, `pfi_repeats.csv`, mean ± std bar plot, repeat boxplot, cumulative importance plot.
- `lime/<method>/`: local explanation PNGs, `lime_local_explanations.csv`, HTML explanations, and term-stability plot. If the optional `lime` package is unavailable, a deterministic LIME-style median-sensitivity fallback is generated so the section is not empty.
- `xai_comparison/<method>/`: `xai_agreement.csv`, normalized importance comparison plot, and rank-correlation heatmap.
- `figures/`: actual-vs-predicted scatter, absolute-error quantile curve, absolute-error over time/order, residual QQ plot, metric performance heatmap, top-methods error bar, optimizer convergence overlay, and fold RMSE heatmap.
- Multi-seed figures are generated only when multi-seed mode is enabled with at least two seeds. In single-seed runs, the dashboard shows a clear not-applicable message instead of silently hiding the section.

Some artifacts are inherently conditional: optimizer convergence requires an optimizer with iteration history; SHAP `None vs HPO` comparison requires both the `None` baseline and at least one HPO optimizer for the same model; multi-seed stability requires two or more seeds.

### Streamlit XAI tab organization

The Streamlit dashboard now groups explainability artifacts into nested tabs for easier review:

- **SHAP**: Global, Local, Dependence, Tables, and model-level Comparison tabs when both baseline and HPO variants are available.
- **PFI**: Importance, Distribution, Cumulative, and Tables tabs.
- **LIME**: Local explanations, Stability, HTML views, and Tables tabs.
- **SHAP vs PFI vs LIME**: Agreement, Importance comparison, Rank correlation, and Tables tabs.

All displayed figures continue to be read from the run folder under `shap/`, `pfi/`, `lime/`, `xai_comparison/`, and `figures/`.


### Multi-seed XAI display in Streamlit

When multi-seed evaluation is enabled, the **Explainable AI (SHAP/LIME/PFI)** page now displays a seed-level tab row above the XAI method tabs. For example:

```text
Seed 42 | Seed 101 | Seed 202
  SHAP | PFI | LIME | SHAP vs PFI vs LIME
```

Each seed tab reads the XAI artifacts from that seed's own output folder, such as `seed_42/shap/`, `seed_42/pfi/`, `seed_42/lime/`, and `seed_42/xai_comparison/`. This keeps single-seed and multi-seed visual inspection consistent while preserving all per-seed files under the main `ard_outputs/multi_seed_*` run directory.

### Streamlit multi-seed visual analytics display

When multi-seed evaluation is enabled, the parent run folder contains aggregate multi-seed summaries while per-seed artifacts are stored under `seed_<value>/` subfolders. The Streamlit result panels now mirror this layout:

- **Visual Analytics** shows a `Multi-seed summary` tab plus one tab per seed, such as `Seed 42`, `Seed 101`, and `Seed 202`.
- **Fold Stability** shows aggregate multi-seed stability first, then per-seed fold tables and figures.
- **Explainable AI (SHAP/LIME/PFI)** shows seed-level tabs above the SHAP, PFI, LIME, and agreement tabs.
- **Figure Gallery** shows both aggregate multi-seed figures and per-seed figure galleries.

This avoids empty parent panels in multi-seed mode because OOF predictions, fold diagnostics, residual plots, SHAP/PFI/LIME artifacts, and optimizer diagnostics are produced inside each seed-specific output folder.

### XAI method comparison toggle

The Streamlit sidebar section **2C. Explainable AI Selection** now separates XAI computation from cross-method comparison. You may compute any subset of SHAP, PFI, and LIME without generating the SHAP/PFI/LIME agreement plots. Enable **Build XAI method-comparison artifacts** only when you need normalized importance comparison, rank-correlation heatmaps, and agreement tables. This is useful for reducing runtime and avoiding unnecessary artifacts in exploratory runs.

In multi-seed mode, Visual Analytics widget keys are scoped per seed output folder so repeated seed tabs can render the same controls without Streamlit duplicate-key errors.

Codex PR test
Codex PR final test
