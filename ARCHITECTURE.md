# ARD Architecture Notes

This revision focuses on maintainability without breaking the current public API.

## What changed

- `ard_engine.py` is now a **compatibility shim**.
  - The engine implementation lives in `ard/engine_core.py`.
  - Existing imports such as `import ard_engine as eng` continue to work.
- `run_cli.py` is now a **thin wrapper** around `ard/cli.py`.
- Streamlit helper functions were extracted into `ard/streamlit_helpers.py`.
  - This removes several hundred lines of implementation detail from `app_streamlit.py`.
  - The UI entrypoint now focuses more on orchestration.

## Resulting structure

```text
ard/
  __init__.py
  cli.py
  engine_core.py
  streamlit_helpers.py
ard_engine.py          # compatibility shim
app_streamlit.py       # Streamlit entrypoint
run_cli.py             # compatibility wrapper
tests/
```

## Why this is better

- Lower coupling between entrypoints and implementation details
- Easier future extraction of engine subdomains such as plotting, SHAP, and artifact writing
- Preserves backward compatibility for tests, scripts, and notebooks
- Makes incremental refactoring safer

## Next refactor candidates

- Split `ard/engine_core.py` into:
  - `data_io.py`
  - `metrics.py`
  - `optimizers.py`
  - `artifacts.py`
  - `shap_utils.py`
  - `experiment_runner.py`
- Split `app_streamlit.py` further into:
  - sidebar config
  - result rendering
  - SHAP views
  - Wilcoxon views
  - diagnostics views
