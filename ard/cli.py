#!/usr/bin/env python3
"""ARD CLI runner (v1.8)

This CLI is intentionally thin: it delegates all core work to `ard_engine.py`
and then exports a LaTeX table for convenient paper integration.

Outputs (under --out):
- metrics.csv
- metrics_ensemble_comparison.csv
- best_params.json
- run_metadata.json
- fold_stability_summary.csv
- figures/ (time-series + diagnostics + fold stability + convergence)
- shap/<method_key>/figures/ (if SHAP is enabled and available)
- metrics_table.tex (IEEE-style table)

Example:
  python run_cli.py --data solar_dataset.csv --target POWER --date_col Date \
    --features TCWL TCIW SP HUM TCC U V TEMP SSRD STRD TSR TP \
    --models LightGBM XGBoost HistGB --optimizers None PSO GA ABC \
    --folds 3 --iters 40 --zone_col ZONE --shap_rows 300 --out ard_outputs
"""

import argparse
from pathlib import Path
from ard import engine_core as eng
from .errors import format_error_result

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV/XLSX/MAT dataset.")
    ap.add_argument("--target", required=True, help="Target column name (Y).")
    ap.add_argument("--features", nargs="+", required=True, help="Feature column names (X).")
    ap.add_argument("--date_col", default=None, help="Datetime column for time-series ordering (optional).")
    ap.add_argument("--datetime_format", default=None, help="Optional datetime parsing format string passed to pandas.")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter override for CSV/TXT input (examples: , ; | \t).")
    ap.add_argument("--strict_time_order", action="store_true", default=False,help="If set, raise an error when time_series_cv=True but date_col is missing/invalid.",)
    ap.add_argument("--deterministic", action="store_true", default=False,
        help="Best-effort determinism: fixes RNG seeds and forces common numeric backends to single-thread.",)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--time_series_cv", dest="time_series_cv", action="store_true", default=True,help="Use TimeSeriesSplit (default: True).",)
    ap.add_argument("--no_time_series_cv", dest="time_series_cv", action="store_false", help="Disable TimeSeriesSplit and use shuffled KFold instead.",)
    ap.add_argument("--nested_cv", dest="nested_cv", action="store_true", default=True, help="Use nested CV for paper-grade unbiased HPO evaluation (default).")
    ap.add_argument("--no_nested_cv", dest="nested_cv", action="store_false", help="Disable nested CV for quick exploratory runs only.")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None, help="Optional multi-seed list, e.g. '42,101,202'. Enables repeated evaluation and mean±std summaries when more than one seed is provided.")
    ap.add_argument("--multi_seed", action="store_true", default=False, help="Force multi-seed mode even if --seeds contains one seed.")
    ap.add_argument("--zone_col", default=None, help="Optional zone/group column. If provided, each zone is evaluated separately and aggregate multi-zone metrics are written.")
    ap.add_argument("--models", nargs="+", default=["LightGBM", "XGBoost", "HistGB"])
    ap.add_argument("--optimizers", nargs="+", default=["None", "PSO", "GA", "ABC"])
    ap.add_argument("--compute_shap", action="store_true", default=False, help="Enable XAI artifact generation. Use --xai_methods to choose SHAP/PFI/LIME.")
    ap.add_argument("--xai_methods", nargs="+", default=["SHAP", "PFI", "LIME"], choices=["SHAP", "PFI", "LIME"], help="XAI methods to compute when --compute_shap is enabled.")
    ap.add_argument("--no_figures", action="store_true", default=False,help="Disable figure generation (faster; recommended for CI).")
    ap.add_argument("--no_diagnostics", action="store_true", default=False,help="Disable diagnostics tables (faster).")
    ap.add_argument("--no_metadata", action="store_true", default=False,help="Disable run_metadata.json writing.")
    ap.add_argument("--shap_rows", type=int, default=300)
    ap.add_argument("--out", default="ard_outputs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = eng.run_experiment_from_path(
        data_path=args.data,
        target=args.target,
        features=args.features,
        date_col=args.date_col,
        delimiter=("\t" if args.delimiter == r"\t" else args.delimiter),
        datetime_format=args.datetime_format,
        folds=int(args.folds),
        time_series_cv=bool(args.time_series_cv),
        nested_cv=bool(args.nested_cv),
        iters=int(args.iters),
        seed=int(args.seed),
        seeds=args.seeds,
        multi_seed=bool(args.multi_seed or args.seeds),
        models_selected=args.models,
        optimizers_selected=args.optimizers,
        with_shap=bool(args.compute_shap),
        xai_methods=list(args.xai_methods),
        shap_rows=int(args.shap_rows),
        strict_time_order=bool(args.strict_time_order),
        deterministic=bool(args.deterministic),
        generate_figures=not bool(args.no_figures),
        compute_diagnostics=not bool(args.no_diagnostics),
        write_metadata=not bool(args.no_metadata),
        out_dir=str(out_dir),
        zone_column=args.zone_col,
    )

    if "error" in res:
        print("ERROR:", format_error_result(res))
        if res.get("traceback"):
            print(res["traceback"])
        raise SystemExit(1)

    tex_path = Path(res["out_dir"]) / "metrics_table.tex"
    try:
        eng.export_latex_table(res["metrics_df"], str(tex_path))
        print(f"LaTeX table written to: {tex_path}")
    except KeyError as exc:
        print(f"WARNING: LaTeX table skipped — missing column in metrics_df: {exc}")
    except Exception as exc:
        print(f"WARNING: LaTeX table export failed: {exc}")

    print("Done. Outputs in:", Path(res["out_dir"]).resolve())

if __name__ == "__main__":
    main()
