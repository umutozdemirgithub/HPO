"""Verify ARD runtime dependencies without starting an experiment."""

from __future__ import annotations

import importlib

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "shap": "shap",
    "optuna": "optuna",
    "lime": "lime",
}


def main() -> int:
    missing: list[str] = []
    for package_name, module_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{package_name} ({module_name}): {exc}")

    if missing:
        print("Missing or failed runtime imports:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("All required ARD runtime imports succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
